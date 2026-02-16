"""
FineTuneFlow — SFT Engine.

Provides SFTEngine class for LoRA/QLoRA fine-tuning using:
  - transformers (AutoModelForCausalLM, AutoTokenizer)
  - trl (SFTTrainer)
  - peft (LoraConfig, prepare_model_for_kbit_training)
  - bitsandbytes (BitsAndBytesConfig for 4-bit QLoRA)
  - Redis Pub/Sub for streaming training logs
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import redis
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════

TRAINING_DEFAULTS = {
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "fp16": False,
    "bf16": True,
    "gradient_checkpointing": True,
    "optim": "paged_adamw_8bit",
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 200,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "early_stopping_patience": 3,
}

LORA_DEFAULTS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "auto",
}

QLORA_DEFAULTS = {
    **LORA_DEFAULTS,
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
}

VRAM_TIGHT_ADJUSTMENTS = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 1024,
    "gradient_checkpointing": True,
}

CPU_ADJUSTMENTS = {
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 512,
    "gradient_checkpointing": False,
    "fp16": False,
    "bf16": False,
    "optim": "adamw_torch",
    "num_epochs": 1,
    "logging_steps": 5,
    "eval_steps": 50,
    "save_steps": 100,
}


# ════════════════════════════════════════════════════════════
#  Training Config Dataclass
# ════════════════════════════════════════════════════════════


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""

    base_model_id: str
    method: str  # "lora" or "qlora"
    output_dir: str
    train_file: str
    eval_file: Optional[str] = None

    # Training hyperparams
    num_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    early_stopping_patience: int = 3

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Redis pub/sub for streaming
    redis_url: Optional[str] = None
    pubsub_channel: Optional[str] = None

    # HF token for gated models
    hf_token: Optional[str] = None

    @classmethod
    def from_hyperparams(
        cls,
        base_model_id: str,
        method: str,
        output_dir: str,
        train_file: str,
        eval_file: Optional[str] = None,
        hyperparams: Optional[dict] = None,
        redis_url: Optional[str] = None,
        pubsub_channel: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> TrainingConfig:
        """Create config from user-supplied hyperparams, falling back to defaults."""
        hp = {**TRAINING_DEFAULTS, **(hyperparams or {})}
        lora_hp = QLORA_DEFAULTS if method == "qlora" else LORA_DEFAULTS

        return cls(
            base_model_id=base_model_id,
            method=method,
            output_dir=output_dir,
            train_file=train_file,
            eval_file=eval_file,
            num_epochs=hp.get("num_epochs", TRAINING_DEFAULTS["num_epochs"]),
            learning_rate=hp.get("learning_rate", TRAINING_DEFAULTS["learning_rate"]),
            per_device_train_batch_size=hp.get("per_device_train_batch_size", hp.get("per_device_batch_size", TRAINING_DEFAULTS["per_device_train_batch_size"])),
            per_device_eval_batch_size=hp.get("per_device_eval_batch_size", TRAINING_DEFAULTS["per_device_eval_batch_size"]),
            gradient_accumulation_steps=hp.get("gradient_accumulation_steps", TRAINING_DEFAULTS["gradient_accumulation_steps"]),
            max_seq_length=hp.get("max_seq_length", TRAINING_DEFAULTS["max_seq_length"]),
            warmup_ratio=hp.get("warmup_ratio", TRAINING_DEFAULTS["warmup_ratio"]),
            weight_decay=hp.get("weight_decay", TRAINING_DEFAULTS["weight_decay"]),
            lr_scheduler_type=hp.get("lr_scheduler_type", TRAINING_DEFAULTS["lr_scheduler_type"]),
            fp16=hp.get("fp16", TRAINING_DEFAULTS["fp16"]),
            bf16=hp.get("bf16", TRAINING_DEFAULTS["bf16"]),
            gradient_checkpointing=hp.get("gradient_checkpointing", TRAINING_DEFAULTS["gradient_checkpointing"]),
            optim=hp.get("optim", TRAINING_DEFAULTS["optim"]),
            logging_steps=hp.get("logging_steps", TRAINING_DEFAULTS["logging_steps"]),
            eval_steps=hp.get("eval_steps", TRAINING_DEFAULTS["eval_steps"]),
            save_steps=hp.get("save_steps", TRAINING_DEFAULTS["save_steps"]),
            save_total_limit=hp.get("save_total_limit", TRAINING_DEFAULTS["save_total_limit"]),
            load_best_model_at_end=hp.get("load_best_model_at_end", TRAINING_DEFAULTS["load_best_model_at_end"]),
            early_stopping_patience=hp.get("early_stopping_patience", TRAINING_DEFAULTS["early_stopping_patience"]),
            lora_r=hp.get("lora_r", lora_hp["r"]),
            lora_alpha=hp.get("lora_alpha", lora_hp["lora_alpha"]),
            lora_dropout=hp.get("lora_dropout", lora_hp["lora_dropout"]),
            redis_url=redis_url,
            pubsub_channel=pubsub_channel,
            hf_token=hf_token,
        )


# ════════════════════════════════════════════════════════════
#  Streaming Callback (Redis Pub/Sub)
# ════════════════════════════════════════════════════════════


class StreamingCallback(TrainerCallback):
    """Publishes training metrics to Redis Pub/Sub for real-time log streaming."""

    def __init__(self, redis_client: redis.Redis, channel: str):
        self.redis = redis_client
        self.channel = channel
        self._last_log_time = 0.0

    def _publish(self, payload: dict):
        """Safely publish a JSON message to the channel."""
        try:
            self.redis.publish(self.channel, json.dumps(payload, default=str))
        except Exception:
            logger.warning("streaming_callback.publish_failed", channel=self.channel)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            progress_pct = round(100 * state.global_step / state.max_steps) if state.max_steps else 0
            self._publish({
                "event": "log",
                "step": state.global_step,
                "total_steps": state.max_steps,
                "progress_pct": progress_pct,
                "epoch": round(state.epoch, 2) if state.epoch else None,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()},
            })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self._publish({
                "event": "eval",
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else None,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
            })

    def on_save(self, args, state, control, **kwargs):
        self._publish({
            "event": "checkpoint",
            "step": state.global_step,
            "epoch": round(state.epoch, 2) if state.epoch else None,
        })

    def on_train_end(self, args, state, control, **kwargs):
        self._publish({
            "event": "complete",
            "total_steps": state.global_step,
            "best_metric": state.best_metric,
        })


# ════════════════════════════════════════════════════════════
#  Cancel Callback
# ════════════════════════════════════════════════════════════


class CancelCheckCallback(TrainerCallback):
    """Checks a cancel flag in Redis and stops training when set."""

    def __init__(self, redis_client: redis.Redis, cancel_key: str):
        self.redis = redis_client
        self.cancel_key = cancel_key

    def on_step_end(self, args, state, control, **kwargs):
        try:
            if self.redis.get(self.cancel_key):
                logger.info("training.cancel_detected", step=state.global_step)
                control.should_training_stop = True
        except Exception:
            pass  # don't break training if Redis is down


# ════════════════════════════════════════════════════════════
#  Formatting function
# ════════════════════════════════════════════════════════════


def format_instruction(example: dict) -> str:
    """Convert an instruction/input/output example to a prompt string for SFTTrainer."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")

    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output_text}"
    )


# ════════════════════════════════════════════════════════════
#  SFT Engine
# ════════════════════════════════════════════════════════════


class SFTEngine:
    """
    Self-contained SFT fine-tuning engine.

    Usage:
        config = TrainingConfig.from_hyperparams(...)
        engine = SFTEngine(config)
        result = engine.run()
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_config = None
        self.trainer = None
        self._redis: Optional[redis.Redis] = None

    # ── Setup ────────────────────────────────────────────

    def _get_redis(self) -> Optional[redis.Redis]:
        if self._redis is None and self.config.redis_url:
            client = None
            try:
                client = redis.from_url(self.config.redis_url)
                client.ping()
                self._redis = client
            except Exception:
                logger.warning("sft_engine.redis_connect_failed")
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        pass
                self._redis = None
        return self._redis

    @property
    def is_cpu(self) -> bool:
        """Detect if we're running on CPU (no CUDA available)."""
        return not torch.cuda.is_available()

    def setup_model(self) -> None:
        """Load the base model with appropriate quantization config."""
        cpu_mode = self.is_cpu
        logger.info(
            "sft_engine.loading_model",
            model=self.config.base_model_id,
            method=self.config.method,
            device="cpu" if cpu_mode else "cuda",
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
        }
        if self.config.hf_token:
            model_kwargs["token"] = self.config.hf_token

        if cpu_mode:
            # CPU mode: fp32, no quantization, explicit CPU placement
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["device_map"] = {"":  "cpu"}
            # Force method to lora (qlora requires CUDA)
            if self.config.method == "qlora":
                logger.info("sft_engine.cpu_mode_forcing_lora")
                self.config.method = "lora"
        elif self.config.method == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if self.config.bf16 else torch.float16
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_id,
            **model_kwargs,
        )

        # NOTE: We do NOT call prepare_model_for_kbit_training() here because
        # SFTTrainer (trl >= 0.5) handles it internally when peft_config is provided.

        # Determine target modules: "auto" doesn't work for all architectures
        # (e.g. GPT-2 uses Conv1D), so we auto-detect linear layers.
        target_modules = self._detect_target_modules()

        # PEFT config
        self.peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        logger.info(
            "sft_engine.model_loaded",
            method=self.config.method,
            device="cpu" if cpu_mode else "cuda",
            target_modules=target_modules,
        )

    def _detect_target_modules(self) -> list[str]:
        """Auto-detect LoRA target modules from the model architecture.

        Falls back to common module names if automatic detection fails.
        """
        try:
            # Try "auto" via peft's internal detection first
            from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
            model_type = getattr(self.model.config, "model_type", "").lower()
            if model_type in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_type]
                logger.info("sft_engine.target_modules_from_mapping", model_type=model_type, modules=modules)
                return modules
        except (ImportError, AttributeError):
            pass

        # Scan model for Linear and Conv1D layers
        target_names = set()
        for name, module in self.model.named_modules():
            cls_name = type(module).__name__
            if cls_name in ("Linear", "Conv1D"):
                # Use the short name (e.g., "c_attn", "c_proj", "q_proj")
                short_name = name.split(".")[-1]
                target_names.add(short_name)

        # Filter out output heads and embeddings
        exclude = {"lm_head", "embed_tokens", "wte", "wpe", "embed_out"}
        target_names -= exclude

        if target_names:
            modules = sorted(target_names)
            logger.info("sft_engine.target_modules_detected", modules=modules)
            return modules

        # Ultimate fallback for common architectures
        logger.warning("sft_engine.target_modules_fallback")
        return ["q_proj", "v_proj"]

    def setup_tokenizer(self) -> None:
        """Load and configure the tokenizer."""
        tok_kwargs = {}
        if self.config.hf_token:
            tok_kwargs["token"] = self.config.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_id,
            trust_remote_code=False,
            **tok_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        logger.info("sft_engine.tokenizer_loaded")

    def load_datasets(self) -> tuple[Dataset, Optional[Dataset]]:
        """Load train and eval JSONL files."""
        train_ds = load_dataset("json", data_files=self.config.train_file, split="train")
        eval_ds = None
        if self.config.eval_file and Path(self.config.eval_file).exists():
            eval_ds = load_dataset("json", data_files=self.config.eval_file, split="train")
        logger.info(
            "sft_engine.datasets_loaded",
            train_size=len(train_ds),
            eval_size=len(eval_ds) if eval_ds else 0,
        )
        return train_ds, eval_ds

    # ── Training ─────────────────────────────────────────

    def build_training_args(self) -> SFTConfig:
        """Build SFTConfig (replaces TrainingArguments for trl >= 0.12)."""
        cfg = self.config
        do_eval = cfg.eval_file is not None and Path(cfg.eval_file).exists()
        cpu_mode = self.is_cpu

        # On CPU: force safe settings
        fp16 = False if cpu_mode else cfg.fp16
        bf16 = False if cpu_mode else cfg.bf16
        optim = "adamw_torch" if cpu_mode else cfg.optim
        gradient_checkpointing = False if cpu_mode else cfg.gradient_checkpointing
        use_cpu = cpu_mode

        if cpu_mode:
            logger.info("sft_engine.cpu_training_args")

        return SFTConfig(
            output_dir=cfg.output_dir,
            max_seq_length=cfg.max_seq_length,
            num_train_epochs=cfg.num_epochs,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler_type,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            optim=optim,
            logging_steps=cfg.logging_steps,
            eval_strategy="steps" if do_eval else "no",
            eval_steps=cfg.eval_steps if do_eval else None,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=cfg.save_total_limit,
            load_best_model_at_end=cfg.load_best_model_at_end if do_eval else False,
            metric_for_best_model="eval_loss" if do_eval else None,
            greater_is_better=False if do_eval else None,
            report_to="none",
            remove_unused_columns=True,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            use_cpu=use_cpu,
        )

    def build_callbacks(self) -> list:
        """Build trainer callbacks (streaming + cancel + early stopping)."""
        callbacks = []
        r = self._get_redis()

        if r and self.config.pubsub_channel:
            callbacks.append(StreamingCallback(r, self.config.pubsub_channel))
            # Cancel check key: train_cancel:{project_id} — extracted from channel name
            parts = self.config.pubsub_channel.split(":")
            if len(parts) == 2:
                cancel_key = f"train_cancel:{parts[1]}"
                callbacks.append(CancelCheckCallback(r, cancel_key))

        # Early stopping (only if eval is configured)
        if self.config.eval_file and Path(self.config.eval_file).exists():
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)
            )

        return callbacks

    def run(self) -> dict[str, Any]:
        """
        Execute the full training pipeline:
        1) Setup model + tokenizer
        2) Load datasets
        3) Build trainer
        4) Train
        5) Save adapter + tokenizer
        6) Return metrics

        Returns:
            dict with keys: metrics, artifacts_dir, train_size, eval_size, duration_seconds
        """
        start_time = time.time()

        # 1. Setup
        self.setup_model()
        self.setup_tokenizer()

        # 2. Datasets
        train_ds, eval_ds = self.load_datasets()

        # 3. Training args + callbacks
        training_args = self.build_training_args()
        callbacks = self.build_callbacks()

        # 4. SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=self.peft_config,
            formatting_func=format_instruction,
            args=training_args,
            callbacks=callbacks,
        )

        # 5. Train
        logger.info("sft_engine.training_start")
        train_result = self.trainer.train()
        logger.info("sft_engine.training_end")

        # 6. Save adapter + tokenizer
        adapter_dir = os.path.join(self.config.output_dir, "adapter")
        self.trainer.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)
        logger.info("sft_engine.adapter_saved", path=adapter_dir)

        # 7. Collect metrics
        metrics = train_result.metrics
        if eval_ds is not None:
            eval_metrics = self.trainer.evaluate()
            metrics.update(eval_metrics)

        duration = time.time() - start_time

        # Compute perplexity
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            metrics["perplexity"] = round(math.exp(eval_loss), 4)

        # Build train loss curve from log history
        train_loss_curve = []
        eval_loss_curve = []
        for entry in self.trainer.state.log_history:
            if "loss" in entry and "step" in entry:
                train_loss_curve.append({"step": entry["step"], "loss": round(entry["loss"], 6)})
            if "eval_loss" in entry and "step" in entry:
                eval_loss_curve.append({"step": entry["step"], "eval_loss": round(entry["eval_loss"], 6)})

        result = {
            "metrics": {
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()},
                "train_loss_curve": train_loss_curve,
                "eval_loss_curve": eval_loss_curve,
            },
            "artifacts_dir": adapter_dir,
            "train_size": len(train_ds),
            "eval_size": len(eval_ds) if eval_ds else 0,
            "duration_seconds": round(duration, 1),
        }

        # Cleanup GPU memory
        self._cleanup()

        return result

    def _cleanup(self):
        """Free GPU memory."""
        try:
            del self.trainer
            del self.model
            del self.tokenizer
            self.trainer = None
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass


# ════════════════════════════════════════════════════════════
#  Utility: Hardware recommendation
# ════════════════════════════════════════════════════════════


def recommend_method(
    vram_gb: float,
    model_params_b: float,
    cuda_available: bool,
    bnb_available: bool,
) -> dict:
    """Return the recommended fine-tuning method based on hardware constraints."""
    if not cuda_available:
        return {
            "method": "lora",
            "reason": "No CUDA GPU detected. Training will run on CPU (slow but functional).",
            "can_train": True,
            "device": "cpu",
            "warnings": [
                "CPU training is very slow — use a small model (< 3B params)",
                "QLoRA requires CUDA; LoRA will be used instead",
                "Consider a cloud GPU for production fine-tuning",
            ],
            "suggestions": ["Use TinyLlama or a small model", "Reduce num_epochs and max_seq_length"],
        }

    vram_fp16 = model_params_b * 2 * 1.2
    vram_4bit = model_params_b * 0.5 * 1.3
    vram_lora = vram_fp16 * 1.15
    vram_qlora = vram_4bit * 1.3

    if vram_gb < 6:
        return {
            "method": None,
            "reason": f"Only {vram_gb}GB VRAM. Minimum 6GB required for QLoRA with small models.",
            "can_train": False,
            "suggestions": ["Use a GPU with more VRAM", "Try a smaller model (< 3B params)"],
        }

    if vram_gb < 12:
        if vram_qlora <= vram_gb * 0.85:
            return {
                "method": "qlora",
                "reason": f"{vram_gb}GB VRAM. QLoRA 4-bit recommended to fit in memory.",
                "can_train": True,
                "estimated_vram_usage_gb": round(vram_qlora, 1),
                "suggestions": ["Reduce batch_size if OOM", "Reduce max_seq_length"],
            }
        return {
            "method": "qlora",
            "reason": f"{vram_gb}GB VRAM might be tight for {model_params_b}B model.",
            "can_train": True,
            "warnings": ["High OOM risk. Reduce batch_size to 1 and max_seq_length to 1024."],
            "estimated_vram_usage_gb": round(vram_qlora, 1),
        }

    if vram_gb < 24:
        if bnb_available and vram_lora > vram_gb * 0.85:
            return {
                "method": "qlora",
                "reason": f"{vram_gb}GB VRAM. Model too large for LoRA FP16, using QLoRA.",
                "can_train": True,
                "estimated_vram_usage_gb": round(vram_qlora, 1),
            }
        return {
            "method": "lora",
            "reason": f"{vram_gb}GB VRAM. LoRA FP16 fits comfortably.",
            "can_train": True,
            "estimated_vram_usage_gb": round(vram_lora, 1),
            "alternative": "qlora (saves ~50% VRAM)",
        }

    return {
        "method": "lora",
        "reason": f"{vram_gb}GB VRAM. LoRA FP16 recommended for best quality.",
        "can_train": True,
        "estimated_vram_usage_gb": round(vram_lora, 1),
        "alternative": "qlora (if you want to save VRAM for larger batch sizes)",
    }

"""
FineTuneFlow — Evaluation Engine.

Runs evaluation on an eval split after training:
  - Computes perplexity from eval_loss (if available from SFTTrainer)
  - Runs sample inference on N examples from eval set
  - Returns structured metrics
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app.core.logging import get_logger

logger = get_logger(__name__)

# Maximum number of eval samples for inference
DEFAULT_INFERENCE_SAMPLES = 5
MAX_NEW_TOKENS = 512


class EvalEngine:
    """
    Evaluation engine for fine-tuned PEFT adapters.

    Usage:
        engine = EvalEngine(
            base_model_id="meta-llama/Llama-3.1-8B",
            adapter_dir="/path/to/adapter",
            eval_file="/path/to/eval.jsonl",
            method="qlora",
        )
        result = engine.run()
    """

    def __init__(
        self,
        base_model_id: str,
        adapter_dir: str,
        eval_file: str,
        method: str = "lora",
        max_seq_length: int = 2048,
        num_inference_samples: int = DEFAULT_INFERENCE_SAMPLES,
        hf_token: Optional[str] = None,
        # Pre-computed metrics from training (if available)
        train_metrics: Optional[dict] = None,
    ):
        self.base_model_id = base_model_id
        self.adapter_dir = adapter_dir
        self.eval_file = eval_file
        self.method = method
        self.max_seq_length = max_seq_length
        self.num_inference_samples = num_inference_samples
        self.hf_token = hf_token
        self.train_metrics = train_metrics or {}
        self.model = None
        self.tokenizer = None

    def _load_model_and_tokenizer(self) -> None:
        """Load base model + PEFT adapter (or full FT model) + tokenizer."""
        logger.info("eval_engine.loading_model", model=self.base_model_id, adapter=self.adapter_dir)

        is_cpu = not torch.cuda.is_available()
        model_dtype = torch.float32 if is_cpu else torch.bfloat16
        device_map = {"":  "cpu"} if is_cpu else "auto"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": False,
            "device_map": device_map,
        }
        if self.hf_token:
            model_kwargs["token"] = self.hf_token

        # QLoRA needs bitsandbytes quantization (CUDA only)
        if self.method == "qlora" and not is_cpu:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["dtype"] = model_dtype

        # Detect if this is a PEFT adapter or a full fine-tuned model
        adapter_config_path = Path(self.adapter_dir) / "adapter_config.json"
        if adapter_config_path.exists():
            # PEFT adapter (LoRA, QLoRA, DoRA, IA³, Prefix, etc.)
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, **model_kwargs
            )
            self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        else:
            # Full fine-tuned model — load directly from adapter_dir
            self.model = AutoModelForCausalLM.from_pretrained(
                self.adapter_dir, **model_kwargs
            )
        self.model.eval()

        tok_kwargs = {}
        if self.hf_token:
            tok_kwargs["token"] = self.hf_token

        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_dir,
                trust_remote_code=False,
                **tok_kwargs,
            )
        except Exception:
            logger.warning("eval_engine.tokenizer_fallback_to_base_model")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=False,
                **tok_kwargs,
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("eval_engine.model_loaded")

    def _compute_perplexity(self, eval_ds) -> Optional[float]:
        """Compute perplexity on the eval dataset."""
        # If we already have eval_loss from training, use it
        eval_loss = self.train_metrics.get("eval_loss")
        if eval_loss is not None:
            try:
                return round(math.exp(min(eval_loss, 700)), 4)  # guard against overflow
            except (OverflowError, ValueError):
                return float("inf")

        # Otherwise, compute manually
        try:
            from app.ml.sft_engine import format_instruction

            total_loss = 0.0
            count = 0

            for example in eval_ds:
                text = format_instruction(example)
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    if loss is not None and not torch.isnan(loss):
                        total_loss += loss.item()
                        count += 1

            if count > 0:
                avg_loss = total_loss / count
                return round(math.exp(avg_loss), 4)
        except Exception:
            logger.exception("eval_engine.perplexity_computation_failed")

        return None

    def _run_inference_samples(self, eval_ds) -> list[dict]:
        """Run inference on a subset of eval examples."""
        samples = []
        indices = list(range(min(self.num_inference_samples, len(eval_ds))))

        for idx in indices:
            example = eval_ds[idx]
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            expected = example.get("output", "")

            # Build prompt (without the expected output) — mirrors format_instruction structure
            if input_text:
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{input_text}\n\n"
                    f"### Response:\n"
                )
            else:
                prompt = (
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n"
                )

            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                ).to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode only the new tokens
                generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                model_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

                samples.append({
                    "instruction": instruction[:500],
                    "input": input_text[:500],
                    "expected_output": expected[:500],
                    "model_output": model_output[:1000],
                    "output_length": len(model_output),
                })
            except Exception as e:
                logger.warning("eval_engine.inference_sample_failed", idx=idx, error=str(e))
                samples.append({
                    "instruction": instruction[:500],
                    "input": input_text[:500],
                    "expected_output": expected[:500],
                    "model_output": f"[Error: {str(e)[:200]}]",
                    "output_length": 0,
                })

        return samples

    def run(self) -> dict[str, Any]:
        """
        Execute the full evaluation pipeline.

        Returns:
            dict with keys: perplexity, inference_samples, eval_loss, duration_seconds
        """
        start = time.time()

        # Load eval dataset
        eval_ds = load_dataset("json", data_files=self.eval_file, split="train")
        logger.info("eval_engine.dataset_loaded", size=len(eval_ds))

        if len(eval_ds) == 0:
            return {
                "perplexity": None,
                "inference_samples": [],
                "eval_loss": self.train_metrics.get("eval_loss"),
                "duration_seconds": round(time.time() - start, 1),
            }

        # Load model
        self._load_model_and_tokenizer()

        # Perplexity
        perplexity = self._compute_perplexity(eval_ds)

        # Sample inference
        inference_samples = self._run_inference_samples(eval_ds)

        duration = round(time.time() - start, 1)

        # Cleanup
        self._cleanup()

        return {
            "perplexity": perplexity,
            "inference_samples": inference_samples,
            "eval_loss": self.train_metrics.get("eval_loss"),
            "train_loss": self.train_metrics.get("train_loss"),
            "duration_seconds": duration,
        }

    def _cleanup(self):
        """Free GPU memory."""
        try:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

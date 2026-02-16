"""
FineTuneFlow — Inference Engine.

Loads a base model + LoRA adapter and generates text.
Caches the model in-process so repeated requests are fast.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── In-process model cache (one model at a time) ─────────
_cache_lock = threading.Lock()
_cached_model: Optional[PeftModel] = None
_cached_tokenizer: Optional[AutoTokenizer] = None
_cached_key: Optional[str] = None  # "base_model::adapter_dir"


def _cache_key(base_model_id: str, adapter_dir: str) -> str:
    return f"{base_model_id}::{adapter_dir}"


def _load_model(base_model_id: str, adapter_dir: str):
    """Load base model + LoRA adapter. Called under lock."""
    global _cached_model, _cached_tokenizer, _cached_key

    key = _cache_key(base_model_id, adapter_dir)
    if _cached_key == key and _cached_model is not None:
        logger.info("inference.cache_hit", key=key)
        return

    # Unload previous
    _unload()

    logger.info("inference.loading_model", base_model=base_model_id, adapter=adapter_dir)

    is_cpu = not torch.cuda.is_available()
    dtype = torch.float32 if is_cpu else torch.float16
    device_map = {"": "cpu"} if is_cpu else "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_key = key

    logger.info("inference.model_loaded", key=key)


def _unload():
    """Free model from memory."""
    global _cached_model, _cached_tokenizer, _cached_key
    if _cached_model is not None:
        del _cached_model
        _cached_model = None
    if _cached_tokenizer is not None:
        del _cached_tokenizer
        _cached_tokenizer = None
    _cached_key = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    base_model_id: str,
    adapter_dir: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> dict[str, Any]:
    """
    Generate text from a fine-tuned model.

    Returns:
        dict with keys: generated_text, prompt, num_tokens_generated
    """
    with _cache_lock:
        _load_model(base_model_id, adapter_dir)

        assert _cached_model is not None
        assert _cached_tokenizer is not None

        # Tokenize
        inputs = _cached_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        input_ids = inputs["input_ids"].to(_cached_model.device)
        attention_mask = inputs["attention_mask"].to(_cached_model.device)

        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=_cached_tokenizer.pad_token_id,
            eos_token_id=_cached_tokenizer.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            output_ids = _cached_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )

        # Decode only the generated part (skip prompt tokens)
        new_tokens = output_ids[0][input_ids.shape[1]:]
        generated_text = _cached_tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "generated_text": generated_text.strip(),
            "prompt": prompt,
            "num_tokens_prompt": input_ids.shape[1],
            "num_tokens_generated": len(new_tokens),
        }

"""
FineTuneFlow — Celery Tasks: Inference.

Tasks:
  - inference_generate: Generate text with a fine-tuned model.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(
    name="app.workers.tasks_inference.inference_generate",
    bind=True,
    soft_time_limit=120,
    time_limit=180,
    acks_late=True,
)
def inference_generate(
    self,
    base_model_id: str,
    adapter_dir: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> dict:
    """Generate text using a fine-tuned model (base + LoRA adapter)."""
    logger.info(
        "task.inference.start",
        base_model=base_model_id,
        adapter=adapter_dir,
        prompt_len=len(prompt),
    )

    try:
        from app.ml.inference_engine import generate

        result = generate(
            base_model_id=base_model_id,
            adapter_dir=adapter_dir,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        logger.info(
            "task.inference.success",
            tokens_generated=result["num_tokens_generated"],
        )
        return {"status": "success", **result}

    except Exception as exc:
        logger.error("task.inference.error", error=str(exc))
        return {"status": "error", "error": str(exc)}

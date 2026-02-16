"""
FineTuneFlow — Playground Routes.

Endpoints:
  GET  /projects/{project_id}/playground/status   — Check if a trained model is available
  POST /projects/{project_id}/playground/generate  — Generate text with the fine-tuned model
"""

from __future__ import annotations

import uuid
from pathlib import Path

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    PlaygroundGenerateRequest,
    PlaygroundGenerateResponse,
    PlaygroundStatusResponse,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import JobStatus, Project, Run
from app.db.session import get_db

logger = get_logger(__name__)

playground_router = APIRouter(
    prefix="/projects/{project_id}/playground",
    tags=["playground"],
)


def _get_successful_run(db: Session, project_id: uuid.UUID) -> Run | None:
    """Get the latest successful training run with an adapter."""
    return (
        db.query(Run)
        .filter(
            Run.project_id == project_id,
            Run.status == JobStatus.success,
            Run.artifacts_dir.isnot(None),
        )
        .order_by(Run.created_at.desc())
        .first()
    )


@playground_router.get("/status")
@limiter.limit("30/minute")
def playground_status(
    request: Request,
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Check if a trained model is available for playground testing."""
    if not project.base_model_id:
        return PlaygroundStatusResponse(
            available=False,
            message="No base model configured for this project.",
        )

    run = _get_successful_run(db, project_id)
    if not run:
        return PlaygroundStatusResponse(
            available=False,
            base_model_id=project.base_model_id,
            message="No successful training run found. Train a model first.",
        )

    adapter_path = Path(run.artifacts_dir) if run.artifacts_dir else None
    if not adapter_path or not adapter_path.exists():
        return PlaygroundStatusResponse(
            available=False,
            base_model_id=project.base_model_id,
            run_id=str(run.id),
            message="Adapter files not found on disk. Re-train or check storage.",
        )

    return PlaygroundStatusResponse(
        available=True,
        base_model_id=project.base_model_id,
        adapter_dir=str(adapter_path),
        run_id=str(run.id),
        method=run.method.value if run.method else None,
        metrics=run.metrics or None,
        message="Model ready for inference.",
    )


@playground_router.post("/generate")
@limiter.limit("10/minute")
def playground_generate(
    request: Request,
    project_id: uuid.UUID,
    body: PlaygroundGenerateRequest,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Generate text using the fine-tuned model."""
    if not project.base_model_id:
        raise HTTPException(status_code=400, detail="No base model configured.")

    run = _get_successful_run(db, project_id)
    if not run or not run.artifacts_dir:
        raise HTTPException(
            status_code=400,
            detail="No successful training run found. Train a model first.",
        )

    adapter_path = Path(run.artifacts_dir)
    if not adapter_path.exists():
        raise HTTPException(
            status_code=400,
            detail="Adapter files not found on disk.",
        )

    # Enqueue inference task on gpu_worker and wait for result
    from app.workers.celery_app import celery_app

    task = celery_app.send_task(
        "app.workers.tasks_inference.inference_generate",
        kwargs={
            "base_model_id": project.base_model_id,
            "adapter_dir": str(adapter_path),
            "prompt": body.prompt,
            "max_new_tokens": body.max_new_tokens,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "top_k": body.top_k,
            "repetition_penalty": body.repetition_penalty,
            "do_sample": body.do_sample,
        },
        queue="gpu_queue",
    )

    # Wait for result synchronously (timeout 120s)
    try:
        result = AsyncResult(task.id, app=celery_app).get(timeout=120)
    except Exception as exc:
        logger.error("playground.generate.timeout", error=str(exc))
        raise HTTPException(
            status_code=504,
            detail=f"Inference timed out or failed: {exc}",
        )

    if result.get("status") == "error":
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {result.get('error', 'Unknown')}",
        )

    return PlaygroundGenerateResponse(
        generated_text=result["generated_text"],
        prompt=result["prompt"],
        num_tokens_prompt=result["num_tokens_prompt"],
        num_tokens_generated=result["num_tokens_generated"],
    )

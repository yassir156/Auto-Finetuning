"""
FineTuneFlow — Training + Export Routes.

Endpoints:
  POST /projects/{project_id}/train/start
  GET  /projects/{project_id}/train/status
  GET  /projects/{project_id}/train/logs/stream
  POST /projects/{project_id}/train/cancel
  POST /projects/{project_id}/export
  GET  /projects/{project_id}/export/files
  GET  /projects/{project_id}/export/download/{filename}
  GET  /projects/{project_id}/export/download
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse as FastAPIFileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    ExportFileInfo,
    ExportFilesResponse,
    JobEnqueuedResponse,
    TrainCancelResponse,
    TrainEnqueuedResponse,
    TrainStartRequest,
    TrainStatusResponse,
)
from app.core.config import settings
from app.core.exceptions import (
    NotFoundError,
    TrainingAlreadyRunningError,
)
from app.core.logging import get_logger
from app.db.models import Job, JobStatus, JobType, Project, Run
from app.db.session import get_db

logger = get_logger(__name__)

train_router = APIRouter(prefix="/projects/{project_id}/train", tags=["training"])
export_router = APIRouter(prefix="/projects/{project_id}/export", tags=["export"])


# ══════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════

@train_router.post("/start", status_code=202)
@limiter.limit("3/minute")
def train_start(
    request: Request,
    project_id: uuid.UUID,
    body: TrainStartRequest,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Start a fine-tuning run."""
    # Check for existing running training
    active_job = (
        db.query(Job)
        .filter(
            Job.project_id == project_id,
            Job.type == JobType.train,
            Job.status.in_([JobStatus.queued, JobStatus.running]),
        )
        .first()
    )
    if active_job:
        raise TrainingAlreadyRunningError(project_id=str(project_id))

    # Create job
    job = Job(
        project_id=project_id,
        type=JobType.train,
        status=JobStatus.queued,
    )
    db.add(job)
    db.flush()

    # Create run
    hyperparams = body.hyperparams.model_dump() if body.hyperparams else {}
    run = Run(
        project_id=project_id,
        job_id=job.id,
        method=body.method,
        status=JobStatus.queued,
        hyperparams=hyperparams,
    )
    db.add(run)
    db.commit()
    db.refresh(job)
    db.refresh(run)

    # Enqueue Celery task (use send_task to avoid importing ML deps)
    from app.workers.celery_app import celery_app

    celery_app.send_task(
        "app.workers.tasks_train.train",
        args=[str(project_id), str(run.id)],
        task_id=str(job.id),
        queue="gpu_queue",
    )
    job.celery_task_id = str(job.id)
    db.commit()

    return TrainEnqueuedResponse(
        job_id=job.id,
        run_id=run.id,
        status="queued",
        message="Training job enqueued on gpu_queue",
    )


@train_router.get("/status")
@limiter.limit("60/minute")
def train_status(
    request: Request,
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Get training status (latest run)."""
    run = (
        db.query(Run)
        .filter(Run.project_id == project_id)
        .order_by(Run.created_at.desc())
        .first()
    )
    if not run:
        return TrainStatusResponse(status="no_runs")

    job = db.query(Job).filter(Job.id == run.job_id).first()

    elapsed = None
    if run.started_at:
        end = run.finished_at or datetime.now(timezone.utc)
        elapsed = (end - run.started_at).total_seconds()

    return TrainStatusResponse(
        run_id=run.id,
        job_id=run.job_id,
        status=run.status.value,
        method=run.method.value,
        progress_pct=job.progress_pct if job else 0,
        current_metrics=run.metrics or None,
        started_at=run.started_at,
        elapsed_seconds=elapsed,
    )


@train_router.get("/logs/stream")
async def train_logs_stream(
    request: Request,
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Stream training logs via SSE (Server-Sent Events)."""
    import asyncio

    import redis.asyncio as aioredis

    redis_client = aioredis.from_url(settings.REDIS_URL)
    channel_name = f"train_logs:{project_id}"

    async def event_generator():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel_name)
        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if msg and msg["type"] == "message":
                    data = msg["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    try:
                        parsed = json.loads(data)
                        event_type = parsed.pop("event", "log")
                        yield f"event: {event_type}\ndata: {json.dumps(parsed)}\n\n"
                    except json.JSONDecodeError:
                        yield f"event: log\ndata: {json.dumps({'raw': data})}\n\n"
                else:
                    # Send keepalive comment every second
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.1)
        finally:
            await pubsub.unsubscribe(channel_name)
            await redis_client.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@train_router.post("/cancel")
@limiter.limit("10/minute")
def train_cancel(
    request: Request,
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Cancel the active training job."""
    active_job = (
        db.query(Job)
        .filter(
            Job.project_id == project_id,
            Job.type == JobType.train,
            Job.status.in_([JobStatus.queued, JobStatus.running]),
        )
        .first()
    )
    if not active_job:
        return TrainCancelResponse(status="no_active_job", message="No active training job to cancel")

    # Revoke Celery task
    from app.workers.celery_app import celery_app

    if active_job.celery_task_id:
        celery_app.control.revoke(active_job.celery_task_id, terminate=True, signal="SIGTERM")

    active_job.status = JobStatus.cancelled
    active_job.finished_at = datetime.now(timezone.utc)

    # Also update the run
    run = db.query(Run).filter(Run.job_id == active_job.id).first()
    if run:
        run.status = JobStatus.cancelled
        run.finished_at = datetime.now(timezone.utc)

    db.commit()

    return TrainCancelResponse(status="cancelled", message="Training job cancelled")


# ══════════════════════════════════════════════
#  Export
# ══════════════════════════════════════════════

@export_router.post("", status_code=202)
@limiter.limit("5/minute")
def trigger_export(
    request: Request,
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    job = Job(
        project_id=project_id,
        type=JobType.export,
        status=JobStatus.queued,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Find latest successful run
    run = (
        db.query(Run)
        .filter(Run.project_id == project_id, Run.status == JobStatus.success)
        .order_by(Run.created_at.desc())
        .first()
    )
    if not run:
        raise NotFoundError(detail="No successful training run found. Train a model first.")

    from app.workers.celery_app import celery_app

    celery_app.send_task(
        "app.workers.tasks_export.export_artifacts",
        args=[str(project_id), str(run.id)],
        task_id=str(job.id),
    )
    job.celery_task_id = str(job.id)
    db.commit()

    return JobEnqueuedResponse(
        job_id=job.id,
        status="queued",
        message="Export job enqueued",
    )


@export_router.get("/files")
@limiter.limit("60/minute")
def list_export_files(
    request: Request,
    project_id: uuid.UUID,
    project: Project = Depends(get_project_or_404),
):
    export_dir = settings.storage_path / str(project_id) / "export"
    if not export_dir.exists():
        return ExportFilesResponse(files=[])

    files = []
    for f in sorted(export_dir.iterdir()):
        if f.is_file():
            files.append(ExportFileInfo(filename=f.name, size_bytes=f.stat().st_size))
    return ExportFilesResponse(files=files)


@export_router.get("/download/{filename}")
def download_export_file(
    project_id: uuid.UUID,
    filename: str,
    project: Project = Depends(get_project_or_404),
):
    file_path = settings.storage_path / str(project_id) / "export" / filename
    resolved = file_path.resolve()
    export_root = (settings.storage_path / str(project_id) / "export").resolve()

    # Sandboxing
    if not str(resolved).startswith(str(export_root) + "/"):
        raise NotFoundError(detail="File not found")
    if not resolved.is_file():
        raise NotFoundError(detail=f"Export file '{filename}' not found")

    return FastAPIFileResponse(
        path=str(resolved),
        filename=filename,
        media_type="application/octet-stream",
    )


@export_router.get("/download")
def download_export_zip(
    project_id: uuid.UUID,
    project: Project = Depends(get_project_or_404),
):
    zip_path = settings.storage_path / str(project_id) / "export" / "finetuneflow_export.zip"
    if not zip_path.exists():
        raise NotFoundError(detail="Export zip not found. Run export first.")
    return FastAPIFileResponse(
        path=str(zip_path),
        filename="finetuneflow_export.zip",
        media_type="application/zip",
    )

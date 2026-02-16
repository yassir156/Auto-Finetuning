"""
FineTuneFlow — Chunks Route.

Endpoints:
  POST /projects/{project_id}/chunks/generate
  GET  /projects/{project_id}/chunks
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    ChunkGenerateRequest,
    ChunkListResponse,
    ChunkResponse,
    JobEnqueuedResponse,
)
from app.db.models import Chunk, Job, JobStatus, JobType, Project
from app.db.session import get_db

router = APIRouter(prefix="/projects/{project_id}/chunks", tags=["chunks"])


@router.post("/generate", status_code=202)
@limiter.limit("5/minute")
def generate_chunks(
    request: Request,
    project_id: uuid.UUID,
    body: ChunkGenerateRequest = ChunkGenerateRequest(),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Enqueue a chunking job."""
    job = Job(
        project_id=project_id,
        type=JobType.chunking,
        status=JobStatus.queued,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Enqueue Celery task
    from app.workers.tasks_dataset import chunk_documents

    chunk_documents.apply_async(
        args=[str(project_id)],
        task_id=str(job.id),
        kwargs={
            "chunk_size_tokens": body.chunk_size_tokens,
            "chunk_overlap_tokens": body.chunk_overlap_tokens,
        },
    )

    # Update job with celery task id
    job.celery_task_id = str(job.id)
    db.commit()

    return JobEnqueuedResponse(
        job_id=job.id,
        status="queued",
        message="Chunking job enqueued",
    )


@router.get("")
@limiter.limit("60/minute")
def list_chunks(
    request: Request,
    project_id: uuid.UUID,
    source_file_id: Optional[uuid.UUID] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    q = db.query(Chunk).filter(Chunk.project_id == project_id)
    if source_file_id:
        q = q.filter(Chunk.source_file_id == source_file_id)

    total = q.count()
    chunks = q.order_by(Chunk.chunk_index).offset(offset).limit(limit).all()

    return ChunkListResponse(
        items=[ChunkResponse.model_validate(c) for c in chunks],
        total=total,
        limit=limit,
        offset=offset,
    )

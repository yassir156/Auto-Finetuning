"""
FineTuneFlow — Jobs Route.

Endpoints:
  GET  /projects/{project_id}/jobs
  GET  /jobs/{job_id}
  POST /jobs/{job_id}/cancel
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    JobCancelResponse,
    JobListResponse,
    JobResponse,
)
from app.core.exceptions import JobNotFoundError
from app.db.models import Job, JobStatus, Project
from app.db.session import get_db

router = APIRouter(tags=["jobs"])


@router.get("/projects/{project_id}/jobs")
@limiter.limit("60/minute")
def list_jobs(
    request: Request,
    project_id: uuid.UUID,
    type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    q = db.query(Job).filter(Job.project_id == project_id)
    if type:
        q = q.filter(Job.type == type)
    if status:
        q = q.filter(Job.status == status)
    jobs = q.order_by(Job.created_at.desc()).all()
    return JobListResponse(
        jobs=[JobResponse.model_validate(j) for j in jobs]
    )


@router.get("/jobs/{job_id}")
@limiter.limit("60/minute")
def get_job(
    request: Request,
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise JobNotFoundError(job_id=str(job_id))
    return JobResponse.model_validate(job)


@router.post("/jobs/{job_id}/cancel")
@limiter.limit("10/minute")
def cancel_job(
    request: Request,
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise JobNotFoundError(job_id=str(job_id))

    if job.status not in (JobStatus.queued, JobStatus.running):
        return JobCancelResponse(status=job.status.value)

    # Revoke Celery task
    from app.workers.celery_app import celery_app

    if job.celery_task_id:
        celery_app.control.revoke(job.celery_task_id, terminate=True, signal="SIGTERM")

    job.status = JobStatus.cancelled
    job.finished_at = datetime.now(timezone.utc)
    db.commit()

    return JobCancelResponse(status="cancelled")

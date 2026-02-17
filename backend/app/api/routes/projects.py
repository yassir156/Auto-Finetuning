"""
FineTuneFlow — Projects CRUD Route.

Endpoints:
  POST   /projects
  GET    /projects
  GET    /projects/{project_id}
  PATCH  /projects/{project_id}
  DELETE /projects/{project_id}
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    ProjectCreate,
    ProjectListItem,
    ProjectListResponse,
    ProjectResponse,
    ProjectStats,
    ProjectUpdate,
)
from app.core.exceptions import ProjectNotFoundError
from app.db.models import (
    DatasetExample,
    File,
    Chunk,
    Job,
    Project,
    ProjectStatus,
)
from app.db.session import get_db
from app.services import storage as storage_svc

router = APIRouter(prefix="/projects", tags=["projects"])


# ── Helpers ───────────────────────────────────

def _project_stats(db: Session, project_id: uuid.UUID) -> ProjectStats:
    """Compute aggregated stats for a project."""
    file_count = db.query(sa_func.count(File.id)).filter(File.project_id == project_id).scalar() or 0
    chunk_count = db.query(sa_func.count(Chunk.id)).filter(Chunk.project_id == project_id).scalar() or 0
    example_count = db.query(sa_func.count(DatasetExample.id)).filter(
        DatasetExample.project_id == project_id
    ).scalar() or 0
    last_job = (
        db.query(Job)
        .filter(Job.project_id == project_id)
        .order_by(Job.created_at.desc())
        .first()
    )
    return ProjectStats(
        file_count=file_count,
        chunk_count=chunk_count,
        example_count=example_count,
        last_job_status=last_job.status.value if last_job else None,
    )


def _batch_project_stats(db: Session, project_ids: list[uuid.UUID]) -> dict[uuid.UUID, ProjectStats]:
    """Compute stats for multiple projects in bulk (avoids N+1 queries)."""
    if not project_ids:
        return {}

    # File counts
    file_counts = dict(
        db.query(File.project_id, sa_func.count(File.id))
        .filter(File.project_id.in_(project_ids))
        .group_by(File.project_id)
        .all()
    )
    # Chunk counts
    chunk_counts = dict(
        db.query(Chunk.project_id, sa_func.count(Chunk.id))
        .filter(Chunk.project_id.in_(project_ids))
        .group_by(Chunk.project_id)
        .all()
    )
    # Example counts
    example_counts = dict(
        db.query(DatasetExample.project_id, sa_func.count(DatasetExample.id))
        .filter(DatasetExample.project_id.in_(project_ids))
        .group_by(DatasetExample.project_id)
        .all()
    )
    # Latest job per project (using a window function / subquery)
    from sqlalchemy import desc
    from sqlalchemy.orm import aliased

    latest_job_sq = (
        db.query(
            Job.project_id,
            Job.status,
            sa_func.row_number().over(
                partition_by=Job.project_id, order_by=desc(Job.created_at)
            ).label("rn"),
        )
        .filter(Job.project_id.in_(project_ids))
        .subquery()
    )
    latest_jobs = dict(
        db.query(latest_job_sq.c.project_id, latest_job_sq.c.status)
        .filter(latest_job_sq.c.rn == 1)
        .all()
    )

    result = {}
    for pid in project_ids:
        result[pid] = ProjectStats(
            file_count=file_counts.get(pid, 0),
            chunk_count=chunk_counts.get(pid, 0),
            example_count=example_counts.get(pid, 0),
            last_job_status=latest_jobs.get(pid),
        )
    return result


def _to_response(project: Project, stats: Optional[ProjectStats] = None) -> dict:
    """Convert ORM project to response dict."""
    data = ProjectResponse.model_validate(project).model_dump()
    if stats:
        data["stats"] = stats.model_dump()
    return data


# ── Routes ────────────────────────────────────

@router.post("", status_code=201)
@limiter.limit("30/minute")
def create_project(
    request: Request,
    body: ProjectCreate,
    db: Session = Depends(get_db),
):
    project = Project(
        name=body.name,
        description=body.description,
        task_type=body.task_type,
        config=body.config.model_dump() if body.config else {},
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return _to_response(project)


@router.get("")
@limiter.limit("60/minute")
def list_projects(
    request: Request,
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Project)
    if status:
        q = q.filter(Project.status == status)
    total = q.count()
    projects = q.order_by(Project.created_at.desc()).offset(offset).limit(limit).all()

    # Batch stats: 4 queries total instead of 4×N
    project_ids = [p.id for p in projects]
    stats_map = _batch_project_stats(db, project_ids)

    items = []
    for p in projects:
        item_data = ProjectListItem.model_validate(p).model_dump()
        item_data["stats"] = stats_map.get(p.id, ProjectStats()).model_dump()
        items.append(item_data)

    return ProjectListResponse(
        items=[ProjectListItem(**i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{project_id}")
@limiter.limit("60/minute")
def get_project(
    request: Request,
    project: Project = Depends(get_project_or_404),
    db: Session = Depends(get_db),
):
    stats = _project_stats(db, project.id)
    return _to_response(project, stats)


@router.patch("/{project_id}")
@limiter.limit("30/minute")
def update_project(
    request: Request,
    body: ProjectUpdate,
    project: Project = Depends(get_project_or_404),
    db: Session = Depends(get_db),
):
    update_data = body.model_dump(exclude_unset=True)
    if "config" in update_data and update_data["config"] is not None:
        # Merge config
        current_config = project.config or {}
        current_config.update(update_data["config"].model_dump() if hasattr(update_data["config"], "model_dump") else update_data["config"])
        update_data["config"] = current_config

    for field, value in update_data.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return _to_response(project)


@router.delete("/{project_id}", status_code=204)
@limiter.limit("10/minute")
def delete_project(
    request: Request,
    project: Project = Depends(get_project_or_404),
    db: Session = Depends(get_db),
):
    # Delete from DB first (cascade will remove all related records)
    db.delete(project)
    db.commit()
    # Then delete storage on disk (after DB commit succeeds)
    storage_svc.delete_project_storage(str(project.id))
    return None

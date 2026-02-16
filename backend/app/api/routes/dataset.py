"""
FineTuneFlow — Dataset Route.

Endpoints:
  POST   /projects/{project_id}/dataset/preview
  POST   /projects/{project_id}/dataset/generate
  POST   /projects/{project_id}/dataset/upload   (JSONL upload)
  GET    /projects/{project_id}/dataset/examples
  DELETE /projects/{project_id}/dataset/examples/{example_id}
  GET    /projects/{project_id}/dataset/stats
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File as FastAPIFile, Query, Request, UploadFile
from sqlalchemy import func as sa_func
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    DatasetExampleListResponse,
    DatasetExampleResponse,
    DatasetFullStats,
    DatasetGenerateRequest,
    DatasetPreviewRequest,
    DatasetSplitStats,
    DatasetStats,
    JobEnqueuedResponse,
)
from app.core.exceptions import NotFoundError, InputValidationError
from app.db.models import DatasetExample, DatasetSplit, Job, JobStatus, JobType, Project, ProjectStatus
from app.db.session import get_db
from app.services.task_registry import get_task_config, resolve_task_key

router = APIRouter(prefix="/projects/{project_id}/dataset", tags=["dataset"])


@router.post("/preview", status_code=202)
@limiter.limit("5/minute")
def dataset_preview(
    request: Request,
    project_id: uuid.UUID,
    body: DatasetPreviewRequest = DatasetPreviewRequest(),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    job = Job(
        project_id=project_id,
        type=JobType.dataset_preview,
        status=JobStatus.queued,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from app.workers.tasks_dataset import dataset_preview as preview_task

    preview_task.apply_async(
        args=[str(project_id)],
        kwargs={"num_examples": body.num_examples},
        task_id=str(job.id),
    )
    job.celery_task_id = str(job.id)
    db.commit()

    return JobEnqueuedResponse(
        job_id=job.id,
        status="queued",
        message="Dataset preview job enqueued",
    )


@router.post("/generate", status_code=202)
@limiter.limit("3/minute")
def dataset_generate(
    request: Request,
    project_id: uuid.UUID,
    body: DatasetGenerateRequest = DatasetGenerateRequest(),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    job = Job(
        project_id=project_id,
        type=JobType.dataset_generate,
        status=JobStatus.queued,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    from app.workers.tasks_dataset import dataset_generate as gen_task

    gen_task.apply_async(
        args=[str(project_id)],
        task_id=str(job.id),
        kwargs={
            "num_examples_target": body.num_examples_target,
            "examples_per_chunk": body.examples_per_chunk,
            "ollama_model": body.ollama_model,
        },
    )
    job.celery_task_id = str(job.id)
    db.commit()

    return JobEnqueuedResponse(
        job_id=job.id,
        status="queued",
        message="Dataset generation job enqueued",
    )


@router.get("/examples")
@limiter.limit("60/minute")
def list_examples(
    request: Request,
    project_id: uuid.UUID,
    split: Optional[str] = Query(default=None),
    is_valid: Optional[bool] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    q = db.query(DatasetExample).filter(DatasetExample.project_id == project_id)
    if split:
        q = q.filter(DatasetExample.split == split)
    if is_valid is not None:
        q = q.filter(DatasetExample.is_valid == is_valid)

    total = q.count()
    examples = q.order_by(DatasetExample.created_at.desc()).offset(offset).limit(limit).all()

    # Global stats (always unfiltered — over entire project)
    global_q = db.query(DatasetExample).filter(DatasetExample.project_id == project_id)
    global_total = global_q.count()
    valid_count = global_q.filter(DatasetExample.is_valid.is_(True)).count()
    invalid_count = global_total - valid_count
    train_count = global_q.filter(DatasetExample.split == "train").count()
    eval_count = global_q.filter(DatasetExample.split == "eval").count()
    avg_tokens = db.query(sa_func.avg(DatasetExample.token_count)).filter(
        DatasetExample.project_id == project_id
    ).scalar()

    stats = DatasetStats(
        total=global_total,
        valid=valid_count,
        invalid=invalid_count,
        train=train_count,
        eval=eval_count,
        avg_token_count=float(avg_tokens) if avg_tokens else None,
    )

    return DatasetExampleListResponse(
        items=[DatasetExampleResponse.model_validate(e) for e in examples],
        total=total,
        limit=limit,
        offset=offset,
        stats=stats,
    )


@router.delete("/examples/{example_id}", status_code=204)
@limiter.limit("30/minute")
def delete_example(
    request: Request,
    project_id: uuid.UUID,
    example_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    ex = (
        db.query(DatasetExample)
        .filter(DatasetExample.id == example_id, DatasetExample.project_id == project_id)
        .first()
    )
    if ex is None:
        raise NotFoundError(detail=f"Dataset example '{example_id}' not found")
    db.delete(ex)
    db.commit()
    return None


@router.get("/stats")
@limiter.limit("60/minute")
def dataset_stats(
    request: Request,
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    base_q = db.query(DatasetExample).filter(DatasetExample.project_id == project_id)

    total = base_q.count()
    valid = base_q.filter(DatasetExample.is_valid.is_(True)).count()
    invalid = total - valid

    by_split: dict[str, DatasetSplitStats] = {}
    for split_name in ("train", "eval", "preview"):
        sq = base_q.filter(DatasetExample.split == split_name)
        cnt = sq.count()
        if cnt > 0:
            avg_t = sq.with_entities(sa_func.avg(DatasetExample.token_count)).scalar()
            min_t = sq.with_entities(sa_func.min(DatasetExample.token_count)).scalar()
            max_t = sq.with_entities(sa_func.max(DatasetExample.token_count)).scalar()
            by_split[split_name] = DatasetSplitStats(
                count=cnt,
                avg_tokens=float(avg_t) if avg_t else None,
                min_tokens=min_t,
                max_tokens=max_t,
            )

    return DatasetFullStats(
        total=total,
        valid=valid,
        invalid=invalid,
        by_split=by_split,
    )


@router.post("/upload")
@limiter.limit("5/minute")
def upload_dataset_jsonl(
    request: Request,
    project_id: uuid.UUID,
    file: UploadFile = FastAPIFile(...),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """
    Upload a user-provided JSONL dataset file.

    Validates each line against the project's task type schema,
    stores valid examples, and returns stats.
    """
    if not file.filename or not file.filename.endswith((".jsonl", ".json")):
        raise InputValidationError(
            detail="File must be a .jsonl or .json file",
        )

    task_type = project.task_type.value if project.task_type else "instruction_tuning"
    config = get_task_config(task_type)
    required_fields = set(config.required_fields)

    content = file.file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise InputValidationError(
            detail="File too large (max 50 MB)",
        )

    text = content.decode("utf-8", errors="replace")
    lines = text.strip().split("\n")

    valid_count = 0
    invalid_count = 0
    errors: list[dict] = []

    # Delete old uploaded examples
    db.query(DatasetExample).filter(
        DatasetExample.project_id == project_id,
        DatasetExample.split.in_([DatasetSplit.train, DatasetSplit.eval]),
    ).delete(synchronize_session="fetch")
    db.commit()

    # Split: 90% train, 10% eval
    import random
    split_ratio = 0.9
    parsed_examples = []

    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            invalid_count += 1
            if len(errors) < 10:
                errors.append({"line": line_num, "error": f"Invalid JSON: {str(e)}"})
            continue

        if not isinstance(obj, dict):
            invalid_count += 1
            if len(errors) < 10:
                errors.append({"line": line_num, "error": "Not a JSON object"})
            continue

        # Validate required fields
        missing = required_fields - set(obj.keys())
        if missing:
            invalid_count += 1
            if len(errors) < 10:
                errors.append({"line": line_num, "error": f"Missing fields: {missing}"})
            continue

        parsed_examples.append(obj)

    # Shuffle and split
    random.shuffle(parsed_examples)
    split_idx = int(len(parsed_examples) * split_ratio)
    train_examples = parsed_examples[:split_idx]
    eval_examples = parsed_examples[split_idx:]

    for ex_data in train_examples:
        token_ct = len(json.dumps(ex_data, ensure_ascii=False).split())
        ch = hashlib.sha256(
            json.dumps(ex_data, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        db.add(DatasetExample(
            project_id=project_id,
            split=DatasetSplit.train,
            data=ex_data,
            is_valid=True,
            token_count=token_ct,
            content_hash=ch,
        ))
        valid_count += 1

    for ex_data in eval_examples:
        token_ct = len(json.dumps(ex_data, ensure_ascii=False).split())
        ch = hashlib.sha256(
            json.dumps(ex_data, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        db.add(DatasetExample(
            project_id=project_id,
            split=DatasetSplit.eval,
            data=ex_data,
            is_valid=True,
            token_count=token_ct,
            content_hash=ch,
        ))
        valid_count += 1

    db.commit()

    # Move project to ready_to_train if we have valid examples
    if valid_count > 0:
        project.status = ProjectStatus.ready_to_train
        project.error_message = None
        db.commit()

    return {
        "total_lines": len(lines),
        "valid": valid_count,
        "invalid": invalid_count,
        "train": len(train_examples),
        "eval": len(eval_examples),
        "errors": errors,
    }

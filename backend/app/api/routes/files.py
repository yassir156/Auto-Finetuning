"""
FineTuneFlow — Files Route.

Endpoints:
  POST   /projects/{project_id}/files/upload
  GET    /projects/{project_id}/files
  DELETE /projects/{project_id}/files/{file_id}
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import FileListResponse, FileResponse, FileUploadResponse
from app.core.config import settings
from app.core.exceptions import (
    FileRecordNotFoundError,
    InputValidationError,
)
from app.db.models import File, FileKind, FileStatus, Project
from app.db.session import get_db
from app.services import storage as storage_svc

router = APIRouter(prefix="/projects/{project_id}/files", tags=["files"])


@router.post("/upload", status_code=201)
@limiter.limit("10/minute")
async def upload_files(
    request: Request,
    project_id: uuid.UUID,
    files: list[UploadFile],
    kind: str = Query(default="raw_doc"),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Upload one or more files to a project."""
    # Validate kind
    try:
        file_kind = FileKind(kind)
    except ValueError:
        raise InputValidationError(
            detail=f"Invalid file kind '{kind}'. Must be one of: {[k.value for k in FileKind]}",
            error_code="VALIDATION_ERROR",
        )

    # Validate batch size
    if len(files) > settings.MAX_FILES_PER_UPLOAD:
        raise InputValidationError(
            detail=f"Too many files in a single upload ({len(files)}). Max: {settings.MAX_FILES_PER_UPLOAD}",
            error_code="VALIDATION_ERROR",
        )

    # Check project file count limit
    existing_count = db.query(File).filter(File.project_id == project_id).count()
    if existing_count + len(files) > settings.MAX_FILES_PER_PROJECT:
        raise InputValidationError(
            detail=f"Would exceed max files per project ({settings.MAX_FILES_PER_PROJECT})",
            error_code="VALIDATION_ERROR",
        )

    # Current total size on disk
    current_total = storage_svc.get_project_storage_size(str(project_id))

    saved_files: list[FileResponse] = []
    saved_paths: list[Path] = []  # track for rollback

    try:
        for upload_file in files:
            filename = upload_file.filename or "unnamed"

            # Validate extension
            storage_svc.validate_extension(filename, kind)

            # Stream to disk with size check
            subdir = "raw_docs" if file_kind == FileKind.raw_doc else "datasets"
            path, sanitized_name, size_bytes = await storage_svc.stream_upload(
                project_id=str(project_id),
                subdir=subdir,
                filename=filename,
                file=upload_file,
            )
            saved_paths.append(path)

            # Check project total
            try:
                storage_svc.validate_project_total_size(
                    str(project_id), current_total, size_bytes
                )
            except Exception:
                storage_svc.delete_file(path)
                raise
            current_total += size_bytes

            # Validate MIME type via libmagic
            try:
                mime_type = storage_svc.validate_mime_type(path, kind)
            except Exception:
                storage_svc.delete_file(path)
                raise

            # Persist to DB
            db_file = File(
                project_id=project_id,
                kind=file_kind,
                status=FileStatus.ready,
                filename=sanitized_name,
                mime_type=mime_type,
                storage_path=str(path),
                size_bytes=size_bytes,
            )
            db.add(db_file)
            db.flush()  # get the id

            saved_files.append(FileResponse.model_validate(db_file))

    except Exception:
        # Rollback: delete all files saved so far in this batch
        for p in saved_paths:
            storage_svc.delete_file(p)
        raise

    db.commit()
    return FileUploadResponse(files=saved_files)


@router.get("")
@limiter.limit("60/minute")
def list_files(
    request: Request,
    project_id: uuid.UUID,
    kind: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    q = db.query(File).filter(File.project_id == project_id)
    if kind:
        q = q.filter(File.kind == kind)
    files = q.order_by(File.created_at.desc()).all()
    return FileListResponse(
        files=[FileResponse.model_validate(f) for f in files]
    )


@router.delete("/{file_id}", status_code=204)
@limiter.limit("30/minute")
def delete_file(
    request: Request,
    project_id: uuid.UUID,
    file_id: uuid.UUID,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    db_file = (
        db.query(File)
        .filter(File.id == file_id, File.project_id == project_id)
        .first()
    )
    if db_file is None:
        raise FileRecordNotFoundError(file_id=str(file_id))

    # Delete from disk
    storage_svc.delete_file(Path(db_file.storage_path))

    # Delete from DB
    db.delete(db_file)
    db.commit()
    return None

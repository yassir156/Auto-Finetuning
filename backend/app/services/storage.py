"""
FineTuneFlow — Storage Service.

Handles all file I/O with sandboxing against path traversal.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Optional

import magic

from app.core.config import settings
from app.core.exceptions import (
    FileEmptyError,
    FileInvalidNameError,
    FileTooLargeError,
    FileUnsupportedTypeError,
    ProjectTotalSizeExceededError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Allowed MIME types per file kind ──────────
ALLOWED_MIME_TYPES: dict[str, set[str]] = {
    "raw_doc": {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/x-markdown",
    },
    "dataset_upload": {
        "application/json",
        "text/plain",
        "text/csv",
        "application/csv",
    },
}

ALLOWED_EXTENSIONS: dict[str, set[str]] = {
    "raw_doc": {".pdf", ".docx", ".txt", ".md"},
    "dataset_upload": {".jsonl", ".json", ".csv"},
}


# ══════════════════════════════════════════════
#  Filename Sanitisation
# ══════════════════════════════════════════════

def sanitize_filename(filename: str) -> str:
    """
    Sanitise an uploaded filename.
    Removes dangerous characters, prevents path traversal.
    """
    # 1. Extract only the base name (no path components)
    filename = os.path.basename(filename)

    # 2. Remove disallowed characters — keep: letters, digits, -, _, .
    filename = re.sub(r"[^\w\-.]", "_", filename)

    # 3. Remove .. (path traversal)
    filename = filename.replace("..", "_")

    # 4. Remove leading dots (hidden files)
    filename = filename.lstrip(".")

    # 5. If empty after sanitisation, generate a name
    if not filename:
        filename = f"file_{uuid.uuid4().hex[:8]}"

    # 6. Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 200:
        name = name[:200]
    filename = name + ext

    return filename


def safe_storage_path(
    project_id: str, subdir: str, filename: str
) -> Path:
    """
    Build a safe storage path and verify it stays within STORAGE_ROOT.
    Raises FileInvalidNameError on path traversal.
    """
    sanitized = sanitize_filename(filename)
    storage_root = settings.storage_path.resolve()
    target = storage_root / project_id / subdir / sanitized
    resolved = target.resolve()

    if not str(resolved).startswith(str(storage_root) + os.sep) and resolved != storage_root:
        raise FileInvalidNameError(filename=filename)

    return target


# ══════════════════════════════════════════════
#  Validation Helpers
# ══════════════════════════════════════════════

def validate_extension(filename: str, kind: str) -> str:
    """Return the lowered extension, or raise."""
    ext = os.path.splitext(filename)[1].lower()
    allowed = ALLOWED_EXTENSIONS.get(kind, set())
    if ext not in allowed:
        raise FileUnsupportedTypeError(filename=filename, mime_type=f"extension={ext}")
    return ext


def validate_mime_type(file_path: Path, kind: str) -> str:
    """Detect the actual MIME type via libmagic and validate."""
    mime = magic.from_file(str(file_path), mime=True)
    allowed = ALLOWED_MIME_TYPES.get(kind, set())
    if mime not in allowed:
        raise FileUnsupportedTypeError(filename=file_path.name, mime_type=mime)
    return mime


def validate_file_size(size_bytes: int, filename: str) -> None:
    """Raise if file exceeds the per-file size limit."""
    if size_bytes == 0:
        raise FileEmptyError(filename=filename)
    max_bytes = settings.max_file_size_bytes
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            filename=filename,
            size_mb=size_bytes / (1024 * 1024),
            max_mb=settings.MAX_FILE_SIZE_MB,
        )


def validate_project_total_size(
    project_id: str,
    current_total_bytes: int,
    new_file_bytes: int,
) -> None:
    """Raise if adding this file would exceed the project total limit."""
    max_total = settings.max_project_total_size_bytes
    if current_total_bytes + new_file_bytes > max_total:
        raise ProjectTotalSizeExceededError(
            project_id=project_id,
            max_mb=settings.MAX_PROJECT_TOTAL_SIZE_MB,
        )


# ══════════════════════════════════════════════
#  File I/O
# ══════════════════════════════════════════════

def save_upload(
    project_id: str,
    subdir: str,
    filename: str,
    content: bytes,
) -> tuple[Path, str, int]:
    """
    Save uploaded bytes to storage.

    Returns (path, sanitized_filename, size_bytes).
    """
    target = safe_storage_path(project_id, subdir, filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Avoid overwriting — append a short uuid if file exists
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        target = target.with_name(f"{stem}_{uuid.uuid4().hex[:6]}{suffix}")

    target.write_bytes(content)

    size = target.stat().st_size
    logger.info(
        "storage.file_saved",
        project_id=project_id,
        filename=target.name,
        size_bytes=size,
    )
    return target, target.name, size


async def stream_upload(
    project_id: str,
    subdir: str,
    filename: str,
    file,  # fastapi.UploadFile
    max_size_bytes: Optional[int] = None,
) -> tuple[Path, str, int]:
    """
    Stream an UploadFile to storage, checking size on the fly.

    Returns (path, sanitized_filename, size_bytes).
    """
    target = safe_storage_path(project_id, subdir, filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        stem = target.stem
        suffix = target.suffix
        target = target.with_name(f"{stem}_{uuid.uuid4().hex[:6]}{suffix}")

    max_bytes = max_size_bytes or settings.max_file_size_bytes
    total = 0
    chunk_size = 1024 * 1024  # 1 MB

    with open(target, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                # Clean up partial file
                target.unlink(missing_ok=True)
                raise FileTooLargeError(
                    filename=filename,
                    size_mb=total / (1024 * 1024),
                    max_mb=settings.MAX_FILE_SIZE_MB,
                )
            f.write(chunk)

    if total == 0:
        target.unlink(missing_ok=True)
        raise FileEmptyError(filename=filename)

    logger.info(
        "storage.file_streamed",
        project_id=project_id,
        filename=target.name,
        size_bytes=total,
    )
    return target, target.name, total


def delete_file(path: Path) -> None:
    """Delete a single file from storage."""
    if path.exists():
        path.unlink()
        logger.info("storage.file_deleted", path=str(path))


def delete_project_storage(project_id: str) -> None:
    """Delete the entire storage directory for a project."""
    project_dir = settings.storage_path / project_id
    if project_dir.exists():
        shutil.rmtree(project_dir)
        logger.info("storage.project_deleted", project_id=project_id)


def get_project_storage_size(project_id: str) -> int:
    """Return the total byte size of files stored for a project."""
    project_dir = settings.storage_path / project_id
    if not project_dir.exists():
        return 0
    total = 0
    for f in project_dir.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total

"""
FineTuneFlow — API Dependencies.

Reusable FastAPI dependencies:
  - get_db: SQLAlchemy session (from db.session)
  - get_project_or_404: fetch project by ID or raise 404
  - rate_limiter: slowapi limiter instance
  - get_validated_file_kind: validate and return file kind enum
"""

from __future__ import annotations

import uuid

from fastapi import Depends, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import ProjectNotFoundError
from app.db.models import Project
from app.db.session import get_db

# ── Rate Limiter ──────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri=settings.REDIS_URL,
)


# ── Project Loader ────────────────────────────
def get_project_or_404(
    project_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Project:
    """Load a project by UUID, or raise ProjectNotFoundError."""
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        raise ProjectNotFoundError(project_id=str(project_id))
    return project

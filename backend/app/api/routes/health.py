"""
FineTuneFlow — Health Route.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Check backend + DB + Redis connectivity."""
    # -- DB check --
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "disconnected"

    # -- Redis check --
    redis_status = "connected"
    try:
        import redis

        r = redis.Redis.from_url(settings.CELERY_BROKER_URL, socket_timeout=2)
        r.ping()
    except Exception:
        redis_status = "disconnected"

    status = "ok" if db_status == "connected" and redis_status == "connected" else "degraded"

    return {
        "status": status,
        "db": db_status,
        "redis": redis_status,
        "version": settings.APP_VERSION,
    }


@router.get("/task-types")
def list_task_types():
    """Return all available task type configurations for the frontend."""
    from app.services.task_registry import TASK_CONFIGS

    return {
        "task_types": [
            {
                "key": cfg.key,
                "label": cfg.label,
                "description": cfg.description,
                "required_fields": cfg.required_fields,
                "optional_fields": cfg.optional_fields,
                "sample_example": cfg.sample_example,
                "display_columns": cfg.display_columns,
            }
            for cfg in TASK_CONFIGS.values()
        ]
    }

"""
FineTuneFlow — Shared Celery Task Utilities.

Common helpers used across all Celery task modules.
"""

from __future__ import annotations

from contextlib import contextmanager

from app.core.logging import get_logger
from app.db.models import Job, Run
from app.db.session import SessionLocal

logger = get_logger(__name__)


@contextmanager
def get_task_db():
    """Create a DB session for Celery tasks (context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def safe_update_job(db, job_id: str, **kwargs):
    """Update job fields, swallowing errors to avoid masking the original exception."""
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            db.commit()
        return job
    except Exception:
        logger.exception("task.job_update_failed", job_id=job_id)
        try:
            db.rollback()
        except Exception:
            pass
        return None


def safe_update_run(db, run_id: str, **kwargs):
    """Update run fields, swallowing errors to avoid masking the original exception."""
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            for k, v in kwargs.items():
                setattr(run, k, v)
            db.commit()
        return run
    except Exception:
        logger.exception("task.run_update_failed", run_id=run_id)
        try:
            db.rollback()
        except Exception:
            pass
        return None

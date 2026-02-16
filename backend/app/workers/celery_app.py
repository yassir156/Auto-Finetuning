"""
FineTuneFlow — Celery Application Configuration.

Two queues:
  - default: dataset generation, chunking, eval, export
  - gpu_queue: training (concurrency=1 to prevent OOM)
"""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "finetuneflow",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# ── Celery Configuration ─────────────────────────────────
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing
    task_routes={
        "app.workers.tasks_train.*": {"queue": "gpu_queue"},
        "app.workers.tasks_inference.*": {"queue": "gpu_queue"},
        "app.workers.tasks_dataset.*": {"queue": "default"},
        "app.workers.tasks_eval.*": {"queue": "gpu_queue"},
        "app.workers.tasks_export.*": {"queue": "default"},
    },

    # Task defaults
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,

    # Result backend
    result_expires=86400,  # 24 hours

    # Concurrency per queue (set via CLI: --queues=default,gpu_queue --concurrency=2)
    # gpu_queue should use --concurrency=1 when starting a GPU worker
)

# Task modules to import.
# Workers set WORKER_QUEUES env var so we only import what's needed.
# This avoids importing torch/ML deps in the default (non-GPU) worker.
import os as _os

_queues = _os.environ.get("WORKER_QUEUES", "all")

if _queues == "default":
    celery_app.conf.include = [
        "app.workers.tasks_dataset",
        "app.workers.tasks_export",
    ]
elif _queues == "gpu_queue":
    celery_app.conf.include = [
        "app.workers.tasks_train",
        "app.workers.tasks_eval",
        "app.workers.tasks_inference",
    ]
else:
    # "all" — used by GPU workers or when everything runs together
    celery_app.conf.include = [
        "app.workers.tasks_dataset",
        "app.workers.tasks_train",
        "app.workers.tasks_eval",
        "app.workers.tasks_export",
        "app.workers.tasks_inference",
    ]


@celery_app.task(name="ping")
def ping() -> str:
    """Health check task."""
    return "pong"

"""
FineTuneFlow — Celery Tasks: Training.

Tasks:
  - train: Fine-tune a model using SFTTrainer + PEFT (LoRA/QLoRA)

Pipeline:
  1. Load project / run from DB
  2. Export dataset to JSONL (train.jsonl + eval.jsonl)
  3. Build TrainingConfig
  4. Run SFTEngine.run()
  5. Store metrics & update DB status
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone

from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import settings
from app.core.exceptions import (
    DatasetTooFewExamplesError,
    GPUNotAvailableError,
)
from app.core.logging import get_logger
from app.db.models import Job, JobStatus, Project, ProjectStatus, Run
from app.db.session import SessionLocal
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _safe_update_job(db, job_id: str, **kwargs):
    """Update job fields, swallowing errors to avoid masking the original exception."""
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            db.commit()
        return job
    except Exception:
        logger.exception("task.train.job_update_failed", job_id=job_id)
        try:
            db.rollback()
        except Exception:
            pass
        return None


def _safe_update_run(db, run_id: str, **kwargs):
    """Update run fields, swallowing errors."""
    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if run:
            for k, v in kwargs.items():
                setattr(run, k, v)
            db.commit()
        return run
    except Exception:
        logger.exception("task.train.run_update_failed", run_id=run_id)
        try:
            db.rollback()
        except Exception:
            pass
        return None


# ════════════════════════════════════════════════════════════
#  Task: train
# ════════════════════════════════════════════════════════════


@celery_app.task(
    name="app.workers.tasks_train.train",
    bind=True,
    max_retries=0,          # NO auto-retry for training (too expensive)
    time_limit=86400,       # 24h hard limit
    soft_time_limit=85800,
    queue="gpu_queue",
)
def train(self, project_id: str, run_id: str) -> dict:
    """
    Fine-tune a model using SFTTrainer + PEFT.

    Steps:
      1. Load project/run from DB, validate state
      2. Export dataset to JSONL
      3. Build SFTEngine + TrainingConfig
      4. Run training
      5. Store metrics, update DB
    """
    logger.info("task.train.start", project_id=project_id, run_id=run_id)
    job_id = self.request.id
    db_gen = _get_db()
    db = next(db_gen)

    try:
        # ── 1. Load project + run ────────────────────────
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} not found")

        if not project.base_model_id:
            raise ValueError("No base_model_id configured on project")

        # Update status to running
        now = datetime.now(timezone.utc)
        _safe_update_job(db, job_id, status=JobStatus.running, started_at=now, progress_pct=5)
        _safe_update_run(db, run_id, status=JobStatus.running, started_at=now)
        project.status = ProjectStatus.training
        db.commit()

        # ── 2. Export dataset to JSONL ───────────────────
        from app.ml.dataset_exporter import export_dataset_to_jsonl

        dataset_dir = settings.storage_path / str(project_id) / "dataset"
        export_result = export_dataset_to_jsonl(db, project_id, dataset_dir)

        if export_result["train_count"] < 10:
            raise DatasetTooFewExamplesError(export_result["train_count"], minimum=10)

        _safe_update_job(db, job_id, progress_pct=15)
        _safe_update_run(
            db, run_id,
            num_train_examples=export_result["train_count"],
            num_eval_examples=export_result["eval_count"],
        )

        logger.info(
            "task.train.dataset_exported",
            train_count=export_result["train_count"],
            eval_count=export_result["eval_count"],
        )

        # ── 3. Build training config ────────────────────
        import torch
        from app.ml.sft_engine import CPU_ADJUSTMENTS, SFTEngine, TrainingConfig

        cpu_mode = not torch.cuda.is_available()
        if cpu_mode:
            logger.info("task.train.cpu_mode", project_id=project_id)

        output_dir = str(settings.storage_path / str(project_id) / "runs" / str(run_id))
        channel = f"train_logs:{project_id}"

        # On CPU: merge CPU-safe hyperparams (user overrides still take precedence)
        hyperparams = run.hyperparams or {}
        if cpu_mode:
            merged_hp = {**CPU_ADJUSTMENTS, **hyperparams}
        else:
            merged_hp = hyperparams

        config = TrainingConfig.from_hyperparams(
            base_model_id=project.base_model_id,
            method=run.method.value,
            output_dir=output_dir,
            train_file=export_result["train_file"],
            eval_file=export_result["eval_file"],
            hyperparams=merged_hp,
            redis_url=settings.REDIS_URL,
            pubsub_channel=channel,
            hf_token=settings.HF_TOKEN,
        )

        # Collect hardware info
        hardware_info = _get_hardware_info()
        hardware_info["cpu_mode"] = cpu_mode
        _safe_update_run(db, run_id, hardware_info=hardware_info)
        _safe_update_job(db, job_id, progress_pct=20)

        # ── 4. Run training ─────────────────────────────
        engine = SFTEngine(config)
        result = engine.run()

        # ── 5. Store results ────────────────────────────
        finished_at = datetime.now(timezone.utc)

        _safe_update_job(
            db, job_id,
            status=JobStatus.success,
            progress_pct=100,
            finished_at=finished_at,
            result_summary={
                "train_loss": result["metrics"].get("train_loss"),
                "eval_loss": result["metrics"].get("eval_loss"),
                "perplexity": result["metrics"].get("perplexity"),
                "duration_seconds": result["duration_seconds"],
            },
        )

        _safe_update_run(
            db, run_id,
            status=JobStatus.success,
            metrics=result["metrics"],
            artifacts_dir=result["artifacts_dir"],
            finished_at=finished_at,
            duration_seconds=int(result["duration_seconds"]),
        )

        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.completed
            db.commit()

        logger.info(
            "task.train.success",
            project_id=project_id,
            run_id=run_id,
            duration=result["duration_seconds"],
            train_loss=result["metrics"].get("train_loss"),
            eval_loss=result["metrics"].get("eval_loss"),
        )

        return {
            "status": "success",
            "project_id": project_id,
            "run_id": run_id,
            "metrics": result["metrics"],
            "duration_seconds": result["duration_seconds"],
        }

    except SoftTimeLimitExceeded:
        logger.error("task.train.timeout", project_id=project_id, run_id=run_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message="Training exceeded time limit (24h)",
            error_code="TRAINING_TIMEOUT",
            finished_at=datetime.now(timezone.utc),
        )
        _safe_update_run(db, run_id, status=JobStatus.failed, finished_at=datetime.now(timezone.utc))
        _set_project_failed(db, project_id)
        return {"status": "failed", "error": "timeout"}

    except (DatasetTooFewExamplesError, GPUNotAvailableError) as exc:
        logger.error("task.train.validation_error", error=str(exc))
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=str(exc),
            error_code=exc.error_code,
            finished_at=datetime.now(timezone.utc),
        )
        _safe_update_run(db, run_id, status=JobStatus.failed, finished_at=datetime.now(timezone.utc))
        _set_project_failed(db, project_id)
        return {"status": "failed", "error": str(exc)}

    except Exception as exc:
        tb = traceback.format_exc()
        error_str = str(exc)
        logger.exception("task.train.failed", project_id=project_id, run_id=run_id)

        # Detect OOM
        error_code = "TRAINING_ERROR"
        error_lower = error_str.lower()
        if "cuda out of memory" in error_lower or "outofmemoryerror" in error_lower:
            error_code = "TRAINING_OOM"
        elif isinstance(exc, (FloatingPointError,)):
            error_code = "TRAINING_DIVERGED"

        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=error_str[:2000],
            error_code=error_code,
            finished_at=datetime.now(timezone.utc),
        )
        _safe_update_run(db, run_id, status=JobStatus.failed, finished_at=datetime.now(timezone.utc))
        _set_project_failed(db, project_id)

        return {"status": "failed", "error": error_str[:500]}

    finally:
        try:
            next(db_gen, None)
        except Exception:
            pass


def _set_project_failed(db, project_id: str):
    """Set project status to failed."""
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass


def _get_hardware_info() -> dict:
    """Gather GPU hardware info (non-fatal)."""
    info = {}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
            info["torch_version"] = torch.__version__
            info["torch_cuda"] = torch.version.cuda
    except Exception:
        info["cuda_available"] = False
    return info

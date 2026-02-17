"""
FineTuneFlow — Celery Tasks: Export.

Tasks:
  - export_artifacts: Package training artifacts into a downloadable export.
    Creates adapter files, training config, metrics, report, and a zip archive.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Job, JobStatus, Project, Run
from app.db.session import SessionLocal
from app.workers.celery_app import celery_app
from app.workers.utils import safe_update_job as _safe_update_job

logger = get_logger(__name__)


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@celery_app.task(
    name="app.workers.tasks_export.export_artifacts",
    bind=True,
    max_retries=2,
    default_retry_delay=10,
    time_limit=600,
    soft_time_limit=540,
)
def export_artifacts(self, project_id: str, run_id: str) -> dict:
    """
    Package training artifacts into a downloadable export.

    Produces:
      export/
        adapter_model.safetensors  (or .bin)
        adapter_config.json
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json
        training_config.json
        metrics.json
        report.md
        finetuneflow_export.zip
    """
    logger.info("task.export.start", project_id=project_id, run_id=run_id)
    job_id = self.request.id
    db_gen = _get_db()
    db = next(db_gen)

    try:
        run = db.query(Run).filter(Run.id == run_id).first()
        if not run:
            raise ValueError(f"Run {run_id} not found")

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        _safe_update_job(
            db, job_id,
            status=JobStatus.running,
            started_at=datetime.now(timezone.utc),
            progress_pct=10,
        )

        # Paths
        project_dir = settings.storage_path / str(project_id)
        export_dir = project_dir / "export"
        adapter_dir = Path(run.artifacts_dir) if run.artifacts_dir else None

        # Clean previous export — with sandboxing check
        storage_root = settings.storage_path.resolve()
        if export_dir.resolve().is_relative_to(storage_root) and export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        _safe_update_job(db, job_id, progress_pct=20)

        # ── 1. Copy adapter files ────────────────────────
        copied_files = []
        if adapter_dir and adapter_dir.exists():
            adapter_extensions = {
                ".safetensors", ".bin", ".json", ".model", ".txt",
            }
            for f in adapter_dir.iterdir():
                if f.is_file() and f.suffix in adapter_extensions:
                    dest = export_dir / f.name
                    shutil.copy2(f, dest)
                    copied_files.append(f.name)
                    logger.debug("task.export.copied_file", file=f.name)
        else:
            logger.warning("task.export.no_adapter_dir", run_id=run_id)

        _safe_update_job(db, job_id, progress_pct=40)

        # ── 2. Write training_config.json ────────────────
        training_config = {
            "base_model_id": project.base_model_id,
            "method": run.method.value,
            "hyperparams": run.hyperparams or {},
            "hardware_info": run.hardware_info or {},
            "num_train_examples": run.num_train_examples,
            "num_eval_examples": run.num_eval_examples,
            "project_name": project.name,
            "task_type": project.task_type.value if hasattr(project.task_type, "value") else str(project.task_type),
        }
        config_path = export_dir / "training_config.json"
        config_path.write_text(
            json.dumps(training_config, indent=2, default=str),
            encoding="utf-8",
        )
        copied_files.append("training_config.json")

        # ── 3. Write metrics.json ────────────────────────
        metrics = run.metrics or {}
        metrics_path = export_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, default=str),
            encoding="utf-8",
        )
        copied_files.append("metrics.json")

        _safe_update_job(db, job_id, progress_pct=60)

        # ── 4. Generate report.md ────────────────────────
        try:
            from app.services.report_generator import generate_report_file

            generate_report_file(
                output_path=export_dir / "report.md",
                project_name=project.name,
                task_type=project.task_type.value if hasattr(project.task_type, "value") else str(project.task_type),
                base_model_id=project.base_model_id or "N/A",
                method=run.method.value,
                hyperparams=run.hyperparams or {},
                metrics=metrics,
                hardware_info=run.hardware_info or {},
                num_train_examples=run.num_train_examples,
                num_eval_examples=run.num_eval_examples,
                duration_seconds=run.duration_seconds,
                inference_samples=metrics.get("inference_samples"),
            )
            copied_files.append("report.md")
        except Exception:
            logger.exception("task.export.report_generation_failed")

        _safe_update_job(db, job_id, progress_pct=80)

        # ── 5. Create zip archive ────────────────────────
        zip_path = export_dir / "finetuneflow_export.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in export_dir.iterdir():
                if f.is_file() and f.name != "finetuneflow_export.zip":
                    zf.write(f, f.name)

        copied_files.append("finetuneflow_export.zip")

        _safe_update_job(
            db, job_id,
            status=JobStatus.success,
            progress_pct=100,
            finished_at=datetime.now(timezone.utc),
            result_summary={
                "export_dir": str(export_dir),
                "files": copied_files,
                "zip_path": str(zip_path),
                "zip_size_bytes": zip_path.stat().st_size,
            },
        )

        logger.info(
            "task.export.success",
            project_id=project_id,
            run_id=run_id,
            num_files=len(copied_files),
        )

        return {
            "status": "success",
            "files": copied_files,
            "zip_size_bytes": zip_path.stat().st_size,
        }

    except SoftTimeLimitExceeded:
        logger.error("task.export.timeout", project_id=project_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message="Export exceeded time limit",
            error_code="EXPORT_TIMEOUT",
            finished_at=datetime.now(timezone.utc),
        )
        return {"status": "failed", "error": "timeout"}

    except Exception as exc:
        logger.exception("task.export.failed", project_id=project_id, run_id=run_id)

        # Retry for transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)

        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=str(exc)[:2000],
            error_code="EXPORT_ERROR",
            finished_at=datetime.now(timezone.utc),
        )
        return {"status": "failed", "error": str(exc)[:500]}

    finally:
        try:
            next(db_gen, None)
        except Exception:
            pass

"""
FineTuneFlow — Celery Tasks: Evaluation.

Tasks:
  - evaluate: Run evaluation on eval set after training completes.
    Computes perplexity and sample inference.
"""

from __future__ import annotations

from datetime import datetime, timezone

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
    name="app.workers.tasks_eval.evaluate",
    bind=True,
    max_retries=1,
    time_limit=3600,
    soft_time_limit=3500,
    queue="gpu_queue",
)
def evaluate(self, project_id: str, run_id: str) -> dict:
    """
    Run evaluation on eval set after training.

    Steps:
      1. Load run from DB (get adapter dir, metrics, config)
      2. Check eval.jsonl exists
      3. Run EvalEngine (perplexity + sample inference)
      4. Merge eval metrics into run.metrics
      5. Update DB
    """
    logger.info("task.evaluate.start", project_id=project_id, run_id=run_id)
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

        # Find eval file
        eval_file = settings.storage_path / str(project_id) / "dataset" / "eval.jsonl"
        if not eval_file.exists():
            logger.warning("task.evaluate.no_eval_file", project_id=project_id)
            _safe_update_job(
                db, job_id,
                status=JobStatus.success,
                progress_pct=100,
                finished_at=datetime.now(timezone.utc),
                result_summary={"note": "No eval file found, skipping evaluation"},
            )
            return {"status": "skipped", "reason": "no eval file"}

        # Find adapter dir
        adapter_dir = run.artifacts_dir
        if not adapter_dir:
            raise ValueError("Run has no artifacts_dir — training may not have completed")

        _safe_update_job(db, job_id, progress_pct=20)

        # Run evaluation
        from app.ml.eval_engine import EvalEngine

        engine = EvalEngine(
            base_model_id=project.base_model_id,
            adapter_dir=adapter_dir,
            eval_file=str(eval_file),
            method=run.method.value,
            max_seq_length=(run.hyperparams or {}).get("max_seq_length", 2048),
            hf_token=settings.HF_TOKEN,
            train_metrics=run.metrics or {},
        )

        _safe_update_job(db, job_id, progress_pct=40)
        eval_result = engine.run()
        _safe_update_job(db, job_id, progress_pct=90)

        # Merge eval results into run metrics
        updated_metrics = dict(run.metrics) if run.metrics else {}
        if eval_result.get("perplexity") is not None:
            updated_metrics["perplexity"] = eval_result["perplexity"]
        if eval_result.get("inference_samples"):
            updated_metrics["inference_samples"] = eval_result["inference_samples"]

        run.metrics = updated_metrics
        db.commit()

        # NOTE: We do NOT set project.status = completed here.
        # The train task handles that. Eval is a follow-up step.

        _safe_update_job(
            db, job_id,
            status=JobStatus.success,
            progress_pct=100,
            finished_at=datetime.now(timezone.utc),
            result_summary={
                "perplexity": eval_result.get("perplexity"),
                "num_inference_samples": len(eval_result.get("inference_samples", [])),
                "eval_duration_seconds": eval_result.get("duration_seconds"),
            },
        )

        logger.info(
            "task.evaluate.success",
            project_id=project_id,
            run_id=run_id,
            perplexity=eval_result.get("perplexity"),
        )

        return {
            "status": "success",
            "perplexity": eval_result.get("perplexity"),
            "num_inference_samples": len(eval_result.get("inference_samples", [])),
        }

    except SoftTimeLimitExceeded:
        logger.error("task.evaluate.timeout", project_id=project_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message="Evaluation exceeded time limit",
            error_code="EVAL_TIMEOUT",
            finished_at=datetime.now(timezone.utc),
        )
        return {"status": "failed", "error": "timeout"}

    except Exception as exc:
        logger.exception("task.evaluate.failed", project_id=project_id, run_id=run_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=str(exc)[:2000],
            error_code="EVAL_ERROR",
            finished_at=datetime.now(timezone.utc),
        )
        return {"status": "failed", "error": str(exc)[:500]}

    finally:
        try:
            next(db_gen, None)
        except Exception:
            pass

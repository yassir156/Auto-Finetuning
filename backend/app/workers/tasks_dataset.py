"""
FineTuneFlow — Celery Tasks: Dataset Generation.

Tasks:
  - chunk_documents: Extract text from docs and create chunks
  - dataset_preview: Generate a small preview dataset via Ollama
  - dataset_generate: Generate full dataset via Ollama
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from celery import current_task
from celery.exceptions import SoftTimeLimitExceeded
from sqlalchemy import func as sa_func

from app.core.config import settings
from app.core.exceptions import OllamaAPIError
from app.core.logging import get_logger
from app.db.models import (
    Chunk,
    DatasetExample,
    DatasetSplit,
    File,
    FileKind,
    FileStatus,
    Job,
    JobStatus,
    Project,
    ProjectStatus,
)
from app.db.session import SessionLocal
from app.services.chunker import chunk_text, count_tokens
from app.services.ollama_client import generate_examples_from_chunk
from app.services.task_registry import get_task_config, resolve_task_key
from app.services.text_extractor import extract_text
from app.workers.celery_app import celery_app
from app.workers.utils import safe_update_job as _safe_update_job

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  Helper: get a fresh DB session for Celery tasks
# ════════════════════════════════════════════════════════════

def _get_db():
    """Create a fresh DB session for Celery tasks (context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _update_job(db, job_id: str, **kwargs):
    """Update job fields."""
    job = db.query(Job).filter(Job.id == job_id).first()
    if job:
        for k, v in kwargs.items():
            setattr(job, k, v)
        db.commit()
    return job


# ════════════════════════════════════════════════════════════
#  Task: chunk_documents
# ════════════════════════════════════════════════════════════


@celery_app.task(
    name="app.workers.tasks_dataset.chunk_documents",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    time_limit=600,       # 10 min hard limit
    soft_time_limit=540,  # 9 min soft limit
)
def chunk_documents(
    self,
    project_id: str,
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
) -> dict:
    """
    Extract text from uploaded documents and create chunks in DB.

    Steps:
    1. Mark job as running, project as chunking
    2. Query all raw_doc files for the project
    3. For each file: extract text → split into chunks → store in DB
    4. Update job as success + project as ready/generating
    """
    job_id = self.request.id
    db = SessionLocal()

    try:
        # ── Mark job running ──
        _update_job(
            db, job_id,
            status=JobStatus.running,
            started_at=datetime.now(timezone.utc),
        )
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        project.status = ProjectStatus.chunking
        db.commit()

        logger.info("task.chunk_documents.start", project_id=project_id)

        # ── Get all raw docs ──
        files = (
            db.query(File)
            .filter(
                File.project_id == project_id,
                File.kind == FileKind.raw_doc,
                File.status == FileStatus.ready,
            )
            .all()
        )

        if not files:
            raise ValueError(f"No raw documents found for project {project_id}")

        # ── Delete existing chunks (re-chunking) ──
        db.query(Chunk).filter(Chunk.project_id == project_id).delete()
        db.commit()

        total_chunks = 0
        total_tokens = 0
        files_processed = 0
        errors: list[dict] = []

        for file_record in files:
            try:
                file_path = Path(file_record.storage_path)
                if not file_path.exists():
                    # Try relative to storage root
                    file_path = settings.storage_path / file_record.storage_path
                if not file_path.exists():
                    errors.append({
                        "file_id": str(file_record.id),
                        "filename": file_record.filename,
                        "error": "File not found on disk",
                    })
                    continue

                # ── Extract text ──
                text = extract_text(file_path, file_record.mime_type)
                if not text.strip():
                    errors.append({
                        "file_id": str(file_record.id),
                        "filename": file_record.filename,
                        "error": "No text extracted (empty document)",
                    })
                    continue

                # ── Chunk the text ──
                chunks = chunk_text(
                    text,
                    chunk_size_tokens=chunk_size_tokens,
                    chunk_overlap_tokens=chunk_overlap_tokens,
                )

                # ── Store chunks in DB ──
                for tc in chunks:
                    chunk_record = Chunk(
                        project_id=project_id,
                        source_file_id=file_record.id,
                        chunk_index=tc.index,
                        content=tc.content,
                        token_count=tc.token_count,
                        char_count=tc.char_count,
                        metadata_={
                            "start_char": tc.start_char,
                            "end_char": tc.end_char,
                            "source_filename": file_record.filename,
                        },
                    )
                    db.add(chunk_record)
                    total_chunks += 1
                    total_tokens += tc.token_count

                db.commit()
                files_processed += 1

                # ── Update progress ──
                progress = int((files_processed / len(files)) * 100)
                _update_job(db, job_id, progress_pct=progress)
                self.update_state(
                    state="PROGRESS",
                    meta={"progress": progress, "files_processed": files_processed},
                )

            except Exception as e:
                logger.error(
                    "task.chunk_documents.file_error",
                    file_id=str(file_record.id),
                    error=str(e),
                )
                errors.append({
                    "file_id": str(file_record.id),
                    "filename": file_record.filename,
                    "error": str(e),
                })
                continue

        if total_chunks == 0:
            raise ValueError("No chunks produced from any document")

        # ── Finalize ──
        result_summary = {
            "files_processed": files_processed,
            "total_files": len(files),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": round(total_tokens / total_chunks, 1) if total_chunks else 0,
            "errors": errors,
        }

        _update_job(
            db, job_id,
            status=JobStatus.success,
            progress_pct=100,
            result_summary=result_summary,
            finished_at=datetime.now(timezone.utc),
        )

        # Move project status forward
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.generating
            project.error_message = None
            db.commit()

        logger.info(
            "task.chunk_documents.done",
            project_id=project_id,
            chunks=total_chunks,
            tokens=total_tokens,
        )
        return result_summary

    except SoftTimeLimitExceeded:
        logger.error("task.chunk_documents.timeout", project_id=project_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message="Chunking task exceeded time limit",
            error_code="TASK_TIMEOUT",
            finished_at=datetime.now(timezone.utc),
        )
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            project.error_message = "Chunking task exceeded time limit"
            try:
                db.commit()
            except Exception:
                db.rollback()
        raise

    except Exception as exc:
        logger.error("task.chunk_documents.failed", project_id=project_id, error=str(exc))
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=str(exc),
            error_code="CHUNKING_FAILED",
            finished_at=datetime.now(timezone.utc),
        )
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            project.error_message = str(exc)
            try:
                db.commit()
            except Exception:
                db.rollback()
        raise

    finally:
        db.close()


# ════════════════════════════════════════════════════════════
#  Validation helpers for dataset examples
# ════════════════════════════════════════════════════════════


def _validate_example(data: dict, task_type: str = "instruction_tuning") -> tuple[bool, str | None]:
    """
    Validate a single dataset example against the task type schema.

    Returns (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Not a dict"

    config = get_task_config(task_type)

    # Check required fields
    for field_name in config.required_fields:
        val = data.get(field_name)
        if val is None:
            return False, f"Missing required field '{field_name}'"
        # For string fields, check non-empty
        if isinstance(val, str) and not val.strip():
            return False, f"Empty required field '{field_name}'"
        # For list/dict fields, check non-empty
        if isinstance(val, (list, dict)) and len(val) == 0:
            return False, f"Empty required field '{field_name}'"

    # Check main output field minimum length
    main_val = data.get(config.main_output_field, "")
    if isinstance(main_val, str):
        if len(main_val.strip()) < config.min_output_len:
            return False, f"'{config.main_output_field}' too short (min {config.min_output_len} chars)"
    elif isinstance(main_val, (dict, list)):
        import json as _json
        serialized = _json.dumps(main_val, ensure_ascii=False)
        if len(serialized) < config.min_output_len:
            return False, f"'{config.main_output_field}' too short"

    # Check max length on all string fields
    for key, val in data.items():
        if isinstance(val, str) and len(val) > config.max_field_len:
            return False, f"'{key}' too long (max {config.max_field_len} chars)"

    # Check for LLM refusal patterns in string values
    refusal_patterns = ["i cannot", "i can't", "as an ai", "i'm sorry, but"]
    for key, val in data.items():
        if isinstance(val, str):
            lower_val = val.lower()
            for pattern in refusal_patterns:
                if pattern in lower_val:
                    return False, f"'{key}' contains refusal pattern: '{pattern}'"

    return True, None


def _content_hash(data: dict, task_type: str = "instruction_tuning") -> str:
    """Compute a SHA-256 hash of the example content for deduplication."""
    config = get_task_config(task_type)
    # Use all required + optional fields for the hash
    all_fields = config.required_fields + config.optional_fields
    canonical_data = {}
    for f in sorted(all_fields):
        val = data.get(f, "")
        if isinstance(val, str):
            canonical_data[f] = val.strip()
        else:
            canonical_data[f] = val
    canonical = json.dumps(canonical_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ════════════════════════════════════════════════════════════
#  Task: dataset_preview
# ════════════════════════════════════════════════════════════


@celery_app.task(
    name="app.workers.tasks_dataset.dataset_preview",
    bind=True,
    max_retries=3,
    default_retry_delay=15,
    time_limit=300,
    soft_time_limit=270,
)
def dataset_preview(
    self,
    project_id: str,
    num_examples: int = 10,
) -> dict:
    """
    Generate a small preview dataset from a random chunk.

    Steps:
    1. Pick a random chunk from the project
    2. Call Ollama to generate examples
    3. Validate & store with split=preview
    4. Return results
    """
    job_id = self.request.id
    db = SessionLocal()

    try:
        _update_job(
            db, job_id,
            status=JobStatus.running,
            started_at=datetime.now(timezone.utc),
        )

        logger.info("task.dataset_preview.start", project_id=project_id, num_examples=num_examples)

        # ── Get project + task_type ──
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        # ── Pick a random chunk ──
        chunk_count = db.query(sa_func.count(Chunk.id)).filter(
            Chunk.project_id == project_id
        ).scalar()

        if not chunk_count:
            raise ValueError(f"No chunks found for project {project_id}. Chunk documents first.")

        # Pick a random chunk with content
        random_offset = random.randint(0, max(0, chunk_count - 1))
        chunk = (
            db.query(Chunk)
            .filter(Chunk.project_id == project_id)
            .order_by(Chunk.chunk_index)
            .offset(random_offset)
            .limit(1)
            .first()
        )

        if not chunk:
            raise ValueError("Failed to select a chunk for preview")

        # ── Call Ollama ──
        task_type = project.task_type.value if project.task_type else "instruction_tuning"
        examples = generate_examples_from_chunk(
            chunk_text=chunk.content,
            task_type=task_type,
            num_examples=num_examples,
        )

        # ── Delete old previews ──
        db.query(DatasetExample).filter(
            DatasetExample.project_id == project_id,
            DatasetExample.split == DatasetSplit.preview,
        ).delete()
        db.commit()

        # ── Validate & store ──
        valid_count = 0
        invalid_count = 0
        stored_examples = []

        for ex_data in examples:
            is_valid, error_msg = _validate_example(ex_data, task_type=task_type)
            token_ct = count_tokens(
                json.dumps(ex_data, ensure_ascii=False)
            )
            ch = _content_hash(ex_data, task_type=task_type)

            example = DatasetExample(
                project_id=project_id,
                job_id=job_id,
                source_chunk_id=chunk.id,
                split=DatasetSplit.preview,
                data=ex_data,
                is_valid=is_valid,
                validation_error=error_msg,
                token_count=token_ct,
                content_hash=ch,
            )
            db.add(example)
            stored_examples.append(ex_data)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        db.commit()

        result_summary = {
            "total_generated": len(examples),
            "valid": valid_count,
            "invalid": invalid_count,
            "chunk_id": str(chunk.id),
            "examples": stored_examples[:5],  # Return first 5 as preview
        }

        _update_job(
            db, job_id,
            status=JobStatus.success,
            progress_pct=100,
            result_summary=result_summary,
            finished_at=datetime.now(timezone.utc),
        )

        logger.info(
            "task.dataset_preview.done",
            project_id=project_id,
            valid=valid_count,
            invalid=invalid_count,
        )
        return result_summary

    except SoftTimeLimitExceeded:
        logger.error("task.dataset_preview.timeout", project_id=project_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message="Preview task exceeded time limit",
            error_code="TASK_TIMEOUT",
            finished_at=datetime.now(timezone.utc),
        )
        raise

    except (OllamaAPIError, ConnectionError, TimeoutError) as exc:
        logger.warning(
            "task.dataset_preview.retryable_error",
            project_id=project_id,
            error=str(exc),
            retry=self.request.retries,
        )
        _safe_update_job(
            db, job_id,
            status=JobStatus.retrying,
            error_message=str(exc),
            retry_count=self.request.retries + 1,
        )
        try:
            raise self.retry(exc=exc, countdown=15 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            _safe_update_job(
                db, job_id,
                status=JobStatus.failed,
                error_message=f"Failed after {self.max_retries} retries: {exc}",
                error_code="DATASET_GENERATION_FAILED",
                finished_at=datetime.now(timezone.utc),
            )
            raise

    except Exception as exc:
        logger.error("task.dataset_preview.failed", project_id=project_id, error=str(exc))
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=str(exc),
            error_code="DATASET_GENERATION_FAILED",
            finished_at=datetime.now(timezone.utc),
        )
        raise

    finally:
        db.close()


# ════════════════════════════════════════════════════════════
#  Task: dataset_generate
# ════════════════════════════════════════════════════════════


@celery_app.task(
    name="app.workers.tasks_dataset.dataset_generate",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    time_limit=7200,       # 2h hard limit
    soft_time_limit=7000,
)
def dataset_generate(
    self,
    project_id: str,
    num_examples_target: int | None = None,
    examples_per_chunk: int | None = None,
    ollama_model: str | None = None,
) -> dict:
    """
    Generate the full dataset from all chunks.

    Steps:
    1. Mark job running, project as generating
    2. Calculate how many examples per chunk
    3. Iterate chunks, call Ollama for each
    4. Validate, deduplicate, split train/eval
    5. Store all examples in DB
    """
    job_id = self.request.id
    db = SessionLocal()

    try:
        _update_job(
            db, job_id,
            status=JobStatus.running,
            started_at=datetime.now(timezone.utc),
        )

        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise ValueError(f"Project {project_id} not found")

        project.status = ProjectStatus.generating
        db.commit()

        target = num_examples_target or settings.DEFAULT_NUM_EXAMPLES_TARGET
        per_chunk = examples_per_chunk or settings.DEFAULT_EXAMPLES_PER_CHUNK
        task_type = project.task_type.value if project.task_type else "instruction_tuning"

        logger.info(
            "task.dataset_generate.start",
            project_id=project_id,
            target=target,
            per_chunk=per_chunk,
        )

        # ── Get all chunks ──
        chunks = (
            db.query(Chunk)
            .filter(Chunk.project_id == project_id)
            .order_by(Chunk.chunk_index)
            .all()
        )

        if not chunks:
            raise ValueError(f"No chunks found for project {project_id}")

        # ── Delete old generated examples (re-generation) ──
        db.query(DatasetExample).filter(
            DatasetExample.project_id == project_id,
            DatasetExample.split.in_([DatasetSplit.train, DatasetSplit.eval]),
        ).delete(synchronize_session="fetch")
        db.commit()

        # ── Calculate distribution ──
        # Try to reach target by distributing across chunks
        max_per_chunk = max(per_chunk, target // len(chunks) + 1)
        # But cap at a reasonable max to avoid huge prompts
        max_per_chunk = min(max_per_chunk, 20)

        all_examples: list[dict] = []
        seen_hashes: set[str] = set()
        duplicates_skipped = 0
        chunk_errors: list[dict] = []
        chunks_processed = 0

        for chunk in chunks:
            if len(all_examples) >= target:
                break

            remaining = target - len(all_examples)
            n = min(max_per_chunk, remaining)

            try:
                examples = generate_examples_from_chunk(
                    chunk_text=chunk.content,
                    task_type=task_type,
                    num_examples=n,
                    model=ollama_model,
                )

                for ex_data in examples:
                    is_valid, error_msg = _validate_example(ex_data, task_type=task_type)

                    ch = _content_hash(ex_data, task_type=task_type)

                    # Dedup
                    if ch in seen_hashes:
                        duplicates_skipped += 1
                        continue
                    seen_hashes.add(ch)

                    token_ct = count_tokens(
                        json.dumps(ex_data, ensure_ascii=False)
                    )

                    all_examples.append({
                        "data": ex_data,
                        "is_valid": is_valid,
                        "validation_error": error_msg,
                        "token_count": token_ct,
                        "content_hash": ch,
                        "source_chunk_id": chunk.id,  # Keep UUID native
                    })

            except Exception as e:
                logger.warning(
                    "task.dataset_generate.chunk_error",
                    chunk_id=str(chunk.id),
                    error=str(e),
                )
                chunk_errors.append({
                    "chunk_id": str(chunk.id),
                    "chunk_index": chunk.chunk_index,
                    "error": str(e),
                })

            chunks_processed += 1
            progress = int((chunks_processed / len(chunks)) * 100)
            _update_job(db, job_id, progress_pct=min(progress, 95))
            self.update_state(
                state="PROGRESS",
                meta={"progress": progress, "chunks_processed": chunks_processed},
            )

        if not all_examples:
            raise ValueError("No examples generated from any chunk")

        # ── Split train/eval ──
        valid_examples = [e for e in all_examples if e["is_valid"]]
        invalid_examples = [e for e in all_examples if not e["is_valid"]]

        random.shuffle(valid_examples)
        split_ratio = settings.DEFAULT_TRAIN_EVAL_SPLIT
        split_idx = int(len(valid_examples) * split_ratio)
        train_examples = valid_examples[:split_idx]
        eval_examples = valid_examples[split_idx:]

        # ── Store in DB ──
        for ex in train_examples:
            db.add(DatasetExample(
                project_id=project_id,
                job_id=job_id,
                source_chunk_id=ex["source_chunk_id"],
                split=DatasetSplit.train,
                data=ex["data"],
                is_valid=True,
                token_count=ex["token_count"],
                content_hash=ex["content_hash"],
            ))

        for ex in eval_examples:
            db.add(DatasetExample(
                project_id=project_id,
                job_id=job_id,
                source_chunk_id=ex["source_chunk_id"],
                split=DatasetSplit.eval,
                data=ex["data"],
                is_valid=True,
                token_count=ex["token_count"],
                content_hash=ex["content_hash"],
            ))

        for ex in invalid_examples:
            db.add(DatasetExample(
                project_id=project_id,
                job_id=job_id,
                source_chunk_id=ex["source_chunk_id"],
                split=DatasetSplit.train,  # Invalid examples stored but flagged
                data=ex["data"],
                is_valid=False,
                validation_error=ex["validation_error"],
                token_count=ex["token_count"],
                content_hash=ex["content_hash"],
            ))

        db.commit()

        # ── Finalize ──
        result_summary = {
            "total_generated": len(all_examples),
            "valid": len(valid_examples),
            "invalid": len(invalid_examples),
            "duplicates_removed": duplicates_skipped,
            "train": len(train_examples),
            "eval": len(eval_examples),
            "chunks_processed": chunks_processed,
            "total_chunks": len(chunks),
            "errors": chunk_errors,
        }

        _update_job(
            db, job_id,
            status=JobStatus.success,
            progress_pct=100,
            result_summary=result_summary,
            finished_at=datetime.now(timezone.utc),
        )

        # Move project to ready_to_train
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.ready_to_train
            project.error_message = None
            db.commit()

        logger.info(
            "task.dataset_generate.done",
            project_id=project_id,
            total=len(all_examples),
            train=len(train_examples),
            eval=len(eval_examples),
        )
        return result_summary

    except SoftTimeLimitExceeded:
        logger.error("task.dataset_generate.timeout", project_id=project_id)
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message="Dataset generation task exceeded time limit",
            error_code="TASK_TIMEOUT",
            finished_at=datetime.now(timezone.utc),
        )
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            project.error_message = "Dataset generation exceeded time limit"
            try:
                db.commit()
            except Exception:
                db.rollback()
        raise

    except (OllamaAPIError, ConnectionError, TimeoutError) as exc:
        logger.warning(
            "task.dataset_generate.retryable_error",
            project_id=project_id,
            error=str(exc),
            retry=self.request.retries,
        )
        _safe_update_job(
            db, job_id,
            status=JobStatus.retrying,
            error_message=str(exc),
            retry_count=self.request.retries + 1,
        )
        try:
            raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))
        except self.MaxRetriesExceededError:
            _safe_update_job(
                db, job_id,
                status=JobStatus.failed,
                error_message=f"Failed after {self.max_retries} retries: {exc}",
                error_code="DATASET_GENERATION_FAILED",
                finished_at=datetime.now(timezone.utc),
            )
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = ProjectStatus.failed
                project.error_message = str(exc)
                try:
                    db.commit()
                except Exception:
                    db.rollback()
            raise

    except Exception as exc:
        logger.error("task.dataset_generate.failed", project_id=project_id, error=str(exc))
        _safe_update_job(
            db, job_id,
            status=JobStatus.failed,
            error_message=str(exc),
            error_code="DATASET_GENERATION_FAILED",
            finished_at=datetime.now(timezone.utc),
        )
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            project.error_message = str(exc)
            try:
                db.commit()
            except Exception:
                db.rollback()
        raise

    finally:
        db.close()

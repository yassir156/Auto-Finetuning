"""
FineTuneFlow — Dataset Exporter.

Exports DatasetExample rows from the database to train.jsonl / eval.jsonl
files on disk, ready for consumption by SFTTrainer.
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import DatasetExample, DatasetSplit, Project
from app.services.task_registry import to_sft_format

logger = get_logger(__name__)


def export_dataset_to_jsonl(
    db: Session,
    project_id: str,
    output_dir: str | Path,
) -> dict:
    """
    Export dataset examples to train.jsonl and eval.jsonl files.

    Each example is converted to SFT format (instruction/input/output)
    via the task_registry before writing.

    Args:
        db: SQLAlchemy session.
        project_id: Project UUID string.
        output_dir: Directory where train.jsonl / eval.jsonl will be written.

    Returns:
        dict with keys: train_file, eval_file, train_count, eval_count
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"

    train_count = 0
    eval_count = 0

    # Get the project to determine task_type
    project = db.query(Project).filter(Project.id == project_id).first()
    task_type = project.task_type.value if project and project.task_type else "instruction_tuning"

    # Query valid examples grouped by split — use yield_per to avoid OOM on large datasets
    query = (
        db.query(DatasetExample)
        .filter(
            DatasetExample.project_id == project_id,
            DatasetExample.is_valid.is_(True),
            DatasetExample.split.in_([DatasetSplit.train, DatasetSplit.eval]),
        )
        .order_by(DatasetExample.created_at)
        .yield_per(500)
    )

    with open(train_file, "w", encoding="utf-8") as tf, \
         open(eval_file, "w", encoding="utf-8") as ef:
        for ex in query:
            data = ex.data
            if not isinstance(data, dict):
                continue

            # Convert to SFT format for training
            sft_data = to_sft_format(task_type, data)
            line = json.dumps(sft_data, ensure_ascii=False)

            if ex.split == DatasetSplit.train:
                tf.write(line + "\n")
                train_count += 1
            elif ex.split == DatasetSplit.eval:
                ef.write(line + "\n")
                eval_count += 1

    logger.info(
        "dataset_exporter.export_complete",
        project_id=project_id,
        train_count=train_count,
        eval_count=eval_count,
        train_file=str(train_file),
        eval_file=str(eval_file),
    )

    result = {
        "train_file": str(train_file),
        "eval_file": str(eval_file) if eval_count > 0 else None,
        "train_count": train_count,
        "eval_count": eval_count,
    }

    # Remove empty eval file
    if eval_count == 0 and eval_file.exists():
        eval_file.unlink()

    return result

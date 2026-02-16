"""initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2025-01-01 00:00:00.000000

Creates all 6 tables + 8 enums for FineTuneFlow.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Enums ────────────────────────────────────────────
    task_type_enum = postgresql.ENUM("instruction", "qa", name="task_type", create_type=False)
    task_type_enum.create(op.get_bind(), checkfirst=True)

    project_status_enum = postgresql.ENUM(
        "draft", "uploading", "chunking", "generating",
        "ready_to_train", "training", "evaluating", "completed", "failed",
        name="project_status", create_type=False,
    )
    project_status_enum.create(op.get_bind(), checkfirst=True)

    file_kind_enum = postgresql.ENUM(
        "raw_doc", "dataset_upload", "dataset_generated", "artifact", "export", "log",
        name="file_kind", create_type=False,
    )
    file_kind_enum.create(op.get_bind(), checkfirst=True)

    file_status_enum = postgresql.ENUM(
        "uploading", "ready", "processing", "failed",
        name="file_status", create_type=False,
    )
    file_status_enum.create(op.get_bind(), checkfirst=True)

    job_type_enum = postgresql.ENUM(
        "chunking", "dataset_preview", "dataset_generate", "train", "eval", "export",
        name="job_type", create_type=False,
    )
    job_type_enum.create(op.get_bind(), checkfirst=True)

    job_status_enum = postgresql.ENUM(
        "queued", "running", "success", "failed", "retrying", "cancelled",
        name="job_status", create_type=False,
    )
    job_status_enum.create(op.get_bind(), checkfirst=True)

    finetune_method_enum = postgresql.ENUM(
        "lora", "qlora",
        name="finetune_method", create_type=False,
    )
    finetune_method_enum.create(op.get_bind(), checkfirst=True)

    dataset_split_enum = postgresql.ENUM(
        "preview", "train", "eval",
        name="dataset_split", create_type=False,
    )
    dataset_split_enum.create(op.get_bind(), checkfirst=True)

    # ── projects ─────────────────────────────────────────
    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("task_type", task_type_enum, nullable=False, server_default="instruction"),
        sa.Column("base_model_id", sa.String(255), nullable=True),
        sa.Column("model_info", postgresql.JSONB, nullable=True),
        sa.Column("status", project_status_enum, nullable=False, server_default="draft"),
        sa.Column("config", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_projects_status", "projects", ["status"])
    op.create_index("idx_projects_created", "projects", [sa.text("created_at DESC")])

    # ── files ────────────────────────────────────────────
    op.create_table(
        "files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", file_kind_enum, nullable=False),
        sa.Column("status", file_status_enum, nullable=False, server_default="ready"),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("mime_type", sa.String(128), nullable=True),
        sa.Column("storage_path", sa.String(1024), nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_files_project", "files", ["project_id"])
    op.create_index("idx_files_kind", "files", ["project_id", "kind"])

    # ── chunks ───────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("files.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("char_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("metadata", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.UniqueConstraint("source_file_id", "chunk_index", name="uq_chunk_file_index"),
    )
    op.create_index("idx_chunks_project", "chunks", ["project_id"])
    op.create_index("idx_chunks_source", "chunks", ["source_file_id"])

    # ── jobs ─────────────────────────────────────────────
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("type", job_type_enum, nullable=False),
        sa.Column("status", job_status_enum, nullable=False, server_default="queued"),
        sa.Column("celery_task_id", sa.String(255), nullable=True),
        sa.Column("progress_pct", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("result_summary", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("retry_count", sa.Integer, server_default="0"),
        sa.Column("max_retries", sa.Integer, server_default="3"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_jobs_project", "jobs", ["project_id"])
    op.create_index("idx_jobs_status", "jobs", ["project_id", "status"])
    op.create_index("idx_jobs_type", "jobs", ["project_id", "type"])

    # ── dataset_examples ─────────────────────────────────
    op.create_table(
        "dataset_examples",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("source_chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True),
        sa.Column("split", dataset_split_enum, nullable=False, server_default="train"),
        sa.Column("data", postgresql.JSONB, nullable=False),
        sa.Column("is_valid", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("validation_error", sa.Text, nullable=True),
        sa.Column("token_count", sa.Integer, server_default="0"),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_examples_project", "dataset_examples", ["project_id"])
    op.create_index("idx_examples_split", "dataset_examples", ["project_id", "split"])
    op.create_index("idx_examples_valid", "dataset_examples", ["project_id", "is_valid"])
    op.create_index("idx_examples_hash", "dataset_examples", ["content_hash"])

    # ── runs ─────────────────────────────────────────────
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("method", finetune_method_enum, nullable=False),
        sa.Column("status", job_status_enum, nullable=False, server_default="queued"),
        sa.Column("hyperparams", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("metrics", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("hardware_info", postgresql.JSONB, nullable=False, server_default="{}"),
        sa.Column("artifacts_dir", sa.String(1024), nullable=True),
        sa.Column("num_train_examples", sa.Integer, server_default="0"),
        sa.Column("num_eval_examples", sa.Integer, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("idx_runs_project", "runs", ["project_id"])
    op.create_index("idx_runs_job", "runs", ["job_id"])


def downgrade() -> None:
    op.drop_table("runs")
    op.drop_table("dataset_examples")
    op.drop_table("jobs")
    op.drop_table("chunks")
    op.drop_table("files")
    op.drop_table("projects")

    # Drop enums
    for name in [
        "dataset_split", "finetune_method", "job_status", "job_type",
        "file_status", "file_kind", "project_status", "task_type",
    ]:
        op.execute(f"DROP TYPE IF EXISTS {name}")

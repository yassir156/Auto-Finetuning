"""
FineTuneFlow — SQLAlchemy ORM Models.

All tables defined in DATABASE_SCHEMA.md are implemented here.
Enums use Python's stdlib enum and are mapped to PostgreSQL enums.
"""

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base


# ════════════════════════════════════════════════════════════
# Enums
# ════════════════════════════════════════════════════════════


class TaskType(str, enum.Enum):
    instruction_tuning = "instruction_tuning"
    qa_grounded = "qa_grounded"
    summarization = "summarization"
    report_generation = "report_generation"
    information_extraction = "information_extraction"
    classification = "classification"
    chat_dialogue_sft = "chat_dialogue_sft"


class ProjectStatus(str, enum.Enum):
    draft = "draft"
    uploading = "uploading"
    chunking = "chunking"
    generating = "generating"
    ready_to_train = "ready_to_train"
    training = "training"
    evaluating = "evaluating"
    completed = "completed"
    failed = "failed"


class FileKind(str, enum.Enum):
    raw_doc = "raw_doc"
    dataset_upload = "dataset_upload"
    dataset_generated = "dataset_generated"
    artifact = "artifact"
    export = "export"
    log = "log"


class FileStatus(str, enum.Enum):
    uploading = "uploading"
    ready = "ready"
    processing = "processing"
    failed = "failed"


class JobType(str, enum.Enum):
    chunking = "chunking"
    dataset_preview = "dataset_preview"
    dataset_generate = "dataset_generate"
    train = "train"
    eval = "eval"
    export = "export"


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    success = "success"
    failed = "failed"
    retrying = "retrying"
    cancelled = "cancelled"


class FinetuneMethod(str, enum.Enum):
    lora = "lora"
    qlora = "qlora"


class DatasetSplit(str, enum.Enum):
    preview = "preview"
    train = "train"
    eval = "eval"


# ════════════════════════════════════════════════════════════
# Models
# ════════════════════════════════════════════════════════════


class Project(Base):
    """A fine-tuning project — the top-level entity."""

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[TaskType] = mapped_column(
        Enum(TaskType, name="task_type", create_constraint=True),
        nullable=False,
        default=TaskType.instruction_tuning,
    )
    base_model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_info: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus, name="project_status", create_constraint=True),
        nullable=False,
        default=ProjectStatus.draft,
    )
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    files: Mapped[list["File"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    jobs: Mapped[list["Job"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    dataset_examples: Mapped[list["DatasetExample"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )
    runs: Mapped[list["Run"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_projects_status", "status"),
        Index("idx_projects_created", created_at.desc()),
    )


class File(Base):
    """A file on disk associated with a project."""

    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[FileKind] = mapped_column(
        Enum(FileKind, name="file_kind", create_constraint=True), nullable=False
    )
    status: Mapped[FileStatus] = mapped_column(
        Enum(FileStatus, name="file_status", create_constraint=True),
        nullable=False,
        default=FileStatus.ready,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="files")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="source_file", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("idx_files_project", "project_id"),
        Index("idx_files_kind", "project_id", "kind"),
    )


class Chunk(Base):
    """A text segment extracted from a document."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    source_file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("files.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="chunks")
    source_file: Mapped["File"] = relationship(back_populates="chunks")
    dataset_examples: Mapped[list["DatasetExample"]] = relationship(
        back_populates="source_chunk", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("source_file_id", "chunk_index", name="uq_chunk_file_index"),
        Index("idx_chunks_project", "project_id"),
        Index("idx_chunks_source", "source_file_id"),
    )


class Job(Base):
    """A long-running task managed by Celery."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    type: Mapped[JobType] = mapped_column(
        Enum(JobType, name="job_type", create_constraint=True), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", create_constraint=True),
        nullable=False,
        default=JobStatus.queued,
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    progress_pct: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result_summary: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="jobs")
    run: Mapped["Run | None"] = relationship(
        back_populates="job", uselist=False
    )

    __table_args__ = (
        Index("idx_jobs_project", "project_id"),
        Index("idx_jobs_status", "project_id", "status"),
        Index("idx_jobs_type", "project_id", "type"),
    )


class DatasetExample(Base):
    """A single example in the generated/uploaded dataset."""

    __tablename__ = "dataset_examples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True
    )
    split: Mapped[DatasetSplit] = mapped_column(
        Enum(DatasetSplit, name="dataset_split", create_constraint=True),
        nullable=False,
        default=DatasetSplit.train,
    )
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    validation_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="dataset_examples")
    source_chunk: Mapped["Chunk | None"] = relationship(back_populates="dataset_examples")
    job: Mapped["Job | None"] = relationship(foreign_keys=[job_id])

    __table_args__ = (
        Index("idx_examples_project", "project_id"),
        Index("idx_examples_split", "project_id", "split"),
        Index("idx_examples_valid", "project_id", "is_valid"),
        Index("idx_examples_hash", "content_hash"),
    )


class Run(Base):
    """A training run — associated with exactly one training job."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    method: Mapped[FinetuneMethod] = mapped_column(
        Enum(FinetuneMethod, name="finetune_method", create_constraint=True),
        nullable=False,
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", create_constraint=True, create_type=False),
        nullable=False,
        default=JobStatus.queued,
    )
    hyperparams: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    hardware_info: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    artifacts_dir: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    num_train_examples: Mapped[int] = mapped_column(Integer, default=0)
    num_eval_examples: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    project: Mapped["Project"] = relationship(back_populates="runs")
    job: Mapped["Job"] = relationship(back_populates="run")

    __table_args__ = (
        Index("idx_runs_project", "project_id"),
        Index("idx_runs_job", "job_id"),
    )

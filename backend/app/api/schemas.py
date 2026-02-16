"""
FineTuneFlow — Pydantic Schemas (Request / Response).

All DTOs for the REST API, grouped by domain.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ══════════════════════════════════════════════
#  Generic / Shared
# ══════════════════════════════════════════════

class ErrorResponse(BaseModel):
    """Standard error payload."""
    detail: str
    error_code: str
    context: Optional[dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """Base for paginated lists."""
    total: int
    limit: int
    offset: int


class JobEnqueuedResponse(BaseModel):
    """Returned when an async job is queued."""
    job_id: uuid.UUID
    status: str = "queued"
    message: str


class TrainEnqueuedResponse(JobEnqueuedResponse):
    run_id: uuid.UUID


# ══════════════════════════════════════════════
#  Project
# ══════════════════════════════════════════════

class ProjectConfig(BaseModel):
    """Nested config stored in project.config JSONB."""
    num_examples_target: int = Field(default=2000, ge=10, le=100000)
    max_seq_length: int = Field(default=2048, ge=128, le=8192)
    train_eval_split: float = Field(default=0.9, ge=0.5, le=0.99)


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    task_type: str = Field(..., pattern=r"^(instruction_tuning|qa_grounded|summarization|report_generation|information_extraction|classification|chat_dialogue_sft)$")
    config: Optional[ProjectConfig] = None

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Project name must not be blank")
        return v.strip()


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    task_type: Optional[str] = Field(default=None, pattern=r"^(instruction_tuning|qa_grounded|summarization|report_generation|information_extraction|classification|chat_dialogue_sft)$")
    base_model_id: Optional[str] = None
    model_info: Optional[dict[str, Any]] = None
    config: Optional[ProjectConfig] = None

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str | None) -> str | None:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Project name must not be blank")
        return v


class ProjectStats(BaseModel):
    file_count: int = 0
    chunk_count: int = 0
    example_count: int = 0
    last_job_status: Optional[str] = None


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: Optional[str]
    task_type: str
    base_model_id: Optional[str]
    model_info: Optional[dict[str, Any]]
    status: str
    config: Optional[dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class ProjectListItem(ProjectResponse):
    stats: Optional[ProjectStats] = None


class ProjectListResponse(PaginatedResponse):
    items: list[ProjectListItem]


# ══════════════════════════════════════════════
#  Model Resolution
# ══════════════════════════════════════════════

class ModelResolveRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=200)


class ModelResolveResponse(BaseModel):
    model_id: str
    model_type: Optional[str] = None
    num_parameters: Optional[int] = None
    estimated_vram_fp16_gb: Optional[float] = None
    estimated_vram_4bit_gb: Optional[float] = None
    architecture: Optional[str] = None
    vocab_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    license: Optional[str] = None
    valid: bool = True
    warnings: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════
#  Files
# ══════════════════════════════════════════════

class FileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    kind: str
    status: str
    mime_type: Optional[str]
    size_bytes: int
    created_at: datetime


class FileListResponse(BaseModel):
    files: list[FileResponse]


class FileUploadResponse(BaseModel):
    files: list[FileResponse]


# ══════════════════════════════════════════════
#  Chunks
# ══════════════════════════════════════════════

class ChunkGenerateRequest(BaseModel):
    chunk_size_tokens: int = Field(default=512, ge=64, le=4096)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=512)


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    source_file_id: uuid.UUID
    chunk_index: int
    content: str
    token_count: Optional[int]
    chunk_metadata: Optional[dict[str, Any]] = Field(
        default=None, alias="metadata_", serialization_alias="metadata"
    )


class ChunkListResponse(PaginatedResponse):
    items: list[ChunkResponse]


# ══════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════

class DatasetPreviewRequest(BaseModel):
    num_examples: int = Field(default=10, ge=1, le=50)
    ollama_model: Optional[str] = None


class DatasetGenerateRequest(BaseModel):
    num_examples_target: Optional[int] = Field(default=None, ge=10, le=100000)
    examples_per_chunk: int = Field(default=5, ge=1, le=20)
    ollama_model: Optional[str] = None


class DatasetExampleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    split: str
    data: dict[str, Any]
    is_valid: bool
    token_count: Optional[int]
    created_at: datetime


class DatasetStats(BaseModel):
    total: int = 0
    valid: int = 0
    invalid: int = 0
    train: int = 0
    eval: int = 0
    avg_token_count: Optional[float] = None


class DatasetSplitStats(BaseModel):
    count: int = 0
    avg_tokens: Optional[float] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None


class DatasetFullStats(BaseModel):
    total: int = 0
    valid: int = 0
    invalid: int = 0
    by_split: dict[str, DatasetSplitStats] = Field(default_factory=dict)
    validation_errors: dict[str, int] = Field(default_factory=dict)


class DatasetExampleListResponse(PaginatedResponse):
    items: list[DatasetExampleResponse]
    stats: Optional[DatasetStats] = None


# ══════════════════════════════════════════════
#  Hardware
# ══════════════════════════════════════════════

class HardwareCheckResponse(BaseModel):
    has_nvidia_smi: bool = False
    gpu_name: Optional[str] = None
    gpu_count: int = 0
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None
    driver_version: Optional[str] = None
    cuda_runtime: Optional[str] = None
    torch_version: Optional[str] = None
    torch_cuda: Optional[str] = None
    cuda_available: bool = False
    bnb_available: bool = False
    recommended_method: Optional[str] = None
    recommendation_reason: Optional[str] = None
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════

class TrainHyperparams(BaseModel):
    num_epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=2e-4, gt=0, le=1.0)
    per_device_batch_size: int = Field(default=4, ge=1, le=128)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=128)
    max_seq_length: int = Field(default=2048, ge=128, le=8192)
    lora_r: int = Field(default=16, ge=4, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0)


class TrainStartRequest(BaseModel):
    method: str = Field(..., pattern=r"^(lora|qlora)$")
    hyperparams: Optional[TrainHyperparams] = None


class TrainStatusResponse(BaseModel):
    run_id: Optional[uuid.UUID] = None
    job_id: Optional[uuid.UUID] = None
    status: str
    method: Optional[str] = None
    progress_pct: int = 0
    current_metrics: Optional[dict[str, Any]] = None
    started_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None


class TrainCancelResponse(BaseModel):
    status: str
    message: str


# ══════════════════════════════════════════════
#  Export
# ══════════════════════════════════════════════

class ExportFileInfo(BaseModel):
    filename: str
    size_bytes: int


class ExportFilesResponse(BaseModel):
    files: list[ExportFileInfo]


# ══════════════════════════════════════════════
#  Jobs
# ══════════════════════════════════════════════

class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    type: str
    status: str
    progress_pct: int
    result_summary: Optional[dict[str, Any]]
    error_message: Optional[str]
    error_code: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class JobListResponse(BaseModel):
    jobs: list[JobResponse]


class JobCancelResponse(BaseModel):
    status: str


# ══════════════════════════════════════════════
#  Playground / Inference
# ══════════════════════════════════════════════

class PlaygroundGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=500)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=3.0)
    do_sample: bool = True


class PlaygroundGenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    num_tokens_prompt: int
    num_tokens_generated: int


class PlaygroundStatusResponse(BaseModel):
    available: bool
    base_model_id: Optional[str] = None
    adapter_dir: Optional[str] = None
    run_id: Optional[str] = None
    method: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    message: str

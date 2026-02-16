"""
FineTuneFlow — Custom Exception Classes.

Hierarchy:
  FineTuneFlowError (base)
    ├── NotFoundError (404)
    │     ├── ProjectNotFoundError
    │     ├── FileRecordNotFoundError
    │     └── JobNotFoundError
    ├── ConflictError (409)
    │     └── TrainingAlreadyRunningError
    ├── InputValidationError (400)
    │     ├── FileUnsupportedTypeError
    │     └── DatasetValidationError
    ├── FileTooLargeError (413)
    ├── RateLimitError (429)
    ├── OllamaAPIError (502)
    └── GPUNotAvailableError (400)
"""

from typing import Any, Optional


class FineTuneFlowError(Exception):
    """Base exception for all FineTuneFlow errors."""

    error_code: str = "INTERNAL_ERROR"
    status_code: int = 500

    def __init__(
        self,
        detail: str,
        context: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        self.detail = detail
        self.context = context or {}
        if error_code is not None:
            self.error_code = error_code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(detail)


# ── 404 Not Found ────────────────────────────────────────


class NotFoundError(FineTuneFlowError):
    error_code = "NOT_FOUND"
    status_code = 404


class ProjectNotFoundError(NotFoundError):
    error_code = "PROJECT_NOT_FOUND"

    def __init__(self, project_id: str):
        super().__init__(
            detail=f"Project '{project_id}' not found",
            context={"project_id": project_id},
        )


class FileRecordNotFoundError(NotFoundError):
    error_code = "FILE_NOT_FOUND"

    def __init__(self, file_id: str):
        super().__init__(
            detail=f"File '{file_id}' not found",
            context={"file_id": file_id},
        )


class JobNotFoundError(NotFoundError):
    error_code = "JOB_NOT_FOUND"

    def __init__(self, job_id: str):
        super().__init__(
            detail=f"Job '{job_id}' not found",
            context={"job_id": job_id},
        )


# ── 409 Conflict ─────────────────────────────────────────


class ConflictError(FineTuneFlowError):
    error_code = "CONFLICT"
    status_code = 409


class TrainingAlreadyRunningError(ConflictError):
    error_code = "TRAINING_ALREADY_RUNNING"

    def __init__(self, project_id: str):
        super().__init__(
            detail=f"A training job is already running for project '{project_id}'",
            context={"project_id": project_id},
        )


class ProjectInvalidStatusError(ConflictError):
    error_code = "PROJECT_INVALID_STATUS"

    def __init__(self, project_id: str, current_status: str, expected: str):
        super().__init__(
            detail=f"Project '{project_id}' is in status '{current_status}', expected '{expected}'",
            context={
                "project_id": project_id,
                "current_status": current_status,
                "expected": expected,
            },
        )


# ── 400 Validation ───────────────────────────────────────


class InputValidationError(FineTuneFlowError):
    error_code = "VALIDATION_ERROR"
    status_code = 400


class FileUnsupportedTypeError(InputValidationError):
    error_code = "FILE_UNSUPPORTED_TYPE"

    def __init__(self, filename: str, mime_type: str):
        super().__init__(
            detail=f"File '{filename}' has unsupported MIME type '{mime_type}'",
            context={"filename": filename, "mime_type": mime_type},
        )


class FileInvalidNameError(InputValidationError):
    error_code = "FILE_INVALID_NAME"

    def __init__(self, filename: str):
        super().__init__(
            detail=f"File name '{filename}' is invalid (possible path traversal)",
            context={"filename": filename},
        )


class FileEmptyError(InputValidationError):
    error_code = "FILE_EMPTY"

    def __init__(self, filename: str):
        super().__init__(
            detail=f"File '{filename}' is empty (0 bytes)",
            context={"filename": filename},
        )


class DatasetValidationError(InputValidationError):
    error_code = "DATASET_VALIDATION_ERROR"


class DatasetTooFewExamplesError(InputValidationError):
    error_code = "DATASET_TOO_FEW_EXAMPLES"

    def __init__(self, count: int, minimum: int = 10):
        super().__init__(
            detail=f"Only {count} valid examples, minimum {minimum} required for training",
            context={"count": count, "minimum": minimum},
        )


class NoChunksError(InputValidationError):
    error_code = "DATASET_NO_CHUNKS"

    def __init__(self, project_id: str):
        super().__init__(
            detail=f"No chunks available for project '{project_id}'. Upload and chunk documents first.",
            context={"project_id": project_id},
        )


# ── 413 Payload Too Large ────────────────────────────────


class FileTooLargeError(FineTuneFlowError):
    error_code = "FILE_TOO_LARGE"
    status_code = 413

    def __init__(self, filename: str, size_mb: float, max_mb: int):
        super().__init__(
            detail=f"File '{filename}' exceeds maximum size of {max_mb} MB (actual: {size_mb:.1f} MB)",
            context={"filename": filename, "size_mb": size_mb, "max_mb": max_mb},
        )


class ProjectTotalSizeExceededError(FineTuneFlowError):
    error_code = "FILE_TOTAL_LIMIT"
    status_code = 413

    def __init__(self, project_id: str, max_mb: int):
        super().__init__(
            detail=f"Total file size for project exceeds limit of {max_mb} MB",
            context={"project_id": project_id, "max_mb": max_mb},
        )


# ── 502 External API ─────────────────────────────────────


class OllamaAPIError(FineTuneFlowError):
    error_code = "OLLAMA_API_ERROR"
    status_code = 502


class OllamaAPIKeyMissingError(FineTuneFlowError):
    error_code = "OLLAMA_API_KEY_MISSING"
    status_code = 400

    def __init__(self):
        super().__init__(
            detail="Ollama Cloud API key is not configured. Set OLLAMA_CLOUD_API_KEY in .env"
        )


class OllamaAPIKeyInvalidError(FineTuneFlowError):
    error_code = "OLLAMA_API_KEY_INVALID"
    status_code = 401

    def __init__(self):
        super().__init__(detail="Ollama Cloud API key is invalid or rejected")


class ModelNotFoundError(FineTuneFlowError):
    error_code = "MODEL_NOT_FOUND"
    status_code = 404

    def __init__(self, model_id: str):
        super().__init__(
            detail=f"Model '{model_id}' not found on HuggingFace Hub",
            context={"model_id": model_id},
        )


class ModelAccessDeniedError(FineTuneFlowError):
    error_code = "MODEL_ACCESS_DENIED"
    status_code = 403

    def __init__(self, model_id: str):
        super().__init__(
            detail=f"Access denied for model '{model_id}'. It may be gated — configure HF_TOKEN in .env",
            context={"model_id": model_id},
        )


# ── Training ─────────────────────────────────────────────


class GPUNotAvailableError(FineTuneFlowError):
    error_code = "GPU_NOT_AVAILABLE"
    status_code = 400

    def __init__(self):
        super().__init__(
            detail="No CUDA GPU detected. GPU fine-tuning not possible on this machine."
        )


class TrainingOOMError(FineTuneFlowError):
    error_code = "TRAINING_OOM"
    status_code = 500

    def __init__(self, detail: str = "CUDA out of memory during training"):
        super().__init__(
            detail=detail,
            context={"suggestion": "Try QLoRA, reduce batch_size to 1, or reduce max_seq_length"},
        )


class TrainingDivergedError(FineTuneFlowError):
    error_code = "TRAINING_DIVERGED"
    status_code = 500

    def __init__(self):
        super().__init__(detail="Training loss diverged (NaN or Inf detected)")

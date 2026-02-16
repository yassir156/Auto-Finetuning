"""
FineTuneFlow — Application Settings.

All config is loaded from environment variables (via .env file).
Uses pydantic-settings for validation and type coercion.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    APP_NAME: str = "FineTuneFlow"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Database ─────────────────────────────────────────
    DATABASE_URL: str = "postgresql+psycopg2://finetune:finetune@postgres:5432/finetuneflow"

    # ── Redis ────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"

    # ── Celery ───────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/1"

    # ── Storage ──────────────────────────────────────────
    STORAGE_ROOT: str = "/app/storage"

    @property
    def storage_path(self) -> Path:
        return Path(self.STORAGE_ROOT)

    # ── Ollama Cloud API ─────────────────────────────────
    OLLAMA_CLOUD_BASE_URL: str = "https://api.ollama.com"
    OLLAMA_CLOUD_API_KEY: str = "CHANGE_ME"
    OLLAMA_MODEL: str = "llama3.1:70b"
    OLLAMA_TEMPERATURE: float = 0.7
    OLLAMA_MAX_TOKENS: int = 4096
    OLLAMA_TIMEOUT_SECONDS: int = 60

    # ── Upload Limits ────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    MAX_PROJECT_TOTAL_SIZE_MB: int = 500
    MAX_FILES_PER_UPLOAD: int = 20
    MAX_FILES_PER_PROJECT: int = 100

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def max_project_total_size_bytes(self) -> int:
        return self.MAX_PROJECT_TOTAL_SIZE_MB * 1024 * 1024

    # ── Training Defaults ────────────────────────────────
    DEFAULT_NUM_EPOCHS: int = 3
    DEFAULT_LEARNING_RATE: float = 2e-4
    DEFAULT_BATCH_SIZE: int = 4
    DEFAULT_MAX_SEQ_LENGTH: int = 2048
    DEFAULT_LORA_R: int = 16
    DEFAULT_LORA_ALPHA: int = 32

    # ── Dataset Generation ───────────────────────────────
    DEFAULT_NUM_EXAMPLES_TARGET: int = 2000
    DEFAULT_EXAMPLES_PER_CHUNK: int = 5
    DEFAULT_CHUNK_SIZE_TOKENS: int = 512
    DEFAULT_CHUNK_OVERLAP_TOKENS: int = 50
    DEFAULT_TRAIN_EVAL_SPLIT: float = 0.9
    DEFAULT_PREVIEW_EXAMPLES: int = 10

    # ── HuggingFace (optional, for gated models) ────────
    HF_TOKEN: Optional[str] = None

    # ── CORS ─────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    CORS_EXTRA_ORIGINS: str = ""  # comma-separated extra origins

    @property
    def all_cors_origins(self) -> list[str]:
        origins = list(self.CORS_ORIGINS)
        if self.CORS_EXTRA_ORIGINS:
            origins.extend(o.strip() for o in self.CORS_EXTRA_ORIGINS.split(",") if o.strip())
        return origins


# Singleton instance
settings = Settings()

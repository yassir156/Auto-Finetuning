"""
FineTuneFlow — FastAPI Application Entry Point.

Sets up:
  - CORS middleware
  - Exception handlers (maps custom exceptions → JSON error responses)
  - Rate limiter (slowapi)
  - Router includes (health, projects, files, chunks, dataset, hardware, train, export, jobs)
  - Startup/shutdown events
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.deps import limiter
from app.api.routes import health
from app.core.config import settings
from app.core.exceptions import FineTuneFlowError
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown."""
    setup_logging()
    logger.info("app.startup", version="0.1.0")
    yield
    logger.info("app.shutdown")


# ── App ───────────────────────────────────────
app = FastAPI(
    title="FineTuneFlow",
    description="Local fine-tuning pipeline: docs → dataset → LoRA/QLoRA SFT → export",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.all_cors_origins if not settings.DEBUG else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiter ──────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Exception Handlers ────────────────────────
@app.exception_handler(FineTuneFlowError)
async def finetuneflow_error_handler(request: Request, exc: FineTuneFlowError):
    """Convert custom exceptions to structured JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": exc.error_code,
            "context": exc.context,
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Catch-all for unexpected errors — log and return 500."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "context": None,
        },
    )


# ── Routes ────────────────────────────────────
app.include_router(health.router)

from app.api.routes import projects, files, chunks, dataset, jobs
from app.api.routes.model_hardware import model_router, hardware_router
from app.api.routes.training import train_router, export_router
from app.api.routes.playground import playground_router

app.include_router(projects.router)
app.include_router(files.router)
app.include_router(chunks.router)
app.include_router(dataset.router)
app.include_router(model_router)
app.include_router(hardware_router)
app.include_router(train_router)
app.include_router(export_router)
app.include_router(playground_router)
app.include_router(jobs.router)
# app.include_router(jobs.router, tags=["jobs"])

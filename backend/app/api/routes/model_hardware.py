"""
FineTuneFlow — Model Resolution + Hardware Route.

Endpoints:
  POST /projects/{project_id}/model/resolve
  GET  /hardware/check
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.api.deps import get_project_or_404, limiter
from app.api.schemas import (
    HardwareCheckResponse,
    ModelResolveRequest,
    ModelResolveResponse,
)
from app.core.exceptions import ModelAccessDeniedError, ModelNotFoundError
from app.core.logging import get_logger
from app.db.models import Project
from app.db.session import get_db

logger = get_logger(__name__)

model_router = APIRouter(tags=["model"])
hardware_router = APIRouter(tags=["hardware"])


# ── Model Resolution ──────────────────────────

@model_router.post("/projects/{project_id}/model/resolve")
@limiter.limit("30/minute")
def resolve_model(
    request: Request,
    project_id: uuid.UUID,
    body: ModelResolveRequest,
    db: Session = Depends(get_db),
    project: Project = Depends(get_project_or_404),
):
    """Resolve a HuggingFace model and estimate VRAM requirements."""
    model_id = body.model_id.strip()
    logger.info("model.resolve", project_id=str(project_id), model_id=model_id)

    try:
        from huggingface_hub import model_info as hf_model_info
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
        )
    except ImportError:
        # huggingface_hub not installed — return minimal info
        return ModelResolveResponse(
            model_id=model_id,
            valid=False,
            warnings=["huggingface_hub not installed"],
        )

    try:
        from app.core.config import settings

        info = hf_model_info(model_id, token=settings.HF_TOKEN)
    except RepositoryNotFoundError:
        raise ModelNotFoundError(model_id=model_id)
    except GatedRepoError:
        raise ModelAccessDeniedError(model_id=model_id)
    except Exception as exc:
        logger.error("model.resolve.error", model_id=model_id, error=str(exc))
        return ModelResolveResponse(
            model_id=model_id,
            valid=False,
            warnings=[f"Failed to resolve model: {exc}"],
        )

    # Extract info
    config = getattr(info, "config", None) or {}
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    elif not isinstance(config, dict):
        config = {}

    safetensors = getattr(info, "safetensors", None)
    num_params = None
    if safetensors and hasattr(safetensors, "total"):
        num_params = safetensors.total
    elif "num_parameters" in config:
        num_params = config.get("num_parameters")

    architecture = None
    architectures = config.get("architectures", [])
    if architectures:
        architecture = architectures[0]

    model_type = config.get("model_type")
    vocab_size = config.get("vocab_size")
    max_pos = config.get("max_position_embeddings")
    license_info = getattr(info, "card_data", None)
    license_str = None
    if license_info and hasattr(license_info, "license"):
        license_str = license_info.license

    # VRAM estimation
    vram_fp16 = None
    vram_4bit = None
    warnings = []
    if num_params:
        vram_fp16 = round(num_params * 2 / (1024**3), 2)  # 2 bytes per param
        vram_4bit = round(num_params * 0.5 / (1024**3) + 1.5, 2)  # ~0.5 bytes + overhead
    else:
        warnings.append("Could not determine parameter count — VRAM estimation unavailable")

    # Update project
    result = ModelResolveResponse(
        model_id=model_id,
        model_type=model_type,
        num_parameters=num_params,
        estimated_vram_fp16_gb=vram_fp16,
        estimated_vram_4bit_gb=vram_4bit,
        architecture=architecture,
        vocab_size=vocab_size,
        max_position_embeddings=max_pos,
        license=license_str,
        valid=True,
        warnings=warnings,
    )

    # Persist model info on project
    project.base_model_id = model_id
    project.model_info = result.model_dump()
    db.commit()

    return result


# ── Hardware Check ────────────────────────────

@hardware_router.get("/hardware/check")
@limiter.limit("30/minute")
def check_hardware(request: Request):
    """Probe local GPU/CUDA availability."""
    result = HardwareCheckResponse()

    # Check nvidia-smi
    import subprocess

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            result.has_nvidia_smi = True
            lines = smi.stdout.strip().split("\n")
            result.gpu_count = len(lines)
            parts = lines[0].split(", ")
            if len(parts) >= 4:
                result.gpu_name = parts[0].strip()
                result.vram_total_gb = round(float(parts[1].strip()) / 1024, 2)
                result.vram_free_gb = round(float(parts[2].strip()) / 1024, 2)
                result.driver_version = parts[3].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check PyTorch
    try:
        import torch

        result.torch_version = torch.__version__
        result.cuda_available = torch.cuda.is_available()
        if result.cuda_available:
            result.torch_cuda = torch.version.cuda
    except ImportError:
        result.notes.append("PyTorch not installed")

    # Check bitsandbytes
    try:
        import bitsandbytes  # noqa: F401

        result.bnb_available = True
    except ImportError:
        result.bnb_available = False

    # Recommendation
    if result.cuda_available and result.vram_total_gb:
        if result.vram_total_gb >= 24:
            result.recommended_method = "qlora"
            result.recommendation_reason = (
                f"{result.vram_total_gb}GB VRAM detected. "
                "QLoRA recommended for 8B+ models to save memory."
            )
        elif result.vram_total_gb >= 16:
            result.recommended_method = "qlora"
            result.recommendation_reason = (
                f"{result.vram_total_gb}GB VRAM. QLoRA strongly recommended."
            )
        elif result.vram_total_gb >= 8:
            result.recommended_method = "qlora"
            result.recommendation_reason = (
                f"Only {result.vram_total_gb}GB VRAM. QLoRA with small models only."
            )
            result.warnings.append("Low VRAM — use small models (<3B params)")
        else:
            result.recommended_method = None
            result.recommendation_reason = (
                f"Only {result.vram_total_gb}GB VRAM — too low for fine-tuning."
            )
            result.warnings.append("VRAM too low for fine-tuning")
    elif not result.cuda_available:
        result.recommendation_reason = (
            "No CUDA GPU detected. GPU fine-tuning not possible on this machine."
        )
        result.warnings.append("No NVIDIA GPU found. Consider using a cloud GPU instance.")

    if not result.warnings and result.cuda_available:
        result.notes.append("All checks passed")

    return result

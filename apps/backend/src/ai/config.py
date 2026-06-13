"""AI-specific configuration dataclass.

Holds model paths, GPU settings, inference parameters, and auto-selects
the ONNX Runtime provider based on available hardware.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


def _resolve_provider(use_gpu: bool, gpu_device_id: int) -> str:
    """Auto-select CUDAExecutionProvider or CPUExecutionProvider.

    Args:
        use_gpu: Whether GPU inference is requested.
        gpu_device_id: Specific CUDA device to target (ignored for CPU).

    Returns:
        ``"CUDAExecutionProvider"`` if GPU is requested and CUDA is
        available, otherwise ``"CPUExecutionProvider"``.
    """
    if use_gpu and torch.cuda.is_available():
        logger.info(
            "Selecting CUDAExecutionProvider (device=%s)",
            torch.cuda.get_device_name(gpu_device_id),
        )
        return "CUDAExecutionProvider"
    logger.info("Selecting CPUExecutionProvider")
    return "CPUExecutionProvider"


@dataclass(frozen=True)
class AiConfig:
    """Frozen configuration for AI model inference.

    All fields have environment-variable overrides for production
    deployment (per D-05 on-premise constraint).
    """

    # --- Paths ---
    model_dir: str = os.getenv("AI_MODEL_DIR", "data/models/")

    # --- GPU ---
    use_gpu: bool = os.getenv("AI_USE_GPU", "true").lower() == "true"
    gpu_device_id: int = int(os.getenv("AI_GPU_DEVICE_ID", "0"))

    # --- Model names (basename without ``.onnx`` suffix) ---
    segmentation_model: str = os.getenv("AI_SEG_MODEL", "segment")
    enhancement_model: str = os.getenv("AI_ENH_MODEL", "enhance")
    extraction_model: str = os.getenv("AI_EXT_MODEL", "extract")

    # --- Inference parameters ---
    input_size: int = int(os.getenv("AI_INPUT_SIZE", "512"))
    confidence_threshold: float = float(
        os.getenv("AI_CONFIDENCE_THRESH", "0.5")
    )

    # --- Computed post-init ---
    provider: str = "CPUExecutionProvider"

    def __post_init__(self) -> None:
        """Resolve the ONNX Runtime provider after field initialisation."""
        # Frozen dataclass: use object.__setattr__ to bypass immutability.
        object.__setattr__(
            self,
            "provider",
            _resolve_provider(self.use_gpu, self.gpu_device_id),
        )

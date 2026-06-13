"""GPU detection and utility helpers for AI/ML inference.

Detects available GPU hardware via PyTorch and exposes flags and
device-info functions that the rest of the application can consult.
"""

from __future__ import annotations

import logging as _logging
import os as _os

_logger = _logging.getLogger(__name__)


def _detect_gpu() -> bool:
    """Attempt to detect a CUDA-capable GPU via PyTorch.

    Returns:
        ``True`` if PyTorch reports a CUDA device is available.
    """
    try:
        import torch  # noqa: F811 — re-imported below in get_device_info

        available = torch.cuda.is_available()
        if available:
            _logger.info("GPU detected via PyTorch: %s", torch.cuda.get_device_name(0))
        else:
            _logger.info("No CUDA-capable GPU detected via PyTorch")
        return available
    except ImportError:
        _logger.warning("PyTorch not installed — GPU detection deferred")
        return False


GPU_AVAILABLE: bool = _detect_gpu()
CUPY_AVAILABLE: bool = False  # Reserved for future CuPy integration


def is_gpu_enabled() -> bool:
    """Check whether GPU acceleration is available and not force-disabled.

    Respects the ``FORCE_CPU`` environment variable so that tests and
    CPU-only deployments can reliably disable GPU usage.
    """
    return GPU_AVAILABLE and _os.getenv("FORCE_CPU", "0") != "1"


def get_device_info() -> dict[str, str]:
    """Return a dictionary describing the active compute device.

    Returns:
        A dict with keys ``backend``, ``device``, and ``available``.
    """
    if GPU_AVAILABLE:
        import torch

        return {
            "backend": "GPU",
            "device": torch.cuda.get_device_name(0),
            "available": "true",
        }
    return {"backend": "CPU", "device": "NumPy", "available": "false"}


FORCE_CPU: bool = _os.getenv("FORCE_CPU", "0") == "1"

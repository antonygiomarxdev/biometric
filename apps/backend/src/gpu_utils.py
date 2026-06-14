"""GPU detection and configuration utilities.

Provides a centralised interface for checking GPU availability,
force-enabling CPU mode, and retrieving device information.

The module-level ``GPU_AVAILABLE`` constant is set once at import time
by probing ``torch.cuda.is_available()``.  Callers can also use the
``is_gpu_enabled()`` function which respects the ``FORCE_CPU``
environment variable.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def _detect_gpu() -> bool:
    """Probe whether a CUDA-capable GPU is present.

    Returns:
        ``True`` if CUDA is available, ``False`` otherwise.
        Exceptions during detection (missing driver, etc.) also
        return ``False``.
    """
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


GPU_AVAILABLE: bool = _detect_gpu()


def is_gpu_enabled() -> bool:
    """Check whether GPU is enabled, respecting ``FORCE_CPU``.

    Returns ``False`` immediately when ``FORCE_CPU=1`` is set, even
    if a CUDA-capable GPU is present.  Otherwise defers to the
    module-level ``GPU_AVAILABLE`` constant.

    Returns:
        ``True`` if a GPU is available and not force-disabled.
    """
    if os.getenv("FORCE_CPU", "0") == "1":
        return False
    return GPU_AVAILABLE


def get_device_info() -> dict[str, str | bool]:
    """Return a summary dict of the current device configuration.

    Returns:
        Dictionary with keys ``backend`` (``"cuda"`` / ``"cpu"``),
        ``device`` (device name or ``"cpu"``), and ``available``
        (boolean).
    """
    if GPU_AVAILABLE:
        return {
            "backend": "cuda",
            "device": torch.cuda.get_device_name(0),
            "available": True,
        }
    return {
        "backend": "cpu",
        "device": "cpu",
        "available": False,
    }

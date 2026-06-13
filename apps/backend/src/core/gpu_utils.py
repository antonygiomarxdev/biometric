"""CPU-only utilities. GPU/CUDA support removed — always falls back to CPU in practice."""

import os

GPU_AVAILABLE = False
CUPY_AVAILABLE = False


def is_gpu_enabled() -> bool:
    return False


def get_device_info() -> dict[str, str]:
    return {"backend": "CPU", "device": "NumPy", "available": "false"}


# Force CPU via env var for explicitness
FORCE_CPU = os.getenv("FORCE_CPU", "1") == "1"

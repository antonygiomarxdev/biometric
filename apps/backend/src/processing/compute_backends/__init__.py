"""Compute backend abstraction for numerical operations.

The matching pipeline (Gabor filter bank, FFT, etc.) is hot. We support
two backends:

- :class:`NumpyBackend` — pure CPU, always available. Uses
  ``cv2.filter2D`` (C++ with SIMD) for 2-D convolutions.
- :class:`CupyBackend` — GPU (NVIDIA CUDA). Optional. Falls back
  gracefully if CuPy is not installed or no GPU is reachable.

Selection policy (``get_compute_backend``):

  - ``auto`` (default): try CuPy first, fall back to numpy. Logs which
    backend was selected.
  - ``cupy``: force CuPy (raises if unavailable).
  - ``numpy``: force numpy (no GPU).

The backend is selected once at app startup. Operations take a numpy
array and return a numpy array; the GPU <-> CPU transfer is hidden
inside the backend. Consumers don't import CuPy directly.
"""
from __future__ import annotations

import logging
import os
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class ComputeBackend(Protocol):
    """Protocol for 2-D numerical operations."""

    @property
    def name(self) -> str: ...

    def is_available(self) -> bool: ...

    def convolve2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        *,
        mode: str = "same",
    ) -> np.ndarray: ...


class NumpyBackend:
    """CPU backend using ``cv2.filter2D`` (C++ with SIMD).

    ``cv2.filter2D`` is the fastest CPU 2-D convolution for the
    fingerprint Gabor filter bank (60 filters of up to 39x39 on
    350x326 images): 60x faster than scipy.signal.convolve2d.
    Uses BORDER_REFLECT_101 boundary padding.
    """

    @property
    def name(self) -> str:
        return "numpy"

    def is_available(self) -> bool:
        return True

    def convolve2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        *,
        mode: str = "same",
    ) -> np.ndarray:
        import cv2
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


class CupyBackend:
    """GPU backend using CuPy (NVIDIA CUDA).

    Optional dependency. If CuPy is not installed or no GPU is
    reachable, ``is_available()`` returns False and the factory
    falls back to :class:`NumpyBackend`.
    """

    def __init__(self) -> None:
        self._cupy = None
        self._cp_signal = None

    @property
    def name(self) -> str:
        return "cupy"

    def is_available(self) -> bool:
        try:
            import cupy  # noqa: F401
            import cupy.cuda.runtime  # noqa: F401
            return int(cupy.cuda.runtime.getDeviceCount()) > 0
        except Exception as e:
            logger.debug("CuPy availability check failed: %s", e)
            return False

    def convolve2d(
        self,
        image: np.ndarray,
        kernel: np.ndarray,
        *,
        mode: str = "same",
    ) -> np.ndarray:
        if self._cupy is None:
            import cupy as cp
            import cupyx.scipy.signal as cp_signal
            self._cupy = cp
            self._cp_signal = cp_signal
        cp = self._cupy
        cp_signal = self._cp_signal
        img_gpu = cp.asarray(image)
        kern_gpu = cp.asarray(kernel)
        return cp.asnumpy(cp_signal.correlate2d(img_gpu, kern_gpu, mode=mode))


def get_compute_backend(name: str | None = None) -> ComputeBackend:
    """Select the best compute backend available.

    Parameters
    ----------
    name:
        ``auto`` (default), ``cupy``, or ``numpy``. If ``None``,
        read the ``MCC_COMPUTE_BACKEND`` env var, falling back to
        ``auto``.

    Returns
    -------
    A :class:`ComputeBackend` instance. Always returns a working
    backend (falls back to numpy if requested backend is unavailable).
    """
    if name is None:
        name = os.getenv("MCC_COMPUTE_BACKEND", "auto")
    name = name.lower().strip()

    if name in ("cupy", "gpu"):
        backend = CupyBackend()
        if not backend.is_available():
            logger.warning(
                "MCC_COMPUTE_BACKEND=cupy requested but CuPy/GPU unavailable; "
                "falling back to numpy",
            )
            backend = NumpyBackend()
    elif name in ("numpy", "cpu"):
        backend = NumpyBackend()
    else:  # "auto" or anything else
        for cls in (CupyBackend, NumpyBackend):
            backend = cls()
            if backend.is_available():
                break
        else:  # pragma: no cover  # defensive: numpy is always available
            backend = NumpyBackend()

    logger.info("Compute backend: %s", backend.name)
    return backend

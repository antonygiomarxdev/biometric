"""
GPU-accelerated fingerprint enhancement using CuPy.

Auto-detects any NVIDIA GPU via CuPy/CUDA runtime. Falls back to
the CPU pipeline transparently when no GPU is available. No manual
GPU type setup required.
"""

from __future__ import annotations

import logging

import numpy as np

from src.core.config import Config
from src.core.interfaces import IPipelineStep, PipelineContext

logger = logging.getLogger(__name__)


def _gpu_available() -> tuple[bool, str]:
    """Check if CuPy + CUDA is available and return (ok, description)."""
    try:
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        if count == 0:
            return False, "no CUDA devices found"
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        return True, f"cuPy + {name}"
    except Exception as e:
        return False, f"cuPy not available ({e})"


class GpuEnhancer(IPipelineStep):
    """GPU-accelerated Gabor enhancement for fingerprints.

    Ports the heavy matrix math (FFT, Gabor, gradient computations) from
    NumPy/CV2 to CuPy, running on any CUDA-capable GPU. Auto-detects
    the available hardware at init time and logs the device info.

    Falls back to CPU enhancement transparently if:
    - No CUDA-capable GPU is detected.
    - CuPy is not installed.
    - GPU memory is exhausted.

    Usage::

        enhancer = GpuEnhancer()
        ctx = PipelineContext(raw_image=img)
        enhancer.process(ctx)
        # ctx.enhanced_image is now populated (np.ndarray on CPU)
    """

    def __init__(self, resize: bool = True) -> None:
        self.resize = resize
        self._use_gpu, self._device_info = _gpu_available()
        if self._use_gpu:
            logger.info("GpuEnhancer: using %s", self._device_info)
            import cupy as cp
            self.cp = cp
        else:
            logger.info("GpuEnhancer: falling back to CPU (%s)", self._device_info)
            self.cp = None

    def process(self, ctx: PipelineContext) -> None:
        source = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image

        if not self._use_gpu or self.cp is None:
            from src.processing.enhancer import create_enhancer
            fallback = create_enhancer()
            ctx.enhanced_image = fallback.enhance(source, resize=self.resize)
            ctx.preprocessed_image = ctx.enhanced_image
            return

        # GPU path
        try:
            result = self._enhance_gpu(source)
            if result.ndim != 2:
                result = result.squeeze()
            ctx.enhanced_image = result.astype(np.uint8)
            ctx.preprocessed_image = ctx.enhanced_image
        except Exception as e:
            logger.warning("GPU enhancement failed (%s), falling back to CPU", e)
            from src.processing.enhancer import create_enhancer
            fallback = create_enhancer()
            ctx.enhanced_image = fallback.enhance(source, resize=self.resize)
            ctx.preprocessed_image = ctx.enhanced_image

    def _enhance_gpu(self, img: np.ndarray) -> np.ndarray:
        """GPU implementation using CuPy (spatial domain, no cuFFT needed)."""
        cp = self.cp

        f = cp.asarray(img, dtype=cp.float32)

        if self.resize:
            from src.core.config import Config
            config = Config()
            rows, cols = img.shape
            new_rows = config.image_resize_width
            new_cols = int(new_rows / (rows / cols))
            from cupyx.scipy.ndimage import zoom
            scale_y = new_rows / rows
            scale_x = new_cols / cols
            f = zoom(f, (scale_y, scale_x), order=1)

        # Normalization
        mn = f.mean()
        sd = f.std()
        if sd > 0:
            f = (f - mn) / sd
            f = cp.clip(f, -3.0, 3.0)
            mn2, mx2 = f.min(), f.max()
            if mx2 > mn2:
                f = ((f - mn2) / (mx2 - mn2) * 255).astype(cp.float32)

        h, w = f.shape

        # Orientation via Sobel
        gx = cp.zeros_like(f)
        gy = cp.zeros_like(f)
        gx[1:-1, 1:-1] = (f[1:-1, 2:] - f[1:-1, :-2]) * 0.5
        gy[1:-1, 1:-1] = (f[2:, 1:-1] - f[:-2, 1:-1]) * 0.5

        rows_b = h // 16
        cols_b = w // 16
        orientim = cp.zeros((rows_b, cols_b), dtype=cp.float32)

        for by in range(rows_b):
            y0 = by * 16
            y1 = y0 + 16
            for bx in range(cols_b):
                x0 = bx * 16
                x1 = x0 + 16
                blk_gx = gx[y0:y1, x0:x1]
                blk_gy = gy[y0:y1, x0:x1]
                gxx = (blk_gx * blk_gx).mean()
                gyy = (blk_gy * blk_gy).mean()
                gxy = (blk_gx * blk_gy).mean()
                denom = gxx + gyy
                if denom > 1e-6:
                    coh = float(cp.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / denom)
                    if coh > 0.3:
                        orientim[by, bx] = 0.5 * float(cp.arctan2(2 * gxy, gxx - gyy))

        # Spatial Gabor filter (no FFT needed)
        mean_angle = float(orientim.mean()) if orientim.max() != 0 else 0.0
        freq = 0.1
        ks = 25
        half = ks // 2
        y_k, x_k = cp.mgrid[-half:half + 1, -half:half + 1]
        xr = x_k * cp.cos(mean_angle) + y_k * cp.sin(mean_angle)
        yr = -x_k * cp.sin(mean_angle) + y_k * cp.cos(mean_angle)
        gw = cp.exp(-(xr ** 2 + 4 * yr ** 2) / (2 * (ks / 4) ** 2))
        gabor_kernel = gw * cp.cos(2 * cp.pi * freq * xr)

        from cupyx.scipy.ndimage import convolve
        filtered = convolve(f, gabor_kernel, mode='reflect')

        result = cp.abs(filtered)
        rmin, rmax = float(result.min()), float(result.max())
        if rmax > rmin:
            result = (result - rmin) / (rmax - rmin) * 255
        result = cp.clip(result, 0, 255)

        return cp.asnumpy(result).astype(np.uint8)

"""
Pre-processors (IPreProcessor) for the fingerprint pipeline.

Runs before core extraction. Each hook transforms the image or generates
a quality mask that downstream stages (post-processors) can use to filter
false positives.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.core.interfaces import IPreProcessor, PreProcessResult

logger = logging.getLogger(__name__)


class QualityMasker(IPreProcessor):
    """Computes a quality mask by analysing block-level variance.

    Divides the image into ``block_size × block_size`` tiles. Tiles with
    variance below ``min_variance`` are considered background / scar / blur
    and masked out.  Optionally detects directional coherence to flag
    altered regions (cuts, obliterations) where ridge flow is chaotic.
    """

    def __init__(
        self,
        block_size: int = 16,
        min_variance: float = 15.0,
        coherence_threshold: float = 0.3,
    ) -> None:
        self.block_size = block_size
        self.min_variance = min_variance
        self.coherence_threshold = coherence_threshold

    def process(self, image: np.ndarray) -> PreProcessResult:
        h, w = image.shape
        mask = np.ones((h, w), dtype=bool)

        # --- Variance-based masking (fondo / borrones) ---
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = image[y: y + self.block_size, x: x + self.block_size]
                if block.size == 0:
                    continue
                var = block.var()
                if var < self.min_variance:
                    mask[y: y + self.block_size, x: x + self.block_size] = False

        # --- Coherence-based masking (cicatrices / alteraciones) ---
        gray_f32 = image.astype(np.float32)
        grad_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)

        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                gx_block = grad_x[y: y + self.block_size, x: x + self.block_size]
                gy_block = grad_y[y: y + self.block_size, x: x + self.block_size]
                if gx_block.size == 0:
                    continue

                gxx = (gx_block ** 2).mean()
                gyy = (gy_block ** 2).mean()
                gxy = (gx_block * gy_block).mean()

                denom = gxx + gyy
                if denom < 1e-6:
                    continue

                coherence = np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2) / denom
                if coherence < self.coherence_threshold:
                    mask[y: y + self.block_size, x: x + self.block_size] = False

        logger.debug(
            "QualityMasker: masked %.1f%% of pixels (var<thr=%.1f coh<thr=%.2f)",
            100.0 * (1.0 - mask.mean()),
            self.min_variance,
            self.coherence_threshold,
        )
        return PreProcessResult(image=image, quality_mask=mask)


class BinarizationHook(IPreProcessor):
    """Applies adaptive thresholding to produce a clean binary image.

    Uses Otsu's method as the primary thresholder, falling back to
    a fixed threshold when Otsu fails.
    """

    def __init__(self, invert: bool = True) -> None:
        self.invert = invert

    def process(self, image: np.ndarray) -> PreProcessResult:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.invert:
            binary = cv2.bitwise_not(binary)

        logger.debug(
            "BinarizationHook: output shape=%s, white pixels=%.1f%%",
            binary.shape,
            100.0 * (binary > 0).mean(),
        )
        return PreProcessResult(image=binary, quality_mask=None)

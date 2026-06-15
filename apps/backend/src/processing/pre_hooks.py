"""
Pre-processors (IPreProcessor) for the fingerprint pipeline.

Each hook reads from / writes to a :class:`PipelineContext`. The
orchestrator passes the context to every hook in declared order; no
hooks know about each other directly.

Available pre-processors
-------------------------
* :class:`OrientationFieldAnalyzer` — Hong-Wan-Jain 1998 orientation field
  + coherence-based quality mask. Industry-standard AFIS preprocessing.
* :class:`SingularityDetector` — finds Core (and Delta) singularities via
  the Poincaré index and produces a Region-of-Interest (ROI) mask centred
  on the strongest Core.
* :class:`BinarizationHook` — adaptive Otsu binarisation.
* :class:`QualityMasker` — legacy variance + Sobel coherence quality mask.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.core.interfaces import IPipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class BinarizationHook(IPipelineStep):
    """Applies Otsu binarisation to the preprocessed image.

    Reads ``ctx.preprocessed_image`` (or ``ctx.raw_image`` if not yet set),
    writes the binary result back to ``ctx.preprocessed_image``.
    """

    def __init__(self, invert: bool = True) -> None:
        self.invert = invert

    def process(self, ctx: PipelineContext) -> None:
        image = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if self.invert:
            binary = cv2.bitwise_not(binary)

        ctx.preprocessed_image = binary
        logger.debug(
            "BinarizationHook: output shape=%s, white pixels=%.1f%%",
            binary.shape,
            100.0 * (binary > 0).mean(),
        )


class OrientationFieldAnalyzer(IPipelineStep):
    """Computes a ridge orientation field + quality mask using Hong-Wan-Jain 1998.

    Reads the preprocessed image (or raw if not set), writes
    ``ctx.orientation_field``, ``ctx.coherence_field``, and
    ``ctx.quality_mask``.

    The ``quality_mask`` uses the **orientation certainty level**:

        OCL = sqrt((Gxx - Gyy)^2 + (2*Gxy)^2) / (Gxx + Gyy)

    A value close to 1 = highly coherent (parallel ridges);
    a value close to 0 = chaotic (noise / scar / cut).
    """

    def __init__(
        self,
        block_size: int = 16,
        min_energy: float = 1e-3,
        coherence_threshold: float = 0.35,
        gaussian_sigma: float = 0.0,
    ) -> None:
        self.block_size = block_size
        self.min_energy = min_energy
        self.coherence_threshold = coherence_threshold
        self.gaussian_sigma = gaussian_sigma
        self.orientation_field: np.ndarray | None = None
        self.coherence_field: np.ndarray | None = None

    def process(self, ctx: PipelineContext) -> None:
        image = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ctx.preprocessed_image = image

        if self.gaussian_sigma > 0:
            blurred = cv2.GaussianBlur(image, (0, 0), self.gaussian_sigma)
        else:
            blurred = image

        f = blurred.astype(np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)

        h, w = f.shape
        bh, bw = self.block_size, self.block_size
        rows = h // bh
        cols = w // bw

        quality_mask = np.ones((h, w), dtype=bool)
        ori_field = np.zeros((rows, cols), dtype=np.float32)
        coh_field = np.zeros((rows, cols), dtype=np.float32)

        for by in range(rows):
            for bx in range(cols):
                y0 = by * bh
                x0 = bx * bw
                gx_block = gx[y0 : y0 + bh, x0 : x0 + bw]
                gy_block = gy[y0 : y0 + bh, x0 : x0 + bw]

                gxx = float((gx_block * gx_block).mean())
                gyy = float((gy_block * gy_block).mean())
                gxy = float((gx_block * gy_block).mean())

                energy = gxx + gyy
                if energy < self.min_energy:
                    quality_mask[y0 : y0 + bh, x0 : x0 + bw] = False
                    continue

                coherence = float(np.sqrt((gxx - gyy) ** 2 + (2 * gxy) ** 2)) / energy
                if coherence < self.coherence_threshold:
                    quality_mask[y0 : y0 + bh, x0 : x0 + bw] = False
                    continue

                ori_field[by, bx] = 0.5 * float(np.arctan2(2 * gxy, gxx - gyy))
                coh_field[by, bx] = coherence

        self.orientation_field = ori_field
        self.coherence_field = coh_field
        ctx.orientation_field = ori_field
        ctx.coherence_field = coh_field
        ctx.quality_mask = quality_mask

        valid_pct = 100.0 * quality_mask.mean()
        logger.debug(
            "OrientationFieldAnalyzer: %.1f%% valid, %dx%d tiles",
            valid_pct, rows, cols,
        )


class SingularityDetector(IPipelineStep):
    """Locates the Core (and Delta) singularities via the Poincaré index.

    Requires that a previous hook (e.g. :class:`OrientationFieldAnalyzer`)
    has populated ``ctx.orientation_field`` and ``ctx.coherence_field``.
    Produces a Region-of-Interest (ROI) disc centred on the strongest
    Core and writes it as the new ``ctx.quality_mask``.
    """

    def __init__(
        self,
        window: int = 5,
        poi_threshold: float = 0.5,
        roi_radius: int = 130,
    ) -> None:
        self.window = window
        self.poi_threshold = poi_threshold
        self.roi_radius = roi_radius
        self.core: tuple[int, int] | None = None
        self.delta: tuple[int, int] | None = None
        self.poi_map: np.ndarray | None = None

    def process(self, ctx: PipelineContext) -> None:
        image = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image
        h, w = image.shape[:2]
        roi_mask = np.ones((h, w), dtype=bool)

        if ctx.orientation_field is None:
            logger.debug("SingularityDetector: no orientation field, passing through")
            ctx.roi_mask = roi_mask
            ctx.quality_mask = roi_mask
            return

        # Smooth the orientation field in the (cos 2θ, sin 2θ) domain
        # so 180°-periodic angles average correctly.
        theta = ctx.orientation_field.astype(np.float32)
        cos2 = np.cos(2 * theta)
        sin2 = np.sin(2 * theta)
        kernel_size = 5
        cos2_s = cv2.blur(cos2, (kernel_size, kernel_size))
        sin2_s = cv2.blur(sin2, (kernel_size, kernel_size))
        theta_s = 0.5 * np.arctan2(sin2_s, cos2_s)

        rows, cols = theta_s.shape
        half = self.window // 2
        poi_map = np.zeros_like(theta_s, dtype=np.float32)

        for by in range(half, rows - half):
            for bx in range(half, cols - half):
                angles: list[float] = []
                for j in range(-half, half + 1):
                    angles.append(float(theta_s[by - half, bx + j]))
                for i in range(-half + 1, half + 1):
                    angles.append(float(theta_s[by + i, bx + half]))
                for j in range(half - 1, -half - 1, -1):
                    angles.append(float(theta_s[by + half, bx + j]))
                for i in range(half - 1, -half, -1):
                    angles.append(float(theta_s[by + i, bx - half]))

                diff_sum = 0.0
                for k in range(len(angles)):
                    a = angles[k]
                    b = angles[(k + 1) % len(angles)]
                    d = b - a
                    while d > np.pi / 2:
                        d -= np.pi
                    while d <= -np.pi / 2:
                        d += np.pi
                    diff_sum += d
                poi_map[by, bx] = diff_sum / np.pi

        self.poi_map = poi_map

        if (poi_map > self.poi_threshold).any():
            by, bx = np.unravel_index(np.argmax(poi_map), poi_map.shape)
            self.core = (int(bx), int(by))
        else:
            self.core = None

        if (poi_map < -self.poi_threshold).any():
            by, bx = np.unravel_index(np.argmin(poi_map), poi_map.shape)
            self.delta = (int(bx), int(by))
        else:
            self.delta = None

        if self.core is not None:
            block_size = self._infer_block_size(ctx)
            cx = self.core[0] * block_size + block_size // 2
            cy = self.core[1] * block_size + block_size // 2
            ctx.core = (int(cx), int(cy))
            yy, xx = np.mgrid[0:h, 0:w]
            roi_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.roi_radius ** 2
            logger.debug(
                "SingularityDetector: core at block=(%d,%d) pixel=(%d,%d), ROI=%.1f%%",
                self.core[0], self.core[1], cx, cy,
                100.0 * roi_mask.mean(),
            )
        else:
            ctx.core = None
            logger.debug("SingularityDetector: no core found, using full image as ROI")

        # Always expose the raw ROI disc so post-processors (e.g.
        # BorderMaskCleaner) can use it independently of the union
        # with the orientation-coherence mask.
        ctx.roi_mask = roi_mask

        # Intersect with any existing mask (e.g. from OrientationFieldAnalyzer)
        if ctx.quality_mask is not None and ctx.quality_mask.shape == roi_mask.shape:
            ctx.quality_mask = ctx.quality_mask & roi_mask
        else:
            ctx.quality_mask = roi_mask

    def _infer_block_size(self, ctx: PipelineContext) -> int:
        if ctx.orientation_field is None:
            return 16
        rows, cols = ctx.orientation_field.shape
        h, w = ctx.preprocessed_image.shape[:2] if ctx.preprocessed_image is not None else ctx.raw_image.shape[:2]
        return max(int(np.sqrt(h * w / max(rows * cols, 1))), 1)


class QualityMasker(IPipelineStep):
    """Legacy variance + Sobel coherence quality mask.

    Kept for benchmarking. New code should use
    :class:`OrientationFieldAnalyzer`.
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

    def process(self, ctx: PipelineContext) -> None:
        image = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape

        mask = np.ones((h, w), dtype=bool)
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                block = image[y : y + self.block_size, x : x + self.block_size]
                if block.size == 0:
                    continue
                if float(block.var()) < self.min_variance:
                    mask[y : y + self.block_size, x : x + self.block_size] = False

        gray_f32 = image.astype(np.float32)
        grad_x = cv2.Sobel(gray_f32, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f32, cv2.CV_32F, 0, 1, ksize=3)
        for y in range(0, h, self.block_size):
            for x in range(0, w, self.block_size):
                gx_block = grad_x[y : y + self.block_size, x : x + self.block_size]
                gy_block = grad_y[y : y + self.block_size, x : x + self.block_size]
                if gx_block.size == 0:
                    continue
                gxx = float((gx_block ** 2).mean())
                gyy = float((gy_block ** 2).mean())
                gxy = float((gx_block * gy_block).mean())
                denom = gxx + gyy
                if denom < 1e-6:
                    continue
                coherence = float(np.sqrt((gxx - gyy) ** 2 + 4 * gxy ** 2)) / denom
                if coherence < self.coherence_threshold:
                    mask[y : y + self.block_size, x : x + self.block_size] = False

        if ctx.quality_mask is not None and ctx.quality_mask.shape == mask.shape:
            ctx.quality_mask = ctx.quality_mask & mask
        else:
            ctx.quality_mask = mask

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

from src.core.config import config
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
        block_size: int | None = None,
        min_energy: float | None = None,
        coherence_threshold: float | None = None,
        gaussian_sigma: float = 0.0,
    ) -> None:
        oc = config.orientation_field
        self.block_size = block_size if block_size is not None else oc.block_size
        self.min_energy = min_energy if min_energy is not None else oc.min_energy
        self.coherence_threshold = (
            coherence_threshold if coherence_threshold is not None else oc.coherence_threshold
        )
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
    """Locates the Core (and Delta) singularities via the Poincaré index
    with DORIC validation to reject false positives from scars/creases.

    Requires that a previous hook (e.g. :class:`OrientationFieldAnalyzer`)
    has populated ``ctx.orientation_field`` and ``ctx.coherence_field``.
    Produces a Region-of-Interest (ROI) disc centred on the strongest
    valid Core and writes it as the new ``ctx.quality_mask``.

    Algorithm (Módulo 4 of LATENT_AFIS_SOTA):
    1. Convert ``θ → (Vx=cos(2θ), Vy=sin(2θ))`` for continuous representation.
    2. Strong Gaussian blur (σ=2.0) on Vx, Vy to suppress scar noise.
    3. Poincaré Index over 8-connected neighborhood.
    4. DORIC validation: fit candidate to theoretical Zero-Pole model;
       reject if RMS residual > 0.15 rad.
    """

    def __init__(
        self,
        gaussian_sigma: float = 2.0,
        poi_core_low: float = 0.25,
        poi_core_high: float = 0.75,
        poi_delta_low: float = -0.75,
        poi_delta_high: float = -0.25,
        doric_radius: int | None = None,
        doric_samples: int | None = None,
        doric_rms_threshold: float | None = None,
        roi_radius: int = 130,
        poi_divisor: float | None = None,
    ) -> None:
        doric = config.doric
        self.gaussian_sigma = gaussian_sigma
        self.poi_core_low = poi_core_low
        self.poi_core_high = poi_core_high
        self.poi_delta_low = poi_delta_low
        self.poi_delta_high = poi_delta_high
        self.doric_radius = doric_radius if doric_radius is not None else doric.radius
        self.doric_samples = doric_samples if doric_samples is not None else doric.n_samples
        self.doric_rms_threshold = doric_rms_threshold if doric_rms_threshold is not None else doric.rms_threshold
        self.roi_radius = roi_radius
        self._poi_divisor = poi_divisor if poi_divisor is not None else doric.poi_divisor
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

        # 1. Continuous vector field: θ → (Vx, Vy)
        theta = ctx.orientation_field.astype(np.float32)
        Vx = np.cos(2 * theta)
        Vy = np.sin(2 * theta)

        # 2. Strong Gaussian smoothing to suppress scar/crease noise
        Vx_smooth = cv2.GaussianBlur(Vx, (0, 0), sigmaX=self.gaussian_sigma, sigmaY=self.gaussian_sigma)
        Vy_smooth = cv2.GaussianBlur(Vy, (0, 0), sigmaX=self.gaussian_sigma, sigmaY=self.gaussian_sigma)

        # Reconstruct smoothed orientation
        theta_smooth = 0.5 * np.arctan2(Vy_smooth, Vx_smooth)

        rows, cols = theta_smooth.shape

        # 3. Poincaré Index over 8-connected neighborhood
        # PI = (1 / 2π) * Σ Δθᵢ
        # Core → Σ Δθ = +π → PI = +0.5
        # Delta → Σ Δθ = -π → PI = -0.5
        poi_map = np.zeros((rows, cols), dtype=np.float32)
        for by in range(1, rows - 1):
            for bx in range(1, cols - 1):
                neighbors = [
                    theta_smooth[by - 1, bx - 1],
                    theta_smooth[by - 1, bx],
                    theta_smooth[by - 1, bx + 1],
                    theta_smooth[by, bx + 1],
                    theta_smooth[by + 1, bx + 1],
                    theta_smooth[by + 1, bx],
                    theta_smooth[by + 1, bx - 1],
                    theta_smooth[by, bx - 1],
                ]
                diff_sum = 0.0
                for k in range(len(neighbors)):
                    d = neighbors[(k + 1) % len(neighbors)] - neighbors[k]
                    d = ((d + np.pi / 2) % np.pi) - np.pi / 2
                    diff_sum += d
                poi_map[by, bx] = diff_sum / self._poi_divisor

        self.poi_map = poi_map

        # 4. DORIC validation — collect valid candidates
        best_core: tuple[int, int, float] | None = None  # (bx, by, pi_value)
        best_delta: tuple[int, int, float] | None = None

        # We iterate over the whole map, then validate each candidate
        core_candidates = np.argwhere(
            (poi_map > self.poi_core_low) & (poi_map < self.poi_core_high)
        )
        delta_candidates = np.argwhere(
            (poi_map > self.poi_delta_low) & (poi_map < self.poi_delta_high)
        )

        for by, bx in core_candidates:
            if self._doric_validate(theta_smooth, int(by), int(bx), is_core=True):
                pi_val = float(poi_map[by, bx])
                if best_core is None or pi_val > best_core[2]:
                    best_core = (int(bx), int(by), pi_val)

        for by, bx in delta_candidates:
            if self._doric_validate(theta_smooth, int(by), int(bx), is_core=False):
                pi_val = float(poi_map[by, bx])
                if best_delta is None or abs(pi_val) > abs(best_delta[2]):
                    best_delta = (int(bx), int(by), pi_val)

        self.core = (best_core[0], best_core[1]) if best_core is not None else None
        self.delta = (best_delta[0], best_delta[1]) if best_delta is not None else None

        block_size = self._infer_block_size(ctx)

        if self.core is not None:
            cx = self.core[0] * block_size + block_size // 2
            cy = self.core[1] * block_size + block_size // 2
            ctx.core = (int(cx), int(cy))
            yy, xx = np.mgrid[0:h, 0:w]
            roi_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= self.roi_radius ** 2
            logger.debug(
                "SingularityDetector: core at block=(%d,%d) pixel=(%d,%d), "
                "PI=%.3f, ROI=%.1f%%",
                self.core[0], self.core[1], cx, cy,
                best_core[2] if best_core else 0.0,
                100.0 * roi_mask.mean(),
            )
        else:
            ctx.core = None
            logger.debug("SingularityDetector: no core found, using full image as ROI")

        if self.delta is not None:
            dx = self.delta[0] * block_size + block_size // 2
            dy = self.delta[1] * block_size + block_size // 2
            ctx.delta = (int(dx), int(dy))
            logger.debug(
                "SingularityDetector: delta at block=(%d,%d) pixel=(%d,%d)",
                self.delta[0], self.delta[1], dx, dy,
            )
        else:
            ctx.delta = None
            logger.debug("SingularityDetector: no delta found")

        ctx.roi_mask = roi_mask

        if ctx.quality_mask is not None and ctx.quality_mask.shape == roi_mask.shape:
            ctx.quality_mask = ctx.quality_mask & roi_mask
        else:
            ctx.quality_mask = roi_mask

    def _doric_validate(
        self,
        theta_smooth: np.ndarray,
        cy: int,
        cx: int,
        is_core: bool,
    ) -> bool:
        """DORIC: validate a singularity candidate against the theoretical
        Zero-Pole model.

        Samples the orientation field on a circle of radius ``r`` blocks
        around the candidate and checks that the angular progression
        matches ±0.5 rad/rad (core/delta) with RMS residual < threshold.
        """
        r = self.doric_radius
        N = self.doric_samples
        rows, cols = theta_smooth.shape

        sampled: list[float] = []
        for i in range(N):
            phi = 2.0 * np.pi * i / N
            sy = cy + r * np.sin(phi)
            sx = cx + r * np.cos(phi)
            if sy < 0 or sy >= rows or sx < 0 or sx >= cols:
                return False  # candidate too close to edge

            # Bilinear interpolation
            y0, x0 = int(sy), int(sx)
            y1, x1 = min(y0 + 1, rows - 1), min(x0 + 1, cols - 1)
            dy, dx = sy - y0, sx - x0
            val = (
                theta_smooth[y0, x0] * (1 - dy) * (1 - dx)
                + theta_smooth[y0, x1] * (1 - dy) * dx
                + theta_smooth[y1, x0] * dy * (1 - dx)
                + theta_smooth[y1, x1] * dy * dx
            )
            if is_core:
                expected = 0.5 * phi
            else:
                expected = -0.5 * phi

            diff = val - expected
            # Wrap to [-π/2, π/2]
            diff = ((diff + np.pi / 2) % np.pi) - np.pi / 2
            sampled.append(float(diff))

        # Estimate theta_0 (base orientation offset) as mean difference
        theta_0 = float(np.mean(sampled))
        residuals = [d - theta_0 for d in sampled]
        # Wrap residuals
        residuals = [((r + np.pi / 2) % np.pi) - np.pi / 2 for r in residuals]
        rms = float(np.sqrt(np.mean(np.array(residuals) ** 2)))

        return rms < self.doric_rms_threshold

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

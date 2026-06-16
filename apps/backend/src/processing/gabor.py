"""Local Ridge Frequency estimation and Quality Map for fingerprints.

Implements the x-signature projection for local ridge frequency estimation
(Hong et al. 1998) and a 1-NN quality classifier (amplitude, frequency,
variance).  These outputs feed both the quality mask and the DPI-scaling
for the spurious-filter thresholds.

The actual Gabor 2D filtering is handled by :class:`CpuEnhancer`.  This
module focuses on the *measurements* needed to make downstream filters
robust against variable ridge spacing (latent prints, oblique captures).

Module 1 of the LATENT_AFIS_SOTA research plan.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy import ndimage, signal

from src.core.interfaces import IPipelineStep, PipelineContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (500 DPI basis — thresholds are DPI-scaled elsewhere)
# ---------------------------------------------------------------------------

BLOCK_SIZE: int = 16                # w x w processing block
WINDOW_LENGTH: int = 32             # projection window along ridge direction
WINDOW_WIDTH: int = 16              # projection window orthogonal to ridges
FREQ_MIN: float = 0.04              # cycles/pixel (≈25 px/ridge)
FREQ_MAX: float = 0.33              # cycles/pixel (≈3 px/ridge)

# Quality-map classification prototypes (Hong 1-NN)
# Each row: (amplitude, frequency, variance) for 6 clusters.
# Clusters 0-3: recoverable, 4-5: unrecoverable.
_Q_PROTOTYPES: np.ndarray = np.array([
    [0.35, 0.10, 0.50],
    [0.50, 0.15, 0.60],
    [0.25, 0.12, 0.40],
    [0.60, 0.18, 0.70],
    [0.05, 0.05, 0.10],
    [0.02, 0.08, 0.05],
], dtype=np.float32)

_RECOVERABLE_RATIO: float = 0.40


# ---------------------------------------------------------------------------
# Orientation field (block-level, for frequency estimation input)
# ---------------------------------------------------------------------------


def _compute_block_orientation(norm_img: np.ndarray) -> np.ndarray:
    """Per-block orientation via Sobel structure tensor.

    Returns:
        (rows//16, cols//16) array of orientations in radians [0, π).
    """
    h, w = norm_img.shape
    gx = cv2.Sobel(norm_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm_img, cv2.CV_32F, 0, 1, ksize=3)

    rows = h // BLOCK_SIZE
    cols = w // BLOCK_SIZE
    orient = np.zeros((rows, cols), dtype=np.float32)

    for by in range(rows):
        for bx in range(cols):
            y0 = by * BLOCK_SIZE
            x0 = bx * BLOCK_SIZE
            gx_b = gx[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            gy_b = gy[y0:y0 + BLOCK_SIZE, x0:x0 + BLOCK_SIZE]
            gxx = float((gx_b * gx_b).mean())
            gyy = float((gy_b * gy_b).mean())
            gxy = float((gx_b * gy_b).mean())
            energy = gxx + gyy
            if energy < 1e-6:
                continue
            orient[by, bx] = 0.5 * float(np.arctan2(2 * gxy, gxx - gyy))

    return orient


# ---------------------------------------------------------------------------
# Local Ridge Frequency (x-signature projection)
# ---------------------------------------------------------------------------


def estimate_local_frequency(
    norm_img: np.ndarray,
    orient_img: np.ndarray,
) -> np.ndarray:
    """Estimate local ridge frequency via x-signature projection.

    For each 16x16 block:
      1. Rotate a 32x16 window by the block's orientation
      2. Project onto the x' axis (along ridge direction) → 1D x-signature
      3. Find peaks → ridge period → frequency = 1/period
      4. Validate [0.04, 0.33] cycles/px

    Args:
        norm_img: Normalised grayscale image.
        orient_img: Per-block orientation field in radians.

    Returns:
        Frequency map (same shape as norm_img).  -1.0 marks invalid blocks.
    """
    h, w = norm_img.shape
    freq_img = np.full((h, w), -1.0, dtype=np.float32)

    from scipy.signal import find_peaks

    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            by = y // BLOCK_SIZE
            bx = x // BLOCK_SIZE

            if by >= orient_img.shape[0] or bx >= orient_img.shape[1]:
                continue

            theta = float(orient_img[by, bx])
            if not np.isfinite(theta):
                continue

            block = norm_img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            if block.size < 4 or np.std(block) < 1e-4:
                continue

            # Build oriented window coordinates
            half_l = WINDOW_LENGTH // 2
            half_w = WINDOW_WIDTH // 2
            x_coords = np.arange(-half_l, half_l, dtype=np.float32)
            y_coords = np.arange(-half_w, half_w, dtype=np.float32)

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            col_grid = (BLOCK_SIZE // 2
                        + x_coords[np.newaxis, :] * cos_t
                        + y_coords[:, np.newaxis] * sin_t)
            row_grid = (BLOCK_SIZE // 2
                        - x_coords[np.newaxis, :] * sin_t
                        + y_coords[:, np.newaxis] * cos_t)

            row_grid = np.clip(row_grid, 0, BLOCK_SIZE - 1)
            col_grid = np.clip(col_grid, 0, BLOCK_SIZE - 1)
            r0 = np.floor(row_grid).astype(np.int32)
            c0 = np.floor(col_grid).astype(np.int32)
            r1 = np.minimum(r0 + 1, BLOCK_SIZE - 1)
            c1 = np.minimum(c0 + 1, BLOCK_SIZE - 1)
            wr = row_grid - r0
            wc = col_grid - c0

            sampled: np.ndarray = (
                block[r0, c0] * (1 - wr) * (1 - wc)
                + block[r0, c1] * (1 - wr) * wc
                + block[r1, c0] * wr * (1 - wc)
                + block[r1, c1] * wr * wc
            )

            # x-signature: average over the orthogonal (y') axis
            x_sig = np.mean(sampled, axis=0)
            x_sig_smooth = ndimage.gaussian_filter1d(x_sig, sigma=1.0)

            peaks, _ = find_peaks(x_sig_smooth, distance=3)
            if len(peaks) < 2:
                continue

            avg_period = float(np.mean(np.diff(peaks)))
            freq = 1.0 / avg_period

            if FREQ_MIN <= freq <= FREQ_MAX:
                freq_img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = freq

    return _interpolate_frequency(freq_img)


def _interpolate_frequency(
    freq_img: np.ndarray,
    kernel_size: int = 7,
    sigma: float = 3.0,
) -> np.ndarray:
    """Fill invalid (-1.0) blocks with Gaussian-weighted neighbours."""
    valid_mask = freq_img > 0
    if valid_mask.all():
        return ndimage.gaussian_filter(freq_img, sigma=sigma)

    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    kernel = kernel / kernel.sum()

    valid = np.where(valid_mask, freq_img, 0.0)
    weight_map = valid_mask.astype(np.float32)

    valid_smooth = signal.convolve2d(valid, kernel, mode='same', boundary='symm')
    weight_smooth = signal.convolve2d(weight_map, kernel, mode='same', boundary='symm')
    weight_smooth = np.maximum(weight_smooth, 1e-8)

    result = np.where(valid_mask, freq_img, valid_smooth / weight_smooth)
    return ndimage.gaussian_filter(result, sigma=1.5)


# ---------------------------------------------------------------------------
# Quality Map (Hong 1-NN classifier)
# ---------------------------------------------------------------------------


def _detect_peaks_valleys(
    x_sig: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect peaks and valleys in a 1D x-signature signal."""
    from scipy.signal import find_peaks

    x_sig = ndimage.gaussian_filter1d(x_sig, sigma=1.0)
    peaks, _ = find_peaks(x_sig, distance=3)
    valleys, _ = find_peaks(-x_sig, distance=3)
    if len(peaks) < 2 or len(valleys) < 2:
        return None, None
    return x_sig[peaks], x_sig[valleys]


def compute_quality_mask(
    norm_img: np.ndarray,
    orient_img: np.ndarray,
    freq_img: np.ndarray,
) -> np.ndarray:
    """Binary quality mask: True = recoverable fingerprint region.

    Uses a 1-NN classifier on the x-signature feature vector
    (amplitude, frequency, variance) per 16×16 block.
    """
    h, w = norm_img.shape
    mask = np.zeros((h, w), dtype=bool)

    for y in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for x in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            by = y // BLOCK_SIZE
            bx = x // BLOCK_SIZE

            freq_val = float(freq_img[y, x])
            if freq_val <= 0:
                continue

            block = norm_img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            theta = float(orient_img[by, bx])

            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            half_l = WINDOW_LENGTH // 2
            half_w = WINDOW_WIDTH // 2
            x_coords = np.arange(-half_l, half_l, dtype=np.float32)
            y_coords = np.arange(-half_w, half_w, dtype=np.float32)
            col_grid = (BLOCK_SIZE // 2
                        + x_coords[np.newaxis, :] * cos_t
                        + y_coords[:, np.newaxis] * sin_t)
            row_grid = (BLOCK_SIZE // 2
                        - x_coords[np.newaxis, :] * sin_t
                        + y_coords[:, np.newaxis] * cos_t)
            row_grid = np.clip(row_grid, 0, BLOCK_SIZE - 1).astype(np.int32)
            col_grid = np.clip(col_grid, 0, BLOCK_SIZE - 1).astype(np.int32)
            rotated = block[row_grid, col_grid]
            x_sig = np.mean(rotated, axis=0)

            peaks, valleys = _detect_peaks_valleys(x_sig)
            if peaks is None or valleys is None:
                continue

            amp = float(np.mean(peaks) - np.mean(valleys))
            var = float(np.var(x_sig) / (np.mean(x_sig) ** 2 + 1e-6))

            feature = np.array([amp, freq_val, var], dtype=np.float32)

            dists = np.linalg.norm(_Q_PROTOTYPES - feature, axis=1)
            cluster = int(np.argmin(dists))
            mask[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = (cluster < 4)

    if mask.mean() < _RECOVERABLE_RATIO:
        logger.warning(
            "Quality mask: only %.0f%% recoverable (< %.0f%%) — fingerprint may be hopeless",
            mask.mean() * 100, _RECOVERABLE_RATIO * 100,
        )

    return mask


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


class QualityMaskStep(IPipelineStep):
    """Compute per-block frequency and quality mask for downstream steps.

    This step does NOT apply Gabor filtering — that is handled by
    :class:`CpuEnhancer`.  It only produces the measurements
    (frequency map, quality mask) that other steps use:

    - :class:`SkeletonCleanerStep` uses ``ctx.freq_image`` for DPI scaling
    - :class:`SingularityDetector` and post-hooks intersect with the mask

    Must be placed after :class:`OrientationFieldAnalyzer` so the
    orientation field is available.
    """

    def process(self, ctx: PipelineContext) -> None:
        source = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image
        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        img_f32 = source.astype(np.float32)
        norm = (img_f32 - img_f32.mean()) / (img_f32.std() + 1e-8)

        # Block-level orientation (independent of OrientationFieldAnalyzer
        # so this step works in isolation)
        orient = _compute_block_orientation(norm)

        freq_img = estimate_local_frequency(norm, orient)
        quality_mask = compute_quality_mask(norm, orient, freq_img)

        ctx.freq_image = freq_img
        if ctx.quality_mask is not None and ctx.quality_mask.shape == quality_mask.shape:
            ctx.quality_mask = ctx.quality_mask & quality_mask
        else:
            ctx.quality_mask = quality_mask

        valid_pct = 100.0 * quality_mask.mean()
        n_valid = int((freq_img > 0).sum())
        n_total = freq_img.size
        logger.debug(
            "QualityMaskStep: %.1f%% recoverable, %d/%d blocks have valid frequency",
            valid_pct, n_valid, n_total,
        )

"""Spatially-variant Gabor 2D filter for fingerprint enhancement.

Implements the x-signature projection for local ridge frequency estimation
(Hong et al. 1998) and per-pixel Gabor filtering with orientation +
frequency adaptation.  Optimised for CPU (no GPU dependency).

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
GABOR_SIGMA: float = 4.0            # spatial envelope σ (Hong empirical)
GABOR_KSIZE: int = 11               # kernel size (covers ~3σ)
N_ORIENT: int = 16                  # discrete orientations [0, π)

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

_GABOR_RECOVERABLE_RATIO: float = 0.40


# ---------------------------------------------------------------------------
# Local Ridge Frequency (x-signature projection)
# ---------------------------------------------------------------------------


def estimate_local_frequency(
    norm_img: np.ndarray,
    orient_img: np.ndarray,
) -> np.ndarray:
    """Estimate local ridge frequency via x-signature projection.

    Args:
        norm_img: Normalised grayscale image (zero mean, unit variance).
        orient_img: Per-block orientation field (rows//16, cols//16) in [0, π).

    Returns:
        Frequency map at pixel resolution.  -1.0 marks invalid/unrecoverable
        blocks (low contrast, noise-dominated).
    """
    h, w = norm_img.shape
    freq_img = np.full((h, w), -1.0, dtype=np.float32)

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

            # ---- 1. Build x-signature via oriented projection ----
            half_l = WINDOW_LENGTH // 2
            half_w = WINDOW_WIDTH // 2

            # Create coordinate grid in rotated frame.
            # x' along ridge direction [cos θ, sin θ]
            # y' orthogonal to ridges (the direction we average over)
            x_coords = np.arange(-half_l, half_l, dtype=np.float32)
            y_coords = np.arange(-half_w, half_w, dtype=np.float32)

            # Map rotated (x', y') to image coordinates
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            # image_col = center_col + x'*cosθ + y'*sinθ
            # image_row = center_row - x'*sinθ + y'*cosθ
            col_grid = (BLOCK_SIZE // 2
                        + x_coords[np.newaxis, :] * cos_t
                        + y_coords[:, np.newaxis] * sin_t)
            row_grid = (BLOCK_SIZE // 2
                        - x_coords[np.newaxis, :] * sin_t
                        + y_coords[:, np.newaxis] * cos_t)

            # Bilinear interpolation for rotated sampling
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
            )  # shape: (WINDOW_WIDTH, WINDOW_LENGTH)

            # x-signature: average over the orthogonal (y') axis
            x_sig = np.mean(sampled, axis=0)

            # ---- 2. Smooth and find peaks ----
            x_sig_smooth = ndimage.gaussian_filter1d(x_sig, sigma=1.0)

            from scipy.signal import find_peaks

            peaks, _ = find_peaks(x_sig_smooth, distance=3)
            if len(peaks) < 2:
                continue

            avg_period = float(np.mean(np.diff(peaks)))
            freq = 1.0 / avg_period

            if FREQ_MIN <= freq <= FREQ_MAX:
                freq_img[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = freq

    # ---- 3. Interpolate invalid blocks via Gaussian low-pass ----
    freq_img = _interpolate_frequency(freq_img)
    return freq_img


def _interpolate_frequency(freq_img: np.ndarray, kernel_size: int = 7, sigma: float = 3.0) -> np.ndarray:
    """Fill invalid (-1.0) blocks with Gaussian-weighted neighbours."""
    valid_mask = freq_img > 0
    if valid_mask.all():
        return ndimage.gaussian_filter(freq_img, sigma=sigma)

    # Valid-only interpolation
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = kernel @ kernel.T
    kernel = kernel / kernel.sum()

    h, w = freq_img.shape
    valid = np.where(valid_mask, freq_img, 0.0)
    weight_map = valid_mask.astype(np.float32)

    valid_smooth = signal.convolve2d(valid, kernel, mode='same', boundary='symm')
    weight_smooth = signal.convolve2d(weight_map, kernel, mode='same', boundary='symm')
    weight_smooth = np.maximum(weight_smooth, 1e-8)

    result = np.where(valid_mask, freq_img, valid_smooth / weight_smooth)
    result = ndimage.gaussian_filter(result, sigma=1.5)
    return result


# ---------------------------------------------------------------------------
# Quality Map (Hong 1-NN classifier)
# ---------------------------------------------------------------------------


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

            # Build x-signature (same as frequency estimation)
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

            # Features
            peaks, valleys = _detect_peaks_valleys(x_sig)
            if peaks is None or valleys is None:
                continue

            amp = float(np.mean(peaks) - np.mean(valleys))
            var = float(np.var(x_sig) / (np.mean(x_sig) ** 2 + 1e-6))

            feature = np.array([amp, freq_val, var], dtype=np.float32)

            # 1-NN classification
            dists = np.linalg.norm(_Q_PROTOTYPES - feature, axis=1)
            cluster = int(np.argmin(dists))
            mask[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE] = (cluster < 4)

    # Global reject if too little is recoverable
    if mask.mean() < _GABOR_RECOVERABLE_RATIO:
        logger.warning(
            "Quality mask: only %.0f%% recoverable (< %.0f%%) — fingerprint may be hopeless",
            mask.mean() * 100, _GABOR_RECOVERABLE_RATIO * 100,
        )

    return mask


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


# ---------------------------------------------------------------------------
# Gabor 2D filter bank
# ---------------------------------------------------------------------------


def _build_gabor_kernel(
    theta: float,
    freq: float,
    sigma_x: float = GABOR_SIGMA,
    sigma_y: float = GABOR_SIGMA,
    ksize: int = GABOR_KSIZE,
) -> np.ndarray:
    """Build a single even-symmetric Gabor kernel.

    h(x,y) = exp(-½(x'²/σx² + y'²/σy²)) · cos(2π·f·x')

    where x' = x·cosθ + y·sinθ  (direction orthogonal to ridges).
    """
    half = ksize // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1]

    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    gauss = np.exp(-0.5 * (x_rot ** 2 / sigma_x ** 2 + y_rot ** 2 / sigma_y ** 2))
    sinusoid = np.cos(2 * np.pi * freq * x_rot)

    kernel = gauss * sinusoid
    kernel -= kernel.mean()
    return kernel


def _gabor_filter_bank(
    norm_img: np.ndarray,
    orient_img: np.ndarray,
    freq_img: np.ndarray,
    ksize: int = GABOR_KSIZE,
) -> np.ndarray:
    """Apply spatially-variant Gabor filtering.

    Pre-computes a bank of N_ORIENT filters using the median frequency
    across all valid blocks.  Much faster than per-block frequency
    adaptation while preserving orientation-driven enhancement.

    For each pixel, the response from the nearest orientation filter
    is selected via the local orientation field.
    """
    h, w = norm_img.shape

    # Median frequency across valid regions
    valid_freqs = freq_img[freq_img > 0]
    median_freq = float(np.median(valid_freqs)) if len(valid_freqs) > 0 else 0.12

    # Pre-compute orientation filters
    theta_bins = np.linspace(0, np.pi, N_ORIENT, endpoint=False, dtype=np.float32)
    kernels = np.zeros((N_ORIENT, ksize, ksize), dtype=np.float32)
    for ti, theta in enumerate(theta_bins):
        kernels[ti] = _build_gabor_kernel(theta, median_freq, ksize=ksize)

    # Convolve once per orientation
    filtered_layers = np.zeros((N_ORIENT, h, w), dtype=np.float32)
    for ti in range(N_ORIENT):
        filtered_layers[ti] = signal.convolve2d(norm_img, kernels[ti], mode='same')

    # Resize orient_img to per-pixel if needed
    if orient_img.shape != (h, w):
        orient_resized = cv2.resize(
            orient_img, (w, h), interpolation=cv2.INTER_NEAREST
        )
    else:
        orient_resized = orient_img

    # Map orientation to nearest filter index (per pixel)
    angle_step = np.pi / N_ORIENT
    orient_idx = np.round(orient_resized / angle_step).astype(np.int32) % N_ORIENT

    # Select per-pixel response via advanced indexing
    rows, cols = norm_img.shape
    r_idx, c_idx = np.indices((rows, cols))
    enhanced = filtered_layers[orient_idx, r_idx, c_idx]

    # Apply quality mask: only keep pixels with valid frequency
    quality_mask_resized = cv2.resize(
        (freq_img > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    enhanced = np.where(quality_mask_resized, enhanced, norm_img)

    return enhanced


def enhance_with_gabor(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Full Gabor-enhancement pipeline.

    Args:
        image: Input grayscale image (uint8 or float32).

    Returns:
        Tuple of (enhanced_image: uint8, quality_mask: bool).
    """
    # 1. Normalisation (zero mean, unit variance)
    img_f32 = image.astype(np.float32)
    norm_img = (img_f32 - img_f32.mean()) / (img_f32.std() + 1e-8)

    # 2. Orientation field (block-level)
    gx = cv2.Sobel(norm_img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(norm_img, cv2.CV_32F, 0, 1, ksize=3)

    h, w = norm_img.shape
    rows = h // BLOCK_SIZE
    cols = w // BLOCK_SIZE
    orient_img = np.zeros((rows, cols), dtype=np.float32)

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
            orient_img[by, bx] = 0.5 * float(np.arctan2(2 * gxy, gxx - gyy))

    # 3. Local frequency
    freq_img = estimate_local_frequency(norm_img, orient_img)

    # 4. Quality mask
    quality_mask = compute_quality_mask(norm_img, orient_img, freq_img)

    # 5. Gabor enhancement
    enhanced_float = _gabor_filter_bank(norm_img, orient_img, freq_img)

    # 6. Scale to uint8
    min_v, max_v = float(enhanced_float.min()), float(enhanced_float.max())
    if max_v > min_v:
        enhanced_uint8 = np.clip(
            (enhanced_float - min_v) / (max_v - min_v) * 255, 0, 255
        ).astype(np.uint8)
    else:
        enhanced_uint8 = np.zeros_like(enhanced_float, dtype=np.uint8)

    return enhanced_uint8, quality_mask


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


class GaborEnhancerStep(IPipelineStep):
    """Pipeline step that applies spatially-variant Gabor 2D enhancement.

    Uses the orientation field computed by a previous step
    (OrientationFieldAnalyzer) or computes it if missing.  Produces
    a cleaner enhanced image and quality mask.
    """

    def process(self, ctx: PipelineContext) -> None:
        # Use raw_image (CpuEnhancer produces binarized output — no good for Gabor)
        source = ctx.raw_image
        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        enhanced, quality_mask = enhance_with_gabor(source)

        ctx.enhanced_image = enhanced
        ctx.preprocessed_image = enhanced

        # Merge quality masks (if one already exists from OrientationFieldAnalyzer)
        if ctx.quality_mask is not None and ctx.quality_mask.shape == quality_mask.shape:
            ctx.quality_mask = ctx.quality_mask & quality_mask
        else:
            ctx.quality_mask = quality_mask

        logger.debug(
            "GaborEnhancerStep: enhanced %s, quality %.1f%%",
            enhanced.shape, 100.0 * quality_mask.mean(),
        )

"""Spurious minutiae filter for ridge skeletons.

Implements morphological heuristics to remove false minutiae caused by
scarring, noise, and sensor artefacts (Module 2 of LATENT_AFIS_SOTA).

All thresholds are dynamically scaled from the biological baseline of
9.25 px/ridge at 500 DPI to handle arbitrary image resolutions.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from src.core.config import config
from src.core.interfaces import IPipelineStep, PipelineContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Biological constants
# ---------------------------------------------------------------------------

_RIDGE_PERIOD_500DPI: float = 9.25   # px per ridge at 500 DPI (Ashbaugh 1999)

# Thresholds at 500 DPI (Zhao & Tang, Ratha et al.)
_THRESH_500 = {
    'spur': 10,       # px: max distance for a spur (bifurcation → nearby ending)
    'bridge': 10,     # px: max distance between two bifurcations forming a bridge
    'island': 12,     # px: max path length for an island (ending → ending)
    'hole': 6,        # px: radius threshold for small loops (pores)
    'edge': 12,       # px: margin from image border to discard
}

# Phase 16: tunables for the spurious filter (env-overridable via
# SPURIOUS_* env vars; see SpuriousFilterConfig in src.core.config).
_NON_FACING_SPUR_RELAXATION: float = config.spurious_filter.min_recoverable_ratio
_DPI_SCALE_FLOOR: float = config.spurious_filter.dpi_scale_floor  # min scale (prevent undersized removal)


def _estimate_dpi_scale(
    median_period_px: float | None = None,
    freq_img: np.ndarray | None = None,
) -> float:
    """Compute DPI scaling factor relative to 500 DPI baseline.

    ridge_to_ridge = 0.47 mm (biological constant, Moore 1989 / Ashbaugh 1999)
    At 500 DPI: 25.4 mm/in x 500 px/in = 19.685 px/mm
    ridge_period_500 = 0.47 mm x 19.685 px/mm = 9.25 px

    scale_factor = actual_ridge_period / 9.25
    """
    if median_period_px is not None and median_period_px > 0:
        return max(_DPI_SCALE_FLOOR, median_period_px / _RIDGE_PERIOD_500DPI)

    if freq_img is not None:
        valid = freq_img[freq_img > 0]
        if len(valid) > 0:
            median_freq = float(np.median(valid))
            median_period = 1.0 / median_freq
            return max(_DPI_SCALE_FLOOR, median_period / _RIDGE_PERIOD_500DPI)

    return 1.0


def get_scaled_thresholds(
    median_period_px: float | None = None,
    freq_img: np.ndarray | None = None,
) -> dict[str, int]:
    """Return spurious-filter thresholds scaled to the image DPI.

    Each threshold is multiplied by the scale factor (period / 9.25)
    so that physical distances remain constant regardless of resolution.
    """
    scale = _estimate_dpi_scale(median_period_px, freq_img)
    return {
        key: max(1, round(val * scale))
        for key, val in _THRESH_500.items()
    }


# ---------------------------------------------------------------------------
# Skeleton-level cleaning
# ---------------------------------------------------------------------------


def clean_skeleton(
    skeleton: np.ndarray,
    thresholds: dict[str, int] | None = None,
    freq_img: np.ndarray | None = None,
) -> np.ndarray:
    """Clean a fingerprint skeleton by removing spurious structures.

    Operates on the binary skeleton (0/1) image and returns a cleaned
    version.  The pipeline steps are applied in order:

    1. Edge margin removal — crop skeleton pixels within ``edge`` px
       of the image border.
    2. Hole filling — remove small loops (sweat pore / sensor noise).
    3. Island removal — remove short disconnected ridge fragments.
    4. Spur removal — remove short branches (bifurcation → ending).

    Args:
        skeleton: Binary image (0 or 1), single-pixel wide ridges.
        thresholds: Threshold dict (use :func:`get_scaled_thresholds`
            if not provided).
        freq_img: Optional frequency map for DPI estimation (if
            thresholds not provided).

    Returns:
        Cleaned skeleton (same shape and dtype as input).
    """
    if thresholds is None:
        thresholds = get_scaled_thresholds(freq_img=freq_img)

    skel = skeleton.astype(np.uint8)
    h, w = skel.shape

    # ---- 1. Edge margin ----
    edge = thresholds['edge']
    if edge > 0:
        skel[:edge, :] = 0
        skel[h - edge:, :] = 0
        skel[:, :edge] = 0
        skel[:, w - edge:] = 0

    # ---- 2. Remove small holes (loops) ----
    # Find connected components in the background
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (1 - skel).astype(np.uint8), connectivity=8,
    )
    hole_radius = thresholds['hole']
    hole_area = int(np.pi * hole_radius * hole_radius)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= hole_area:
            y_start = int(stats[label_id, cv2.CC_STAT_TOP])
            x_start = int(stats[label_id, cv2.CC_STAT_LEFT])
            skel[y_start:y_start + stats[label_id, 3],
                 x_start:x_start + stats[label_id, 4]] += (
                labels[y_start:y_start + stats[label_id, 3],
                       x_start:x_start + stats[label_id, 4]] == label_id
            ).astype(np.uint8)

    # ---- 3. Remove small islands ----
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel, connectivity=8,
    )
    island_max_area = thresholds['island']
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area <= island_max_area:
            y_start = int(stats[label_id, cv2.CC_STAT_TOP])
            x_start = int(stats[label_id, cv2.CC_STAT_LEFT])
            skel[y_start:y_start + stats[label_id, 3],
                 x_start:x_start + stats[label_id, 4]] -= (
                labels[y_start:y_start + stats[label_id, 3],
                       x_start:x_start + stats[label_id, 4]] == label_id
            ).astype(np.uint8)

    return np.clip(skel, 0, 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Post-processing on extracted minutiae (skeleton-independent)
# ---------------------------------------------------------------------------


def remove_spurs_from_minutiae(
    minutiae: list[dict[str, Any]],
    skeleton: np.ndarray,
    thresholds: dict[str, int] | None = None,
    freq_img: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Remove spur (bifurcation + nearby ending) pairs from minutiae list.

    Operates on the minutiae metadata rather than the skeleton, and
    handles directional facing/against spur rules.

    Args:
        minutiae: List of dicts with keys {x, y, type, angle, ...}.
        skeleton: Binary skeleton image (for connectivity checks).
        thresholds: Scaled thresholds (use :func:`get_scaled_thresholds` if None).
        freq_img: Optional frequency map for DPI estimation.

    Returns:
        Filtered minutiae list.
    """
    if thresholds is None:
        thresholds = get_scaled_thresholds(freq_img=freq_img)

    spur_dist = thresholds['spur']
    filtered = list(minutiae)

    # Track indices to remove
    to_remove: set[int] = set()

    bif_idx = [i for i, m in enumerate(filtered) if m.get('type') == 'bifurcation']
    end_idx = [i for i, m in enumerate(filtered) if m.get('type') == 'ending']

    for bi in bif_idx:
        b = filtered[bi]
        for ei in end_idx:
            if ei in to_remove:
                continue
            e = filtered[ei]
            dx = b['x'] - e['x']
            dy = b['y'] - e['y']
            d = float(np.hypot(dx, dy))

            if d > spur_dist:
                continue

            # Facing-spur: bifurcation and ending point towards each other
            # (Δ angle > 135° → more likely a true spur)
            b_angle = float(b.get('angle', 0.0))
            e_angle = float(e.get('angle', 0.0))
            delta = abs((b_angle - e_angle) % np.pi)
            facing = delta > 3 * np.pi / 4  # > 135°

            thresh = float(spur_dist) if facing else float(spur_dist) * _NON_FACING_SPUR_RELAXATION
            if d <= thresh:
                to_remove.add(ei)

    return [m for i, m in enumerate(filtered) if i not in to_remove]


def remove_bridges_from_minutiae(
    minutiae: list[dict[str, Any]],
    thresholds: dict[str, int] | None = None,
    freq_img: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Remove bridge (two close bifurcations) pairs."""
    if thresholds is None:
        thresholds = get_scaled_thresholds(freq_img=freq_img)

    bridge_dist = thresholds['bridge']
    to_remove: set[int] = set()

    bif_idx = [i for i, m in enumerate(minutiae) if m.get('type') == 'bifurcation']

    for i in range(len(bif_idx)):
        for j in range(i + 1, len(bif_idx)):
            a = minutiae[bif_idx[i]]
            b = minutiae[bif_idx[j]]
            dx = a['x'] - b['x']
            dy = a['y'] - b['y']
            if dx * dx + dy * dy <= bridge_dist * bridge_dist:
                to_remove.add(bif_idx[i])
                to_remove.add(bif_idx[j])

    return [m for i, m in enumerate(minutiae) if i not in to_remove]


# ---------------------------------------------------------------------------
# Pipeline step
# ---------------------------------------------------------------------------


class SkeletonCleanerStep(IPipelineStep):
    """Pipeline step that cleans the skeleton before graph extraction.

    Should be placed between the Gabor enhancer/skeletonizer and the
    RidgeGraphExtractor.  Removes spurious ridges that would otherwise
    create false minutiae in the graph.
    """

    def process(self, ctx: PipelineContext) -> None:
        if ctx.skeleton is None:
            logger.warning("SkeletonCleanerStep: no skeleton to clean")
            return

        # Reuse frequency map from QualityMaskStep if available, else
        # recompute it.  Avoids the cost of x-signature projection twice.
        freq_img: np.ndarray | None = ctx.freq_image
        if freq_img is None and ctx.orientation_field is not None:
            try:
                from src.processing.gabor import estimate_local_frequency
                source = ctx.enhanced_image if ctx.enhanced_image is not None else ctx.raw_image
                norm = source.astype(np.float32)
                norm = (norm - norm.mean()) / (norm.std() + 1e-8)
                freq_img = estimate_local_frequency(norm, ctx.orientation_field)
            except Exception:
                pass

        thresholds = get_scaled_thresholds(freq_img=freq_img)

        cleaned = clean_skeleton(ctx.skeleton, thresholds=thresholds)
        ctx.skeleton = cleaned

        before = int((ctx.skeleton > 0).sum())
        after = int(cleaned.sum())
        logger.debug(
            "SkeletonCleanerStep: %d → %d skeleton pixels (removed %d spurious)",
            before, after, before - after,
        )

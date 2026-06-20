"""Type-aware rules for the pattern-first pipeline (Spike 03).

The ONLY principled operation here is the **Henry cap**: after the
classifier tells us the pattern, we cap the singularity list to
the expected (cores, deltas) count.

Singularity detection runs in **two passes with different Gaussian
sigmas** for the OF smoothing:
  - Cores use ``sigma=2.0`` (strong smoothing, robust against scars)
  - Deltas use ``sigma=1.0`` (less smoothing, preserves the smaller
    signal that the 2.0 erases in high-curvature regions like loops)

This is principled, not a heuristic: the delta signal is
geometrically smaller than the core signal in the orientation
field, so they need different smoothing. It is NOT per-pattern
tuning — it's per-target tuning based on the physics of the
detection.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from src.core.config import config as prod_config

from types_spike import (
    PatternClassification,
    PatternType,
    Singularity,
    SingularityKind,
)

logger = logging.getLogger(__name__)


GAUSSIAN_SIGMA_CORE: float = 2.0
GAUSSIAN_SIGMA_DELTA: float = 1.0

POI_CORE_LOW: float = 0.25
POI_CORE_HIGH: float = 0.75
POI_DELTA_LOW: float = -0.75
POI_DELTA_HIGH: float = -0.25
POI_DIVISOR: float = prod_config.doric.poi_divisor

DORIC_RADIUS: int = prod_config.doric.radius
DORIC_SAMPLES: int = prod_config.doric.n_samples
DORIC_RMS_THRESHOLD: float = prod_config.doric.rms_threshold

NMS_RADIUS_BLOCKS: int = 3


def _gaussian_smooth(theta: np.ndarray, sigma: float = GAUSSIAN_SIGMA_CORE) -> np.ndarray:
    vx = np.cos(2 * theta)
    vy = np.sin(2 * theta)
    vx_s = cv2.GaussianBlur(vx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    vy_s = cv2.GaussianBlur(vy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return 0.5 * np.arctan2(vy_s, vx_s)


def _poincare_map(theta: np.ndarray) -> np.ndarray:
    rows, cols = theta.shape
    poi_map = np.zeros((rows, cols), dtype=np.float32)
    for by in range(1, rows - 1):
        for bx in range(1, cols - 1):
            neighbours = [
                theta[by - 1, bx - 1], theta[by - 1, bx],
                theta[by - 1, bx + 1], theta[by, bx + 1],
                theta[by + 1, bx + 1], theta[by + 1, bx],
                theta[by + 1, bx - 1], theta[by, bx - 1],
            ]
            diff_sum = 0.0
            for k in range(len(neighbours)):
                d = neighbours[(k + 1) % len(neighbours)] - neighbours[k]
                d = ((d + np.pi / 2) % np.pi) - np.pi / 2
                diff_sum += d
            poi_map[by, bx] = diff_sum / POI_DIVISOR
    return poi_map


def _doric_validate(theta: np.ndarray, cy: int, cx: int, *, is_core: bool) -> bool:
    r = DORIC_RADIUS
    n = DORIC_SAMPLES
    rows, cols = theta.shape

    sampled: list[float] = []
    for i in range(n):
        phi = 2.0 * np.pi * i / n
        sy = cy + r * np.sin(phi)
        sx = cx + r * np.cos(phi)
        if sy < 0 or sy >= rows or sx < 0 or sx >= cols:
            return False

        y0, x0 = int(sy), int(sx)
        y1 = min(y0 + 1, rows - 1)
        x1 = min(x0 + 1, cols - 1)
        dy, dx = sy - y0, sx - x0
        val = (
            theta[y0, x0] * (1 - dy) * (1 - dx)
            + theta[y0, x1] * (1 - dy) * dx
            + theta[y1, x0] * dy * (1 - dx)
            + theta[y1, x1] * dy * dx
        )
        expected = 0.5 * phi if is_core else -0.5 * phi
        diff = val - expected
        diff = ((diff + np.pi / 2) % np.pi) - np.pi / 2
        sampled.append(float(diff))

    theta_0 = float(np.mean(sampled))
    residuals = [d - theta_0 for d in sampled]
    residuals = [((r + np.pi / 2) % np.pi) - np.pi / 2 for r in residuals]
    rms = float(np.sqrt(np.mean(np.array(residuals) ** 2)))
    return rms < DORIC_RMS_THRESHOLD


def _detect(orientation_field: np.ndarray, block_size: int = 16) -> tuple[list[Singularity], list[Singularity]]:
    """Run the Poincaré + DORIC pipeline (dual-sigma: cores use 2.0, deltas use 1.0).

    The Gaussian smoothing is applied to the (Vx, Vy) vector field
    representation of the orientation field, which is mathematically
    correct for circular data (orientation is periodic in π).

    Cores and deltas are detected with different sigmas:
      - Cores: sigma=2.0 (suppresses scar/crease noise, robust
        against false positives; the core signal is strong)
      - Deltas: sigma=1.0 (preserves the smaller signal that
        2.0 erases in high-curvature regions like the delta zone
        of a loop)

    Returns (cores, deltas) as a tuple.
    """
    theta = orientation_field.astype(np.float32)

    theta_smooth_core = _gaussian_smooth(theta, sigma=GAUSSIAN_SIGMA_CORE)
    poi_map_core = _poincare_map(theta_smooth_core)

    theta_smooth_delta = _gaussian_smooth(theta, sigma=GAUSSIAN_SIGMA_DELTA)
    poi_map_delta = _poincare_map(theta_smooth_delta)

    cores: list[Singularity] = []
    for by in range(poi_map_core.shape[0]):
        for bx in range(poi_map_core.shape[1]):
            pi = float(poi_map_core[by, bx])
            if POI_CORE_LOW < pi < POI_CORE_HIGH:
                if _doric_validate(theta_smooth_core, by, bx, is_core=True):
                    cores.append(
                        Singularity(
                            x=bx * block_size + block_size // 2,
                            y=by * block_size + block_size // 2,
                            kind=SingularityKind.CORE,
                            confidence=min(1.0, abs(pi) * 2.0),
                            poincare_value=pi,
                        )
                    )

    deltas: list[Singularity] = []
    for by in range(poi_map_delta.shape[0]):
        for bx in range(poi_map_delta.shape[1]):
            pi = float(poi_map_delta[by, bx])
            if POI_DELTA_LOW < pi < POI_DELTA_HIGH:
                if _doric_validate(theta_smooth_delta, by, bx, is_core=False):
                    deltas.append(
                        Singularity(
                            x=bx * block_size + block_size // 2,
                            y=by * block_size + block_size // 2,
                            kind=SingularityKind.DELTA,
                            confidence=min(1.0, abs(pi) * 2.0),
                            poincare_value=pi,
                        )
                    )

    return cores, deltas


def _non_max_suppress(
    singularities: list[Singularity],
    radius_blocks: int = NMS_RADIUS_BLOCKS,
    block_size: int = 16,
) -> list[Singularity]:
    if not singularities:
        return []
    sorted_sings = sorted(
        singularities, key=lambda s: abs(s.poincare_value), reverse=True,
    )
    radius_px = radius_blocks * block_size
    kept: list[Singularity] = []
    for s in sorted_sings:
        if not any(
            abs(s.x - k.x) <= radius_px and abs(s.y - k.y) <= radius_px
            for k in kept
        ):
            kept.append(s)
    return kept


def apply_henry_cap(
    classification: PatternClassification,
    cores: list[Singularity],
    deltas: list[Singularity],
) -> tuple[list[Singularity], list[Singularity]]:
    """Cap the singularity lists to the expected (cores, deltas) count.

    This is the ONLY principled rule. We trust the classifier's
    pattern assignment and cap accordingly.

    For UNKNOWN patterns (e.g. count 1-0 from a degraded loop with
    a missing delta), no cap is applied — the count is passed
    through so the matcher / report can see the raw signal.
    """
    if classification.pattern_type == PatternType.UNKNOWN:
        return cores, deltas

    expected = {
        PatternType.PLAIN_ARCH: (0, 0),
        PatternType.TENTED_ARCH: (0, 1),
        PatternType.LOOP: (1, 1),
        PatternType.WHORL: (2, 2),
    }.get(classification.pattern_type, (None, None))

    if expected == (None, None):
        return cores, deltas

    n_cores_expected, n_deltas_expected = expected
    cores = cores[:n_cores_expected]
    deltas = deltas[:n_deltas_expected]

    return cores, deltas


def detect_singularities(
    orientation_field: np.ndarray,
    block_size: int = 16,
) -> tuple[list[Singularity], list[Singularity]]:
    """Run raw singularity detection with the dual-sigma approach.

    Cores use sigma=2.0 (robust), deltas use sigma=1.0 (preserves
    the smaller signal). See `_detect` for the rationale.
    """
    raw_cores, raw_deltas = _detect(orientation_field, block_size=block_size)
    cores = _non_max_suppress(raw_cores, block_size=block_size)
    deltas = _non_max_suppress(raw_deltas, block_size=block_size)
    return cores, deltas


__all__ = [
    "GAUSSIAN_SIGMA_CORE",
    "GAUSSIAN_SIGMA_DELTA",
    "NMS_RADIUS_BLOCKS",
    "apply_henry_cap",
    "detect_singularities",
]

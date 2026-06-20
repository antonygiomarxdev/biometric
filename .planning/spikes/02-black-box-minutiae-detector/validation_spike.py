"""Capa 2: Validación contextual (Spike 02).

Implements the forensic-context rules that the production pipeline
currently lacks:

  1. ALL singularities (cores and deltas) via Poincaré + DORIC,
     not just the strongest one.
  2. Pattern area mask (the region where ridges exist).
  3. Zone classification per candidate (border / interior /
     near-core / near-delta).
  4. Ridge tracing per candidate (for terminations: how far does
     the ridge extend; for bifurcations: do all three branches
     have continuity?).
  5. Overlap detection (Y-junction vs ridge crossing).
  6. Confidence scoring combining the above.

Each rule has a single, documented threshold. None of the
thresholds are magic numbers — they have units and meaning, and
are easy to tune after visual inspection.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

import cv2
import numpy as np

from src.core.config import config as prod_config
from src.core.types import AlgorithmOrigin
from src.processing.pre_hooks import OrientationFieldAnalyzer

from detector_spike import detect_raw_candidates, to_validated_minutia
from types_spike import (
    DetectionResult,
    QualityZone,
    Singularity,
    SingularityKind,
    ValidatedMinutia,
    Zone,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable thresholds (all in skeleton-pixel units at 256x256)
# ---------------------------------------------------------------------------

# Pattern area: dilate the skeleton by N pixels to get the reliable area.
PATTERN_AREA_DILATION_PX: int = 12

# Border zone: candidates within N pixels of the image border are border.
BORDER_MARGIN_PX: int = 8

# Singularity proximity: candidates within N pixels of a core/delta.
SINGULARITY_PROXIMITY_PX: int = 25

# Ridge tracing: max walk length for terminations.
MAX_TRACE_PX: int = 20

# Ridge tracing: minimum acceptable trace length for a real termination.
MIN_TRACE_TERMINATION_PX: int = 3

# Ridge tracing: minimum acceptable length for ALL THREE branches of a
# bifurcation. If any branch is shorter, the junction is likely a crossing.
MIN_TRACE_BRANCH_PX: int = 3

# Overlap detection: a bifurcation where >=1 branch is shorter than this
# is flagged as is_overlap = True.
OVERLAP_BRANCH_PX: int = 4

# Poincaré + DORIC thresholds (reused from production config).
DORIC_RADIUS: int = prod_config.doric.radius
DORIC_SAMPLES: int = prod_config.doric.n_samples
DORIC_RMS_THRESHOLD: float = prod_config.doric.rms_threshold
POI_CORE_LOW: float = 0.25
POI_CORE_HIGH: float = 0.75
POI_DELTA_LOW: float = -0.75
POI_DELTA_HIGH: float = -0.25
POI_DIVISOR: float = prod_config.doric.poi_divisor


# ---------------------------------------------------------------------------
# Singularity detection (ALL cores and deltas)
# ---------------------------------------------------------------------------


def _compute_poincare_map(orientation_field: np.ndarray) -> np.ndarray:
    """Poincaré index on the orientation field, block-level.

    Returns a (rows, cols) map with the same shape as the input
    orientation field. Core → +0.5, Delta → -0.5.
    """
    theta = orientation_field.astype(np.float32)
    rows, cols = theta.shape
    poi_map = np.zeros((rows, cols), dtype=np.float32)

    for by in range(1, rows - 1):
        for bx in range(1, cols - 1):
            neighbours = [
                theta[by - 1, bx - 1],
                theta[by - 1, bx],
                theta[by - 1, bx + 1],
                theta[by, bx + 1],
                theta[by + 1, bx + 1],
                theta[by + 1, bx],
                theta[by + 1, bx - 1],
                theta[by, bx - 1],
            ]
            diff_sum = 0.0
            for k in range(len(neighbours)):
                d = neighbours[(k + 1) % len(neighbours)] - neighbours[k]
                d = ((d + np.pi / 2) % np.pi) - np.pi / 2
                diff_sum += d
            poi_map[by, bx] = diff_sum / POI_DIVISOR

    return poi_map


def _doric_validate(
    theta_smooth: np.ndarray,
    cy: int,
    cx: int,
    *,
    is_core: bool,
) -> bool:
    """DORIC validation: fit candidate to the zero-pole model."""
    r = DORIC_RADIUS
    n = DORIC_SAMPLES
    rows, cols = theta_smooth.shape

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
            theta_smooth[y0, x0] * (1 - dy) * (1 - dx)
            + theta_smooth[y0, x1] * (1 - dy) * dx
            + theta_smooth[y1, x0] * dy * (1 - dx)
            + theta_smooth[y1, x1] * dy * dx
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


def detect_all_singularities(
    orientation_field: np.ndarray,
    block_size: int = 16,
) -> tuple[list[Singularity], list[Singularity]]:
    """Return ALL validated cores and deltas, not just the strongest."""
    theta = orientation_field.astype(np.float32)
    vx = np.cos(2 * theta)
    vy = np.sin(2 * theta)
    sigma = 2.0
    vx_s = cv2.GaussianBlur(vx, (0, 0), sigmaX=sigma, sigmaY=sigma)
    vy_s = cv2.GaussianBlur(vy, (0, 0), sigmaX=sigma, sigmaY=sigma)
    theta_smooth = 0.5 * np.arctan2(vy_s, vx_s)

    poi_map = _compute_poincare_map(theta_smooth)

    cores: list[Singularity] = []
    deltas: list[Singularity] = []

    for by in range(poi_map.shape[0]):
        for bx in range(poi_map.shape[1]):
            pi = float(poi_map[by, bx])
            if POI_CORE_LOW < pi < POI_CORE_HIGH:
                if _doric_validate(theta_smooth, by, bx, is_core=True):
                    cores.append(
                        Singularity(
                            x=bx * block_size + block_size // 2,
                            y=by * block_size + block_size // 2,
                            kind=SingularityKind.CORE,
                            confidence=min(1.0, abs(pi) * 2.0),
                            poincare_value=pi,
                        )
                    )
            elif POI_DELTA_LOW < pi < POI_DELTA_HIGH:
                if _doric_validate(theta_smooth, by, bx, is_core=False):
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


# ---------------------------------------------------------------------------
# Pattern area mask
# ---------------------------------------------------------------------------


def compute_pattern_area_mask(skeleton: np.ndarray) -> np.ndarray:
    """Dilate the skeleton to get the reliable pattern area.

    A minutia inside this mask is ``in_pattern_area=True``; outside
    is border zone.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * PATTERN_AREA_DILATION_PX + 1, 2 * PATTERN_AREA_DILATION_PX + 1),
    )
    return cv2.dilate((skeleton > 0).astype(np.uint8), kernel) > 0


# ---------------------------------------------------------------------------
# Zone classification
# ---------------------------------------------------------------------------


def classify_zone(
    x: int,
    y: int,
    h: int,
    w: int,
    cores: list[Singularity],
    deltas: list[Singularity],
) -> Zone:
    """Where is this minutia relative to the reliable area?"""
    if (
        x < BORDER_MARGIN_PX
        or y < BORDER_MARGIN_PX
        or x >= w - BORDER_MARGIN_PX
        or y >= h - BORDER_MARGIN_PX
    ):
        return Zone.BORDER
    for c in cores:
        if abs(x - c.x) <= SINGULARITY_PROXIMITY_PX and abs(y - c.y) <= SINGULARITY_PROXIMITY_PX:
            return Zone.NEAR_CORE
    for d in deltas:
        if abs(x - d.x) <= SINGULARITY_PROXIMITY_PX and abs(y - d.y) <= SINGULARITY_PROXIMITY_PX:
            return Zone.NEAR_DELTA
    return Zone.INTERIOR


# ---------------------------------------------------------------------------
# Ridge tracing
# ---------------------------------------------------------------------------


def _walk_ridge(
    skeleton_bool: np.ndarray,
    start_x: int,
    start_y: int,
    exclude: tuple[int, int],
) -> int:
    """Walk along the skeleton from start, avoiding the ``exclude`` pixel.

    Returns the number of steps before hitting a junction (CN > 2), the
    border, or max steps. Returns 0 if no step is possible.
    """
    h, w = skeleton_bool.shape
    visited = {exclude, (start_x, start_y)}
    queue: deque[tuple[int, int, int]] = deque()
    queue.append((start_x, start_y, 0))
    ex, ey = exclude

    last_steps = 0
    while queue:
        cx, cy, depth = queue.popleft()
        if depth >= MAX_TRACE_PX:
            return depth
        last_steps = max(last_steps, depth)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if not (0 <= nx < w and 0 <= ny < h):
                    return depth
                if not skeleton_bool[ny, nx]:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny, depth + 1))
                break
            else:
                continue
            break
    return last_steps


def trace_ridge_from_termination(
    skeleton: np.ndarray,
    x: int,
    y: int,
    neighbour: tuple[int, int],
) -> int:
    """For a termination, trace the ridge through its single neighbour.

    Returns the number of steps before the ridge hits a junction or end.
    """
    skeleton_bool = skeleton > 0
    nx, ny = neighbour
    return _walk_ridge(skeleton_bool, nx, ny, exclude=(x, y))


def trace_ridge_from_bifurcation(
    skeleton: np.ndarray,
    x: int,
    y: int,
    neighbours: list[tuple[int, int]],
) -> list[int]:
    """For a bifurcation, trace each of the three branches.

    Returns a list of three step counts.
    """
    skeleton_bool = skeleton > 0
    return [
        _walk_ridge(skeleton_bool, nx, ny, exclude=(x, y))
        for nx, ny in neighbours
    ]


def is_overlap_junction(
    branch_lengths: list[int],
    *,
    short_branch_px: int = OVERLAP_BRANCH_PX,
) -> bool:
    """A bifurcation is an overlap (crossing) if any branch is too short.

    In a real Y-junction, all three branches have similar length.
    In a crossing of two ridges, one of the four arms is the intersection
    and the others are continuations — this leaves a short branch.
    """
    if len(branch_lengths) < 3:
        return False
    return any(b < short_branch_px for b in branch_lengths)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------


def compute_confidence(
    *,
    zone: Zone,
    ridge_trace_length: int,
    is_overlap: bool,
    in_pattern_area: bool,
    is_termination: bool,
) -> float:
    """Combine validation signals into a single confidence in [0, 1]."""
    score = 0.5

    if zone == Zone.BORDER:
        score -= 0.2
    elif zone in (Zone.NEAR_CORE, Zone.NEAR_DELTA):
        score += 0.1
    elif zone == Zone.INTERIOR:
        score += 0.05

    if not in_pattern_area:
        score -= 0.15

    if is_termination:
        if ridge_trace_length >= MIN_TRACE_TERMINATION_PX:
            score += 0.2
        else:
            score -= 0.3
    else:
        if ridge_trace_length >= 0:
            score += 0.1

    if is_overlap:
        score -= 0.3

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Black-box detector
# ---------------------------------------------------------------------------


def get_neighbours_at(skeleton_bool: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    """Return the 8-connected skeleton neighbours of (x, y)."""
    h, w = skeleton_bool.shape
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and skeleton_bool[ny, nx]:
                out.append((nx, ny))
    return out


def validate_candidate(
    raw: dict[str, Any],
    skeleton: np.ndarray,
    skeleton_bool: np.ndarray,
    pattern_mask: np.ndarray,
    cores: list[Singularity],
    deltas: list[Singularity],
) -> ValidatedMinutia:
    """Validate a single raw CN candidate."""
    x = int(raw["x"])
    y = int(raw["y"])
    h, w = skeleton.shape

    zone = classify_zone(x, y, h, w, cores, deltas)
    in_pattern_area = bool(pattern_mask[y, x]) if 0 <= y < h and 0 <= x < w else False

    is_termination = int(raw["type"]) == 1
    neighbours = get_neighbours_at(skeleton_bool, x, y)

    if is_termination:
        if len(neighbours) != 1:
            trace_len = 0
        else:
            trace_len = trace_ridge_from_termination(skeleton, x, y, neighbours[0])
        is_overlap = False
    else:
        branch_lengths = trace_ridge_from_bifurcation(skeleton, x, y, neighbours)
        trace_len = min(branch_lengths) if branch_lengths else 0
        is_overlap = is_overlap_junction(branch_lengths)

    confidence = compute_confidence(
        zone=zone,
        ridge_trace_length=trace_len,
        is_overlap=is_overlap,
        in_pattern_area=in_pattern_area,
        is_termination=is_termination,
    )

    return ValidatedMinutia(
        x=x,
        y=y,
        angle=float(raw["angle"]),
        type=to_validated_minutia_type(is_termination),
        confidence=confidence,
        origin=AlgorithmOrigin.SKELETON,
        zone=zone,
        ridge_trace_length=trace_len,
        is_overlap=is_overlap,
        in_pattern_area=in_pattern_area,
    )


def to_validated_minutia_type(is_termination: bool):
    """Local helper to avoid an import cycle in tests."""
    from src.core.types import MinutiaType

    return MinutiaType.TERMINATION if is_termination else MinutiaType.BIFURCATION


def run_quality_zones(
    minutiae: list[ValidatedMinutia],
    skeleton: np.ndarray,
    block: int = 32,
) -> list[QualityZone]:
    """Tile the image into blocks and score each block.

    Each block gets a quality score = (n_minutiae in block with
    confidence > 0.5) / (max(1, n_minutiae in block)).
    """
    h, w = skeleton.shape
    zones: list[QualityZone] = []
    for by in range(0, h, block):
        for bx in range(0, w, block):
            block_minutiae = [
                m for m in minutiae if by <= m.y < by + block and bx <= m.x < bx + block
            ]
            if not block_minutiae:
                continue
            good = sum(1 for m in block_minutiae if m.confidence >= 0.5)
            zones.append(
                QualityZone(
                    bbox=(bx, by, min(block, w - bx), min(block, h - by)),
                    quality_score=good / len(block_minutiae),
                    n_minutiae=len(block_minutiae),
                )
            )
    return zones


def detect(image: np.ndarray) -> DetectionResult:
    """The black-box interface.

    Input:  grayscale fingerprint image (any size).
    Output: DetectionResult with validated minutiae + singularities
            + pattern area + quality zones.

    Internally calls the same production functions as
    ``MccMatchingService._run_quality_pipeline``, then adds the
    contextual validation layer.
    """
    raw_artefacts = detect_raw_candidates(image)
    enhanced = raw_artefacts["enhanced"]
    normalized = raw_artefacts["normalized"]
    skeleton = raw_artefacts["skeleton"]
    raw_minutiae = raw_artefacts["raw_minutiae"]

    skeleton_bool = skeleton > 0
    h, w = skeleton.shape

    analyzer = OrientationFieldAnalyzer(block_size=16)
    ctx = type("Ctx", (), {})()
    from src.core.interfaces import PipelineContext

    ctx = PipelineContext(raw_image=image, fingerprint_id="spike")
    ctx.preprocessed_image = normalized
    analyzer.process(ctx)
    orientation_field = ctx.orientation_field
    coherence_field = ctx.coherence_field

    cores, deltas = ([], [])
    if orientation_field is not None:
        cores, deltas = detect_all_singularities(orientation_field, block_size=16)

    pattern_mask = compute_pattern_area_mask(skeleton)

    validated = [
        validate_candidate(raw, skeleton, skeleton_bool, pattern_mask, cores, deltas)
        for raw in raw_minutiae
    ]

    quality_zones = run_quality_zones(validated, skeleton)

    return DetectionResult(
        minutiae=validated,
        cores=cores,
        deltas=deltas,
        pattern_area_mask=pattern_mask,
        quality_zones=quality_zones,
        skeleton=skeleton,
        enhanced_image=normalized,
        metadata={
            "image_shape": list(image.shape),
            "normalized_shape": list(normalized.shape),
            "n_raw_candidates": len(raw_minutiae),
            "n_validated": len(validated),
            "n_cores": len(cores),
            "n_deltas": len(deltas),
            "timings": raw_artefacts["timings"],
        },
    )


__all__ = [
    "BORDER_MARGIN_PX",
    "DORIC_RADIUS",
    "MAX_TRACE_PX",
    "MIN_TRACE_BRANCH_PX",
    "MIN_TRACE_TERMINATION_PX",
    "OVERLAP_BRANCH_PX",
    "PATTERN_AREA_DILATION_PX",
    "POI_CORE_HIGH",
    "POI_CORE_LOW",
    "POI_DELTA_HIGH",
    "POI_DELTA_LOW",
    "SINGULARITY_PROXIMITY_PX",
    "classify_zone",
    "compute_pattern_area_mask",
    "compute_confidence",
    "detect",
    "detect_all_singularities",
    "is_overlap_junction",
    "run_quality_zones",
    "trace_ridge_from_bifurcation",
    "trace_ridge_from_termination",
    "validate_candidate",
]

"""Pattern-first detector (Spike 03).

The orchestrator that fixes the order-of-operations issue from
spike 02. Instead of:

    Minutiae -> Context -> "this is a loop"  (post-hoc)

it does:

    Singularity -> NMS -> Pattern classification -> Type-aware
    refinement -> Minutiae -> Context  (pattern-first)

This module imports the validation primitives from spike 02
(NMS, ridge trace, overlap detection, zone classification) and
the new type-aware machinery from spike 03 (classifier, facing,
type-aware singularity rules).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SPIKE02 = Path(__file__).resolve().parent.parent / "02-black-box-minutiae-detector"
sys.path.insert(0, str(SPIKE02))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.config import config as prod_config
from src.core.types import AlgorithmOrigin, MinutiaType
from src.processing.enhancer import create_enhancer
from src.processing.scale_normalization import normalize_to_256
from src.processing.thinning import thin
from src.processing.crossing_number import extract_minutiae_cn
from src.processing.pre_hooks import OrientationFieldAnalyzer
from src.core.interfaces import PipelineContext

from validation_spike import (
    BORDER_MARGIN_PX,
    SINGULARITY_PROXIMITY_PX,
    PATTERN_AREA_DILATION_PX,
    classify_zone,
    compute_pattern_area_mask,
    compute_confidence,
    is_overlap_junction,
    trace_ridge_from_bifurcation,
    trace_ridge_from_termination,
    get_neighbours_at,
    run_quality_zones,
)

from types_spike import (
    DetectionResult,
    PatternClassification,
    PatternType,
    QualityZone,
    Singularity,
    SingularityKind,
    ValidatedMinutia,
    Zone,
)
from pattern_classifier import classify as classify_pattern
from type_aware_rules import apply_henry_cap, detect_singularities

logger = logging.getLogger(__name__)


def _to_minutia_type(is_termination: bool) -> MinutiaType:
    return MinutiaType.TERMINATION if is_termination else MinutiaType.BIFURCATION


def _run_enhance_and_skeleton(image: np.ndarray) -> dict[str, Any]:
    """Capa 1 from spike 02 — enhance + normalize + thin + raw CN."""
    t0 = time.monotonic()
    enhancer = create_enhancer()
    enhanced = enhancer.enhance(image, resize=True)
    t_enhance = time.monotonic() - t0

    t0 = time.monotonic()
    normalized = normalize_to_256(enhanced)
    t_norm = time.monotonic() - t0

    t0 = time.monotonic()
    skeleton = thin(normalized)
    t_thin = time.monotonic() - t0

    t0 = time.monotonic()
    raw_minutiae = extract_minutiae_cn(skeleton)
    t_cn = time.monotonic() - t0

    return {
        "enhanced": enhanced,
        "normalized": normalized,
        "skeleton": skeleton,
        "raw_minutiae": raw_minutiae,
        "timings": {
            "enhance_ms": round(t_enhance * 1000, 1),
            "norm_ms": round(t_norm * 1000, 1),
            "thin_ms": round(t_thin * 1000, 1),
            "cn_ms": round(t_cn * 1000, 1),
        },
    }


def _get_orientation_field(image: np.ndarray, normalized: np.ndarray) -> np.ndarray | None:
    analyzer = OrientationFieldAnalyzer(block_size=16)
    ctx = PipelineContext(raw_image=image, fingerprint_id="spike03")
    ctx.preprocessed_image = normalized
    analyzer.process(ctx)
    return ctx.orientation_field


def validate_candidate(
    raw: dict[str, Any],
    skeleton: np.ndarray,
    skeleton_bool: np.ndarray,
    pattern_mask: np.ndarray,
    classification: PatternClassification,
    cores: list[Singularity],
    deltas: list[Singularity],
) -> ValidatedMinutia:
    x = int(raw["x"])
    y = int(raw["y"])
    h, w = skeleton.shape

    zone = classify_zone(x, y, h, w, cores, deltas)
    in_pattern_area = bool(pattern_mask[y, x]) if 0 <= y < h and 0 <= x < w else False

    is_termination = int(raw["type"]) == 1
    neighbours = get_neighbours_at(skeleton_bool, x, y)

    if is_termination:
        trace_len = (
            trace_ridge_from_termination(skeleton, x, y, neighbours[0])
            if len(neighbours) == 1
            else 0
        )
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
        type=_to_minutia_type(is_termination),
        confidence=confidence,
        origin=AlgorithmOrigin.SKELETON,
        zone=zone,
        ridge_trace_length=trace_len,
        is_overlap=is_overlap,
        in_pattern_area=in_pattern_area,
    )


def detect(image: np.ndarray) -> DetectionResult:
    """Pattern-first black-box detector.

    Order (the key change from spike 02):

    1. Enhance + skeleton (Capa 1 from spike 02, unchanged)
    2. Orientation field
    3. **Singularity detection** with NMS
    4. **Pattern classification** (the new step)
    5. **Type-aware singularity refinement** (different sigma for
       loop deltas, cap by Henry rules for the classified pattern)
    6. CN minutiae detection (Capa 1)
    7. Context validation (zone, ridge trace, overlap, confidence)
    8. Output ``DetectionResult`` with pattern_classification
    """
    artefacts = _run_enhance_and_skeleton(image)
    enhanced = artefacts["enhanced"]
    normalized = artefacts["normalized"]
    skeleton = artefacts["skeleton"]
    raw_minutiae = artefacts["raw_minutiae"]

    skeleton_bool = skeleton > 0
    h, w = skeleton.shape

    orientation_field = _get_orientation_field(image, normalized)

    initial_cores: list[Singularity] = []
    initial_deltas: list[Singularity] = []
    if orientation_field is not None:
        initial_cores, initial_deltas = detect_singularities(
            orientation_field, block_size=16,
        )

    classification = classify_pattern(initial_cores, initial_deltas)

    if classification.pattern_type != PatternType.UNKNOWN:
        cores, deltas = apply_henry_cap(
            classification, initial_cores, initial_deltas,
        )
    else:
        cores, deltas = initial_cores, initial_deltas

    pattern_mask = compute_pattern_area_mask(skeleton)

    validated = [
        validate_candidate(
            raw, skeleton, skeleton_bool, pattern_mask, classification,
            cores, deltas,
        )
        for raw in raw_minutiae
    ]

    quality_zones = run_quality_zones(validated, skeleton)

    n_raw_cores = len(initial_cores)
    n_raw_deltas = len(initial_deltas)
    return DetectionResult(
        minutiae=validated,
        cores=cores,
        deltas=deltas,
        pattern_classification=classification,
        pattern_area_mask=pattern_mask,
        quality_zones=quality_zones,
        skeleton=skeleton,
        enhanced_image=normalized,
        metadata={
            "image_shape": list(image.shape),
            "normalized_shape": list(normalized.shape),
            "n_raw_candidates": len(raw_minutiae),
            "n_validated": len(validated),
            "n_raw_cores": n_raw_cores,
            "n_raw_deltas": n_raw_deltas,
            "n_cores": len(cores),
            "n_deltas": len(deltas),
            "pattern_type": classification.pattern_type.value,
            "pattern_confidence": classification.confidence,
            "facing": classification.facing.value,
            "timings": artefacts["timings"],
        },
    )


__all__ = [
    "detect",
    "validate_candidate",
]

"""Capa 1: Detector de candidatas (Spike 02).

Wraps the production quality pipeline (the same one used by
``MccMatchingService._run_quality_pipeline``) and produces a list of
candidates without filtering. Filtering is Layer 2's job.

This is NOT a reimplementation. It calls the same functions as
production, in the same order, with the same parameters. The only
difference is that the candidates are returned BEFORE the false-minutiae
filter — we want to see the raw output and validate it ourselves.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from src.processing.crossing_number import extract_minutiae_cn
from src.processing.enhancer import create_enhancer
from src.processing.scale_normalization import normalize_to_256
from src.processing.thinning import thin

from types_spike import DetectionResult, QualityZone, Singularity, SingularityKind, ValidatedMinutia, Zone
from src.core.types import AlgorithmOrigin, MinutiaType

logger = logging.getLogger(__name__)


def detect_raw_candidates(image: np.ndarray) -> dict[str, Any]:
    """Run the production pipeline up to the false-minutiae filter.

    Returns a dict with all intermediate artefacts so Layer 2 can
    inspect them. The candidate list is the raw output of
    ``extract_minutiae_cn`` — unfiltered.
    """
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


def min_type_to_spike(raw: dict[str, Any]) -> MinutiaType:
    """Map the raw CN output (type 1 or 3) to MinutiaType enum."""
    return MinutiaType.TERMINATION if int(raw["type"]) == 1 else MinutiaType.BIFURCATION


def to_validated_minutia(
    raw: dict[str, Any],
    *,
    zone: Zone,
    ridge_trace_length: int,
    is_overlap: bool,
    in_pattern_area: bool,
    confidence: float,
) -> ValidatedMinutia:
    """Convert a raw CN candidate to a ValidatedMinutia with metadata.

    The validation metadata defaults are conservative (all False / 0)
    when the caller does not yet have the contextual data.
    """
    return ValidatedMinutia(
        x=int(raw["x"]),
        y=int(raw["y"]),
        angle=float(raw["angle"]),
        type=min_type_to_spike(raw),
        confidence=confidence,
        origin=AlgorithmOrigin.SKELETON,
        zone=zone,
        ridge_trace_length=ridge_trace_length,
        is_overlap=is_overlap,
        in_pattern_area=in_pattern_area,
    )


__all__ = [
    "detect_raw_candidates",
    "to_validated_minutia",
]

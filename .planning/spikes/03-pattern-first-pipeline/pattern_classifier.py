"""Pattern classifier for the pattern-first pipeline (Spike 03).

A **principled** classifier based ONLY on the Henry rule pair:

    (0, 0) -> plain_arch
    (0, 1) -> tented_arch
    (1, 1) -> loop
    (2, 2) -> whorl
    (1, 0) -> loop_missing_delta (degraded)
    (0, 2+) -> ambiguous (over-detection or arch with noise)
    other  -> unknown

Why count-only (no curvature, no other heuristics)?

The Henry rule pair is the only signal that is **principled** — it
comes from forensic theory, not from data tuning. Everything else
(curvature thresholds, sigma tuning per type, etc.) is a heuristic
that overfits to the dataset used for tuning.

Per the LESSONS_LEARNED:
    "Do not add more heuristic patches without benchmark data."
    "Custom score formulas based on intuition." -> anti-pattern.

So this module does the minimum: takes the count, returns the pattern.
No thresholds to tune, no parameters to fit.
"""
from __future__ import annotations

from types_spike import (
    FacingDirection,
    PatternClassification,
    PatternType,
    Singularity,
)


def classify(
    cores: list[Singularity],
    deltas: list[Singularity],
) -> PatternClassification:
    """Classify the fingerprint by the Henry rule pair.

    Args:
        cores: Post-NMS list of cores.
        deltas: Post-NMS list of deltas.

    Returns:
        ``PatternClassification`` with the type determined by the
        count pair and confidence set by the cleanness of the match.
    """
    n_cores = len(cores)
    n_deltas = len(deltas)
    count_pair = (n_cores, n_deltas)

    if count_pair == (0, 0):
        pattern_type = PatternType.PLAIN_ARCH
        confidence = 0.9
    elif count_pair == (0, 1):
        pattern_type = PatternType.TENTED_ARCH
        confidence = 0.9
    elif count_pair == (1, 1):
        pattern_type = PatternType.LOOP
        confidence = 0.9
    elif count_pair == (2, 2):
        pattern_type = PatternType.WHORL
        confidence = 0.9
    else:
        pattern_type = PatternType.UNKNOWN
        confidence = 0.0

    facing = FacingDirection.UNKNOWN
    if (
        pattern_type == PatternType.LOOP
        and n_cores >= 1
        and n_deltas >= 1
    ):
        from facing_direction import compute_facing

        facing = compute_facing(cores[0], deltas[0])

    return PatternClassification(
        pattern_type=pattern_type,
        confidence=confidence,
        mean_curvature=0.0,
        singularity_count=count_pair,
        facing=facing,
        metadata={"classifier": "henry_count_only"},
    )


__all__ = ["classify"]

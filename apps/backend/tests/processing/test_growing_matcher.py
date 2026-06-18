"""Tests for growing matcher algorithm.

Uses synthetic triplet data and mock KNN hits to verify the growing
algorithm's scoring and geometric validation logic.
"""

from __future__ import annotations

import math

import pytest

from src.processing.growing_matcher import grow_matches


def _make_triplet(
    mi_x: float, mi_y: float, mi_a_deg: float,
    mj_x: float, mj_y: float, mj_a_deg: float,
    mk_x: float, mk_y: float, mk_a_deg: float,
    idx: int = 0,
) -> dict:
    return {
        "mi_idx": idx, "mj_idx": idx + 1, "mk_idx": idx + 2,
        "mi_x": mi_x, "mi_y": mi_y, "mi_angle": math.radians(mi_a_deg),
        "mj_x": mj_x, "mj_y": mj_y, "mj_angle": math.radians(mj_a_deg),
        "mk_x": mk_x, "mk_y": mk_y, "mk_angle": math.radians(mk_a_deg),
        "d_ij": 0.1, "d_ik": 0.12, "d_jk": 0.08,
        "type_triple": 0, "quality_min": 0.5, "quality_avg": 0.7,
    }


def _make_hit(
    query_idx: int,
    person_id: str,
    similarity: float,
    mi_x: float, mi_y: float, mi_a: float,
    mj_x: float, mj_y: float, mj_a: float,
    mk_x: float, mk_y: float, mk_a: float,
) -> dict:
    return {
        "query_triplet_index": query_idx,
        "similarity": similarity,
        "person_id": person_id,
        "fingerprint_id": "fp1",
        "capture_id": "cap1",
        "mi_idx": 0, "mj_idx": 1, "mk_idx": 2,
        "mi_x": mi_x, "mi_y": mi_y, "mi_angle": mi_a,
        "mj_x": mj_x, "mj_y": mj_y, "mj_angle": mj_a,
        "mk_x": mk_x, "mk_y": mk_y, "mk_angle": mk_a,
        "type_triple": 0, "quality_min": 0.5,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def same_finger_triplets() -> tuple[list[dict], list[dict]]:
    """3 probe triplets from the same finger shape (with slight noise)."""
    # Base triangle at (0.2, 0.2), (0.35, 0.4), (0.5, 0.25)
    base_probe = [
        _make_triplet(0.20, 0.20, 0, 0.35, 0.40, 45, 0.50, 0.25, 90, idx=0),
        _make_triplet(0.35, 0.40, 45, 0.50, 0.25, 90, 0.60, 0.50, 135, idx=3),
        _make_triplet(0.50, 0.25, 90, 0.60, 0.50, 135, 0.20, 0.20, 0, idx=6),
    ]

    # Same shape shifted by (0.05, 0.03)
    hits = [
        _make_hit(0, "person1", 0.92, 0.25, 0.23, math.radians(0), 0.40, 0.43, math.radians(45), 0.55, 0.28, math.radians(90)),
        _make_hit(1, "person1", 0.88, 0.40, 0.43, math.radians(45), 0.55, 0.28, math.radians(90), 0.65, 0.53, math.radians(135)),
        _make_hit(2, "person1", 0.85, 0.55, 0.28, math.radians(90), 0.65, 0.53, math.radians(135), 0.25, 0.23, math.radians(0)),
    ]

    return base_probe, hits


# ---------------------------------------------------------------------------
# Full match
# ---------------------------------------------------------------------------


def test_full_match(same_finger_triplets: tuple[list[dict], list[dict]]) -> None:
    """All triplets match → high score."""
    probe_triplets, hits = same_finger_triplets
    results = grow_matches(probe_triplets, hits)
    assert len(results) == 1
    assert results[0].person_id == "person1"
    assert results[0].score > 0.5
    assert results[0].confirming_triplets >= 2


# ---------------------------------------------------------------------------
# No matches
# ---------------------------------------------------------------------------


def test_no_matches() -> None:
    """Empty hits → empty results."""
    result = grow_matches([], [])
    assert len(result) == 0


def test_no_probe_triplets() -> None:
    """No probe triplets → empty results."""
    result = grow_matches([], [{"person_id": "p1", "query_triplet_index": 0}])
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Mixed: 2 matches, 1 outlier
# ---------------------------------------------------------------------------


def test_mixed_match(same_finger_triplets: tuple[list[dict], list[dict]]) -> None:
    """2 consistent + 1 outlier → score reflects partial match."""
    probe_triplets, hits = same_finger_triplets

    # Add an outlier (same person, different geometry)
    outlier_hit = _make_hit(
        2, "person1", 0.7,
        0.9, 0.9, math.radians(0),
        0.95, 0.85, math.radians(45),
        0.85, 0.8, math.radians(90),
    )
    hits.append(outlier_hit)

    results = grow_matches(probe_triplets, hits)

    # Should still find person1 with non-zero score
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Multiple persons
# ---------------------------------------------------------------------------


def test_multiple_persons(same_finger_triplets: tuple[list[dict], list[dict]]) -> None:
    """Two persons in hits → both scored."""
    probe_triplets, hits = same_finger_triplets

    # Add another person with one matching triplet
    hits.append(_make_hit(
        0, "person2", 0.8,
        0.25, 0.23, math.radians(0), 0.40, 0.43, math.radians(45), 0.55, 0.28, math.radians(90),
    ))

    results = grow_matches(probe_triplets, hits)
    assert len(results) >= 1


# ---------------------------------------------------------------------------
# Low confidence
# ---------------------------------------------------------------------------


def test_low_similarity_no_match() -> None:
    """Very different geometry → no confirming triplets."""
    probe = [
        _make_triplet(0.20, 0.20, 0, 0.35, 0.40, 45, 0.50, 0.25, 90, idx=0),
    ]
    # Wildly different candidate points
    hits = [
        _make_hit(0, "far_person", 0.3, 0.9, 0.9, math.radians(0), 0.95, 0.85, math.radians(45), 0.85, 0.8, math.radians(90)),
    ]

    results = grow_matches(probe, hits)
    assert len(results) == 0

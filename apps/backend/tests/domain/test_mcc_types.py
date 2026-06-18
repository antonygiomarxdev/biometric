"""Tests for MCC domain types (Phase 21)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types import MccCylinder, MccCylinderHit, MccPersonHit


def test_mcc_cylinder_cosine_similarity_identical() -> None:
    arr = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
    c = MccCylinder(values=arr)
    assert c.cosine_similarity(c) == pytest.approx(1.0, abs=1e-5)


def test_mcc_cylinder_cosine_similarity_orthogonal() -> None:
    a = MccCylinder(values=np.array([[[1.0, 0.0]]], dtype=np.float32))
    b = MccCylinder(values=np.array([[[0.0, 1.0]]], dtype=np.float32))
    assert a.cosine_similarity(b) == pytest.approx(0.0, abs=1e-5)


def test_mcc_cylinder_hit_is_frozen() -> None:
    hit = MccCylinderHit(
        person_id="p1",
        fingerprint_id="f1",
        capture_id="c1",
        similarity=0.9,
    )
    with pytest.raises(AttributeError):
        hit.person_id = "p2"  # type: ignore[misc]


def test_mcc_person_hit_score_sums() -> None:
    hit = MccPersonHit(person_id="p1", total_score=2.5, hits=5)
    assert hit.total_score == 2.5
    assert hit.hits == 5


def test_match_trace_entry_pydantic() -> None:
    """MatchTraceEntry and MinutiaSummary are constructible + frozen (Phase 23)."""
    from src.core.types import MatchTraceEntry, MinutiaSummary

    # MinutiaSummary construction
    m = MinutiaSummary(x=10, y=20, angle=0.5, type=1)
    assert m.x == 10
    assert m.y == 20
    assert m.angle == 0.5
    assert m.type == 1

    # MatchTraceEntry construction with all 10 fields
    entry = MatchTraceEntry(
        probe_cylinder_index=2,
        probe_x=10,
        probe_y=20,
        probe_angle=0.5,
        candidate_capture_id="cap-1",
        candidate_fingerprint_id="fp-1",
        candidate_x=100,
        candidate_y=200,
        candidate_angle=1.2,
        similarity=0.85,
    )
    assert entry.probe_cylinder_index == 2
    assert entry.similarity == 0.85

    # Frozen: mutation must raise
    with pytest.raises(Exception):
        entry.probe_x = 999  # type: ignore[misc]
    with pytest.raises(Exception):
        m.x = 999  # type: ignore[misc]

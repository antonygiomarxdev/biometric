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
    with pytest.raises(Exception):
        hit.person_id = "p2"  # type: ignore[misc]


def test_mcc_person_hit_score_sums() -> None:
    hit = MccPersonHit(person_id="p1", total_score=2.5, hits=5)
    assert hit.total_score == 2.5
    assert hit.hits == 5

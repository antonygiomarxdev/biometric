"""Tests for OF similarity module (Phase 26, Plan 26-01, T1).

Uses synthetic OF arrays so tests are fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.processing.of_similarity import (
    OFSimilarity,
    OFSimilarityError,
    of_pseudo_core,
)


@pytest.fixture
def identity_of() -> OFSimilarity:
    """Uniform orientation field (all 0 rad)."""
    ori = np.zeros((4, 4), dtype=np.float32)
    coh = np.ones((4, 4), dtype=np.float32) * 0.8
    return OFSimilarity(ori, coh)


@pytest.fixture
def perpendicular_of() -> OFSimilarity:
    """Uniform orientation field at 90° (π/2 rad)."""
    ori = np.ones((4, 4), dtype=np.float32) * (np.pi / 2)
    coh = np.ones((4, 4), dtype=np.float32) * 0.8
    return OFSimilarity(ori, coh)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_score_zero(identity_of: OFSimilarity) -> None:
    score = identity_of.compare(identity_of)
    assert score == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Perpendicular
# ---------------------------------------------------------------------------


def test_perpendicular_score_high(
    identity_of: OFSimilarity,
    perpendicular_of: OFSimilarity,
) -> None:
    score = identity_of.compare(perpendicular_of)
    assert score == pytest.approx(2.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Masked blocks ignored
# ---------------------------------------------------------------------------


def test_masked_blocks_ignored(identity_of: OFSimilarity) -> None:
    other_ori = np.ones((4, 4), dtype=np.float32) * np.pi
    other_coh = np.zeros((4, 4), dtype=np.float32)
    score = identity_of.compare_raw(other_ori, other_coh)
    assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Threshold default
# ---------------------------------------------------------------------------


def test_threshold_default() -> None:
    from src.processing.of_similarity import OF_SIMILARITY_THRESHOLD
    assert OF_SIMILARITY_THRESHOLD == pytest.approx(0.50, abs=1e-6)


# ---------------------------------------------------------------------------
# Build from image
# ---------------------------------------------------------------------------


def test_build_from_image() -> None:
    img = np.ones((100, 100), dtype=np.uint8) * 200
    of = OFSimilarity.build(img, block_size=16)
    assert of.shape == (16, 16)
    assert of.block_size == 16


# ---------------------------------------------------------------------------
# Pseudo core
# ---------------------------------------------------------------------------


def test_pseudo_core_centroid() -> None:
    coh = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8, 0.8, 0.0],
        [0.0, 0.8, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float64)
    row, col = of_pseudo_core(coh)
    assert (row, col) == (2, 2)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_shape_raises() -> None:
    ori = np.zeros((4, 4), dtype=np.float32)
    coh = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(OFSimilarityError):
        OFSimilarity(ori, coh)


def test_pseudo_core_uniform_weight() -> None:
    coh = np.ones((5, 5), dtype=np.float64)
    row, col = of_pseudo_core(coh)
    assert (row, col) == (2, 2)

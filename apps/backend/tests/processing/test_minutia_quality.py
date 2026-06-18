"""Tests for minutia quality scoring.

Uses synthetic skeleton images to verify score components.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.processing.minutia_quality import score_minutia


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skeleton_256() -> np.ndarray:
    """256×256 binary skeleton (all zeros)."""
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def shape_256() -> tuple[int, int]:
    return (256, 256)


def _make_skeleton_with_branch(
    cx: int, cy: int, branches: int, size: int = 256,
) -> np.ndarray:
    """Create a skeleton with *branches* foreground arms radiating from (cx, cy).

    Each arm is 1 pixel wide and 3 pixels long.
    """
    skel = np.zeros((size, size), dtype=np.uint8)
    skel[cy, cx] = 1

    angles = [2 * np.pi * i / branches for i in range(branches)]
    for angle in angles:
        for r in range(1, 4):
            nx = int(round(cx + r * np.cos(angle)))
            ny = int(round(cy + r * np.sin(angle)))
            if 0 <= nx < size and 0 <= ny < size:
                skel[ny, nx] = 1

    return skel


# ---------------------------------------------------------------------------
# Type scoring
# ---------------------------------------------------------------------------


def test_bifurcation_type_score() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    m = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # Type contributes 0.7 * 0.4 = 0.28, position ~1.0 * 0.3 = 0.3,
    # support = 0 (skeleton bg) → total ~0.58
    assert score > 0.5
    assert score < 0.8


def test_termination_type_score() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    m = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 1}
    score = score_minutia(m, skel, (256, 256))
    # Type contributes 0.4 * 0.4 = 0.16, position ~1.0 * 0.3 = 0.3,
    # support = 0 → total ~0.46
    assert score > 0.35
    assert score < 0.65


def test_unknown_type_score() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    m = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 2}
    score = score_minutia(m, skel, (256, 256))
    # Type = 0.0 * 0.4 = 0.0, position ~1.0 * 0.3 = 0.3, support = 0 → 0.3
    assert score == pytest.approx(0.3, abs=0.01)


def test_bifurcation_higher_than_termination() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    m_bif = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 3}
    m_term = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 1}
    s_bif = score_minutia(m_bif, skel, (256, 256))
    s_term = score_minutia(m_term, skel, (256, 256))
    assert s_bif > s_term


# ---------------------------------------------------------------------------
# Position scoring
# ---------------------------------------------------------------------------


def test_center_position_high_score() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    m = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # position contribution should be 1.0 * 0.3 = 0.3
    # type = 0.7 * 0.4 = 0.28, total = 0.58
    assert score > 0.5


def test_border_position_low_score() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    # At pixel (0, 0) — extreme corner
    m = {"x": 0.0, "y": 0.0, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # position = 0.0, type = 0.28, support = 0 → 0.28
    assert score < 0.35


def test_near_border_interpolation() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    # 5% margin = 12.8 pixels → pixel 6 is ~halfway
    m = {"x": 6.0 / 256, "y": 128.0 / 256, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # position ~6/12.8 = 0.47 → 0.47 * 0.3 = 0.14
    # type = 0.28, support = 0 → total ~0.42
    assert 0.3 < score < 0.55


# ---------------------------------------------------------------------------
# Support scoring
# ---------------------------------------------------------------------------


def test_termination_with_support() -> None:
    """Termination with exactly 1 skeleton neighbour."""
    # Termination at center, one arm going right
    skel = _make_skeleton_with_branch(128, 128, 1)
    m = {"x": 128.0 / 256, "y": 128.0 / 256, "angle": 0.0, "type": 1}
    score = score_minutia(m, skel, (256, 256))
    # support = 1/1 = 1.0 → 1.0 * 0.3 = 0.3
    # type = 0.4 * 0.4 = 0.16, position = 1.0 * 0.3 = 0.3
    # total = 0.76
    assert score > 0.6


def test_bifurcation_with_support() -> None:
    """Bifurcation with exactly 3 skeleton neighbours."""
    skel = _make_skeleton_with_branch(128, 128, 3)
    m = {"x": 128.0 / 256, "y": 128.0 / 256, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # support = 3/3 = 1.0 → 0.3
    # type = 0.7 * 0.4 = 0.28, position = 0.3
    # total = 0.88
    assert score > 0.8


def test_no_support_returns_zero_support_score() -> None:
    skel = np.zeros((256, 256), dtype=np.uint8)
    m = {"x": 0.5, "y": 0.5, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # type = 0.28, position → ~0.3, support = 0
    # total = 0.58
    # Verify no support contribution
    no_type_pos = 0.28 + 0.3  # approximate
    assert score <= no_type_pos + 0.05  # support is 0


def test_bifurcation_partial_support() -> None:
    """Bifurcation with only 2 of 3 expected neighbours."""
    skel = _make_skeleton_with_branch(128, 128, 2)
    m = {"x": 128.0 / 256, "y": 128.0 / 256, "angle": 0.0, "type": 3}
    score = score_minutia(m, skel, (256, 256))
    # support = 2/3 = 0.67 → 0.67 * 0.3 = 0.2
    # type = 0.28, position = 0.3
    # total = 0.78
    assert 0.7 < score < 0.9

"""Tests for 3-point alignment module.

Uses synthetic point correspondences to verify that the Procrustes-based
similarity transform recovers the expected parameters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.processing.triplet_alignment import align_3pts, align_n_pts, apply_transform


@pytest.fixture
def triangle() -> np.ndarray:
    """A simple non-degenerate triangle in normalised coordinates."""
    return np.array([[0.2, 0.2], [0.3, 0.5], [0.5, 0.3]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity(triangle: np.ndarray) -> None:
    """Same point sets → identity transform."""
    t = align_3pts(triangle, triangle)
    assert t.scale == pytest.approx(1.0, abs=1e-6)
    assert t.angle == pytest.approx(0.0, abs=1e-6)
    assert t.dx == pytest.approx(0.0, abs=1e-6)
    assert t.dy == pytest.approx(0.0, abs=1e-6)

    # Apply should return original points
    transformed = apply_transform(triangle, t)
    np.testing.assert_allclose(transformed, triangle, atol=1e-6)


# ---------------------------------------------------------------------------
# Pure translation
# ---------------------------------------------------------------------------


def test_translation(triangle: np.ndarray) -> None:
    """Shifted points → recover translation."""
    dx, dy = 0.1, 0.05
    shifted = triangle + np.array([dx, dy])

    t = align_3pts(triangle, shifted)
    assert t.scale == pytest.approx(1.0, abs=1e-6)
    assert t.angle == pytest.approx(0.0, abs=1e-6)
    assert t.dx == pytest.approx(dx, abs=1e-6)
    assert t.dy == pytest.approx(dy, abs=1e-6)

    transformed = apply_transform(triangle, t)
    np.testing.assert_allclose(transformed, shifted, atol=1e-6)


# ---------------------------------------------------------------------------
# Pure rotation
# ---------------------------------------------------------------------------


def test_rotation_90deg(triangle: np.ndarray) -> None:
    """90° rotation around origin → angle ≈ π/2."""
    angle = math.pi / 2
    c, s = math.cos(angle), math.sin(angle)
    rotated = triangle @ np.array([[c, -s], [s, c]])

    # Center points at origin for clean rotation
    center = np.mean(triangle, axis=0)
    tri_centered = triangle - center
    rot_centered = rotated - center

    t = align_3pts(tri_centered, rot_centered)
    assert abs(t.angle) == pytest.approx(angle, abs=1e-4)

    transformed = apply_transform(tri_centered, t)
    np.testing.assert_allclose(transformed, rot_centered, atol=1e-4)


# ---------------------------------------------------------------------------
# Pure scale
# ---------------------------------------------------------------------------


def test_scale_double(triangle: np.ndarray) -> None:
    """Doubled distances → scale ≈ 2."""
    scaled = triangle * 2.0

    t = align_3pts(triangle, scaled)
    assert t.scale == pytest.approx(2.0, abs=1e-4)
    assert abs(t.angle) == pytest.approx(0.0, abs=1e-4)

    transformed = apply_transform(triangle, t)
    np.testing.assert_allclose(transformed, scaled, atol=1e-4)


# ---------------------------------------------------------------------------
# Combined transform
# ---------------------------------------------------------------------------


def test_combined(triangle: np.ndarray) -> None:
    """Scale + rotation + translation."""
    s = 1.5
    angle = math.radians(30)
    dx, dy = 0.05, 0.1
    c, s_ = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s_], [s_, c]])

    transformed_pts = s * (triangle @ R.T) + np.array([dx, dy])

    t = align_3pts(triangle, transformed_pts)
    assert t.scale == pytest.approx(s, abs=1e-3)
    assert t.angle == pytest.approx(angle, abs=1e-3)

    result = apply_transform(triangle, t)
    np.testing.assert_allclose(result, transformed_pts, atol=1e-3)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip(triangle: np.ndarray) -> None:
    """Apply then invert (approximate) → original."""
    dx, dy = 0.1, 0.05
    angle = math.radians(15)
    c, s = math.cos(angle), math.sin(angle)

    # Forward transform
    R = np.array([[c, -s], [s, c]])
    fwd = triangle @ R.T + np.array([dx, dy])

    t = align_3pts(triangle, fwd)
    result = apply_transform(triangle, t)
    np.testing.assert_allclose(result, fwd, atol=1e-4)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_wrong_shape() -> None:
    """Non-(3,2) arrays raise ValueError."""
    with pytest.raises(ValueError, match="Expected"):
        align_3pts(
            np.array([[0.2, 0.2], [0.3, 0.5]], dtype=np.float64),
            np.array([[0.2, 0.2], [0.3, 0.5]], dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# align_n_pts — N-point Procrustes
# ---------------------------------------------------------------------------


def test_n_pts_identity() -> None:
    """Identity transform with 6 identical point pairs."""
    pts = np.array([
        [0.1, 0.2], [0.3, 0.4], [0.5, 0.1],
        [0.7, 0.6], [0.2, 0.8], [0.9, 0.3],
    ], dtype=np.float64)
    t = align_n_pts(pts, pts)
    assert t.scale == pytest.approx(1.0, abs=1e-6)
    assert t.angle == pytest.approx(0.0, abs=1e-6)
    assert t.dx == pytest.approx(0.0, abs=1e-6)
    assert t.dy == pytest.approx(0.0, abs=1e-6)


def test_n_pts_translation() -> None:
    """Translation with 5 corresponding pairs."""
    base = np.array([
        [0.1, 0.2], [0.3, 0.4], [0.5, 0.1], [0.7, 0.6], [0.2, 0.8],
    ], dtype=np.float64)
    dx, dy = 0.1, 0.05
    shifted = base + np.array([dx, dy])

    t = align_n_pts(base, shifted)
    assert t.scale == pytest.approx(1.0, abs=1e-6)
    assert t.angle == pytest.approx(0.0, abs=1e-6)
    assert t.dx == pytest.approx(dx, abs=1e-6)
    assert t.dy == pytest.approx(dy, abs=1e-6)

    transformed = apply_transform(base, t)
    np.testing.assert_allclose(transformed, shifted, atol=1e-6)


def test_n_pts_combined_9_pairs() -> None:
    """Scale + rotation + translation with 9 point pairs.

    Validates that the least-squares solution over 9 pairs recovers
    the original transform within numerical tolerance.
    """
    base = np.array([
        [0.1, 0.2], [0.3, 0.4], [0.5, 0.1],
        [0.7, 0.6], [0.2, 0.8], [0.4, 0.3],
        [0.6, 0.5], [0.8, 0.2], [0.1, 0.5],
    ], dtype=np.float64)
    s = 1.5
    angle = math.radians(20)
    dx, dy = 0.05, 0.1
    c, s_ = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s_], [s_, c]])
    transformed_pts = s * (base @ R.T) + np.array([dx, dy])

    t = align_n_pts(base, transformed_pts)
    assert t.scale == pytest.approx(s, abs=1e-3)
    assert t.angle == pytest.approx(angle, abs=1e-3)
    assert t.dx == pytest.approx(dx, abs=1e-3)
    assert t.dy == pytest.approx(dy, abs=1e-3)


def test_n_pts_noisy_inliers() -> None:
    """With noisy inliers, least-squares over 6 pairs is more robust than 3."""
    base = np.array([
        [0.1, 0.2], [0.3, 0.4], [0.5, 0.1],
        [0.7, 0.6], [0.2, 0.8], [0.9, 0.3],
    ], dtype=np.float64)
    s = 1.2
    angle = math.radians(15)
    dx, dy = 0.03, 0.07
    c, s_ = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s_], [s_, c]])

    # Add Gaussian noise to the candidate points
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.005, size=base.shape)
    cand_noisy = s * (base @ R.T) + np.array([dx, dy]) + noise

    t = align_n_pts(base, cand_noisy)
    assert t.scale == pytest.approx(s, abs=0.05)
    assert t.angle == pytest.approx(angle, abs=0.05)
    assert t.dx == pytest.approx(dx, abs=0.02)
    assert t.dy == pytest.approx(dy, abs=0.02)


def test_n_pts_too_few_points() -> None:
    """Fewer than 3 points raises ValueError."""
    with pytest.raises(ValueError, match="at least 3"):
        align_n_pts(
            np.array([[0.2, 0.2], [0.3, 0.5]], dtype=np.float64),
            np.array([[0.2, 0.2], [0.3, 0.5]], dtype=np.float64),
        )


def test_n_pts_shape_mismatch() -> None:
    """Mismatched shapes between probe and cand raise ValueError."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        align_n_pts(
            np.array([[0.2, 0.2], [0.3, 0.5], [0.5, 0.3]], dtype=np.float64),
            np.array([[0.2, 0.2], [0.3, 0.5]], dtype=np.float64),
        )


def test_align_3pts_delegates_to_n_pts() -> None:
    """align_3pts returns the same result as align_n_pts with 3 pairs."""
    triangle = np.array([[0.2, 0.2], [0.3, 0.5], [0.5, 0.3]], dtype=np.float64)
    s = 1.3
    angle = math.radians(10)
    c, s_ = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s_], [s_, c]])
    cand = s * (triangle @ R.T) + np.array([0.02, 0.04])

    t3 = align_3pts(triangle, cand)
    tn = align_n_pts(triangle, cand)

    assert t3.scale == pytest.approx(tn.scale, abs=1e-9)
    assert t3.angle == pytest.approx(tn.angle, abs=1e-9)
    assert t3.dx == pytest.approx(tn.dx, abs=1e-9)
    assert t3.dy == pytest.approx(tn.dy, abs=1e-9)

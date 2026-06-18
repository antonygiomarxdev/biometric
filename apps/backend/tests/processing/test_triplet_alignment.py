"""Tests for 3-point alignment module.

Uses synthetic point correspondences to verify that the Procrustes-based
similarity transform recovers the expected parameters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.processing.triplet_alignment import align_3pts, apply_transform


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

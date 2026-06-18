"""Tests for triplet extraction.

Uses synthetic minutiae lists and skeleton images to verify correctness,
determinism, and geometric invariance.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from src.processing.triplet_extractor import (
    extract_triplets,
    triplet_to_vector,
    _encode_type_triple,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_skeleton() -> np.ndarray:
    return np.zeros((256, 256), dtype=np.uint8)


@pytest.fixture
def shape_256() -> tuple[int, int]:
    return (256, 256)


def _make_minutiae(
    coords: list[tuple[float, float, float, int]],
) -> list[dict]:
    """Build a minutiae list from (x, y, angle_deg, type) tuples."""
    return [
        {"x": x, "y": y, "angle": math.radians(a), "type": t}
        for x, y, a, t in coords
    ]


def _make_skeleton_with_minutiae(
    coords: list[tuple[float, float]],
    size: int = 256,
) -> np.ndarray:
    """Create a skeleton with foreground pixels at given normalised coords."""
    skel = np.zeros((size, size), dtype=np.uint8)
    for nx, ny in coords:
        px = int(round(nx * size))
        py = int(round(ny * size))
        if 0 <= px < size and 0 <= py < size:
            skel[py, px] = 1
    return skel


# ---------------------------------------------------------------------------
# Type encoding
# ---------------------------------------------------------------------------


def test_type_triple_all_terminations() -> None:
    assert _encode_type_triple(1, 1, 1) == 0


def test_type_triple_all_bifurcations() -> None:
    assert _encode_type_triple(3, 3, 3) == 13  # 1*9 + 1*3 + 1


def test_type_triple_mixed() -> None:
    # 1→0, 3→1, 2→2
    assert _encode_type_triple(1, 3, 2) == 0 * 9 + 1 * 3 + 2  # 5


def test_type_triple_unknown_handling() -> None:
    assert _encode_type_triple(2, 2, 2) == 26  # 2*9 + 2*3 + 2


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_three_minutiae_one_triplet(empty_skeleton, shape_256) -> None:
    """3 minutiae all within radius → 1 triplet."""
    coords = [(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)]
    minutiae = _make_minutiae(coords)
    triplets = extract_triplets(
        minutiae, empty_skeleton, shape_256,
        quality_threshold=0.0,
        max_radius=0.25,
    )
    assert len(triplets) == 1


def test_zero_minutiae_no_triplets(empty_skeleton, shape_256) -> None:
    triplets = extract_triplets(
        [], empty_skeleton, shape_256,
    )
    assert len(triplets) == 0


def test_one_minutia_no_triplets(empty_skeleton, shape_256) -> None:
    m = _make_minutiae([(0.5, 0.5, 0.0, 1)])
    triplets = extract_triplets(m, empty_skeleton, shape_256)
    assert len(triplets) == 0


def test_two_minutiae_no_triplets(empty_skeleton, shape_256) -> None:
    m = _make_minutiae([(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3)])
    triplets = extract_triplets(m, empty_skeleton, shape_256)
    assert len(triplets) == 0


def test_determinism(empty_skeleton, shape_256) -> None:
    """Same input twice → identical triplets."""
    coords = [
        (0.2, 0.2, 0.0, 1),
        (0.25, 0.3, 45.0, 3),
        (0.3, 0.25, 90.0, 1),
        (0.4, 0.4, 10.0, 3),
        (0.5, 0.5, 30.0, 1),
        (0.15, 0.35, 60.0, 3),
    ]
    minutiae = _make_minutiae(coords)

    t1 = extract_triplets(
        minutiae, empty_skeleton, shape_256,
        quality_threshold=0.0,
        max_radius=0.3,
        max_triplets=500,
    )
    t2 = extract_triplets(
        minutiae, empty_skeleton, shape_256,
        quality_threshold=0.0,
        max_radius=0.3,
        max_triplets=500,
    )

    # Compare by sorted vector representations
    def _key(t):
        return tuple(round(v, 6) for v in triplet_to_vector(t))

    keys1 = sorted(_key(t) for t in t1)
    keys2 = sorted(_key(t) for t in t2)
    assert keys1 == keys2


def test_max_triplets_cap(empty_skeleton, shape_256) -> None:
    """Many minutiae produce bounded triplets."""
    coords = []
    for i in range(15):
        x = 0.2 + 0.02 * i
        y = 0.2 + 0.02 * (i % 5)
        coords.append((x, y, float(i * 10), 3))
    minutiae = _make_minutiae(coords)

    triplets = extract_triplets(
        minutiae, empty_skeleton, shape_256,
        quality_threshold=0.0,
        max_radius=0.25,
        max_triplets=50,
    )
    assert len(triplets) <= 50


def test_quality_threshold_filters(shape_256) -> None:
    """Minutiae below quality threshold produce fewer triplets."""
    # Place minutiae at border (low position score) vs center (high position)
    skeleton = np.zeros((256, 256), dtype=np.uint8)

    border = [(0.01, 0.5, 0.0, 3), (0.03, 0.52, 45.0, 3), (0.02, 0.48, 90.0, 3)]
    center = [(0.5, 0.5, 0.0, 3), (0.55, 0.55, 45.0, 3), (0.5, 0.45, 90.0, 3)]

    border_m = _make_minutiae(border)
    center_m = _make_minutiae(center)

    t_border = extract_triplets(
        border_m, skeleton, shape_256,
        quality_threshold=0.3,
        max_radius=0.25,
    )
    t_center = extract_triplets(
        center_m, skeleton, shape_256,
        quality_threshold=0.3,
        max_radius=0.25,
    )

    # Center groups should have more or equal triplets vs border
    assert len(t_center) >= len(t_border)


# ---------------------------------------------------------------------------
# Vector descriptor
# ---------------------------------------------------------------------------


def test_vector_six_components(empty_skeleton, shape_256) -> None:
    m = _make_minutiae([(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)])
    triplets = extract_triplets(
        m, empty_skeleton, shape_256,
        quality_threshold=0.0, max_radius=0.25,
    )
    assert len(triplets) == 1
    v = triplet_to_vector(triplets[0])
    assert len(v) == 6
    # Vector should be unit-length
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_two_triplets_different_vectors(empty_skeleton, shape_256) -> None:
    """Two different triplet configurations produce different vectors."""
    m1 = _make_minutiae([(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)])
    m2 = _make_minutiae([(0.2, 0.2, 0.0, 1), (0.28, 0.25, 45.0, 3), (0.25, 0.32, 90.0, 1)])

    t1 = extract_triplets(m1, empty_skeleton, shape_256, quality_threshold=0.0, max_radius=0.25)
    t2 = extract_triplets(m2, empty_skeleton, shape_256, quality_threshold=0.0, max_radius=0.25)

    assert len(t1) >= 1 and len(t2) >= 1
    v1 = triplet_to_vector(t1[0])
    v2 = triplet_to_vector(t2[0])

    # Vectors should differ (different geometry)
    diff = sum(abs(a - b) for a, b in zip(v1, v2, strict=False))
    assert diff > 1e-4


# ---------------------------------------------------------------------------
# Invariance tests
# ---------------------------------------------------------------------------


def test_translation_invariance(empty_skeleton, shape_256) -> None:
    """Translating all points by same vector produces same vector."""
    base = [(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)]
    shift = 0.15
    translated = [(x + shift, y + shift, a, t) for x, y, a, t in base]

    m1 = _make_minutiae(base)
    m2 = _make_minutiae(translated)

    t1 = extract_triplets(m1, empty_skeleton, shape_256, quality_threshold=0.0)
    t2 = extract_triplets(m2, empty_skeleton, shape_256, quality_threshold=0.0)

    assert len(t1) >= 1 and len(t2) >= 1
    v1 = triplet_to_vector(t1[0])
    v2 = triplet_to_vector(t2[0])

    for a, b in zip(v1, v2, strict=False):
        assert abs(a - b) < 1e-6


def test_rotation_invariance(empty_skeleton, shape_256) -> None:
    """Rotating all points by same angle produces same vector."""
    base = [(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)]
    cx, cy = 0.35, 0.35
    angle = math.radians(30)

    rotated = []
    for x, y, a, t in base:
        rx = cx + (x - cx) * math.cos(angle) - (y - cy) * math.sin(angle)
        ry = cy + (x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        rotated.append((rx, ry, a + math.degrees(angle), t))

    m1 = _make_minutiae(base)
    m2 = _make_minutiae(rotated)

    t1 = extract_triplets(m1, empty_skeleton, shape_256, quality_threshold=0.0)
    t2 = extract_triplets(m2, empty_skeleton, shape_256, quality_threshold=0.0)

    assert len(t1) >= 1 and len(t2) >= 1
    v1 = triplet_to_vector(t1[0])
    v2 = triplet_to_vector(t2[0])

    for a, b in zip(v1, v2, strict=False):
        assert abs(a - b) < 1e-4


def test_scale_invariance(empty_skeleton, shape_256) -> None:
    """Scaling all distances by factor produces same vector."""
    base = [(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)]
    cx, cy = 0.3, 0.3
    factor = 0.5

    scaled = []
    for x, y, a, t in base:
        sx = cx + (x - cx) * factor
        sy = cy + (y - cy) * factor
        scaled.append((sx, sy, a, t))

    m1 = _make_minutiae(base)
    m2 = _make_minutiae(scaled)

    t1 = extract_triplets(m1, empty_skeleton, shape_256, quality_threshold=0.0)
    t2 = extract_triplets(m2, empty_skeleton, shape_256, quality_threshold=0.0)

    assert len(t1) >= 1 and len(t2) >= 1
    v1 = triplet_to_vector(t1[0])
    v2 = triplet_to_vector(t2[0])

    for a, b in zip(v1, v2, strict=False):
        assert abs(a - b) < 1e-6


# ---------------------------------------------------------------------------
# Triplet dict structure
# ---------------------------------------------------------------------------


def test_triplet_dict_keys(empty_skeleton, shape_256) -> None:
    """Triplet dict contains all expected keys."""
    m = _make_minutiae([(0.3, 0.3, 0.0, 1), (0.35, 0.35, 45.0, 3), (0.4, 0.3, 90.0, 1)])
    triplets = extract_triplets(m, empty_skeleton, shape_256, quality_threshold=0.0)

    expected_keys = {
        "mi_idx", "mj_idx", "mk_idx",
        "mi_x", "mi_y", "mi_angle",
        "mj_x", "mj_y", "mj_angle",
        "mk_x", "mk_y", "mk_angle",
        "d_ij", "d_ik", "d_jk",
        "type_triple",
        "quality_min", "quality_avg",
    }

    actual_keys = set(triplets[0].keys())
    assert actual_keys >= expected_keys, f"Missing: {expected_keys - actual_keys}"


def test_triplet_indices_preserved(empty_skeleton, shape_256) -> None:
    """Indices into the original minutiae list are correct."""
    m = _make_minutiae([
        (0.1, 0.1, 0.0, 1),
        (0.2, 0.2, 45.0, 3),
        (0.3, 0.3, 90.0, 1),
        (0.5, 0.5, 30.0, 3),  # far away
    ])
    triplets = extract_triplets(m, empty_skeleton, shape_256, quality_threshold=0.0)
    for t in triplets:
        assert 0 <= t["mi_idx"] < len(m)
        assert 0 <= t["mj_idx"] < len(m)
        assert 0 <= t["mk_idx"] < len(m)
        assert len({t["mi_idx"], t["mj_idx"], t["mk_idx"]}) == 3

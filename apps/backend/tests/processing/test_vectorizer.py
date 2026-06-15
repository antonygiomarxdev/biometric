"""Tests for TripletVectorizer (Delaunay-based invariants)."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
)


def make_minutia(x: int, y: int, angle: float = 0.0,
                 mtype: MinutiaType = MinutiaType.TERMINATION) -> MinutiaCandidate:
    return MinutiaCandidate(
        x=x, y=y, angle=angle,
        type=mtype, confidence=1.0,
        origin=AlgorithmOrigin.SKELETON,
    )


class TestTripletVectorizer:
    """Triplet-based vectorisation for partial/latent prints."""

    def test_empty_returns_zeros(self) -> None:
        from src.processing.vectorizer import TripletVectorizer

        v = TripletVectorizer()
        vec = v.to_vector([])
        assert vec.shape == (v.target_dim,)
        assert vec.sum() == 0.0

    def test_fewer_than_3_returns_zeros(self) -> None:
        from src.processing.vectorizer import TripletVectorizer

        v = TripletVectorizer()
        vec = v.to_vector([make_minutia(0, 0), make_minutia(1, 1)])
        assert vec.sum() == 0.0

    def test_with_three_minutiae(self) -> None:
        from src.processing.vectorizer import TripletVectorizer

        v = TripletVectorizer()
        pts = [
            make_minutia(0, 0, mtype=MinutiaType.TERMINATION),
            make_minutia(10, 0, mtype=MinutiaType.BIFURCATION),
            make_minutia(5, 8, mtype=MinutiaType.TERMINATION),
        ]
        vec = v.to_vector(pts)
        assert vec.shape == (v.target_dim,)
        # First 9 values should be the triangle features
        assert vec[0] > 0  # side length d12
        assert vec[3] > 0  # angle A
        assert vec[6] in (0.0, 1.0)  # type

    def test_rotation_invariance(self) -> None:
        """Rotated minutiae produce identical triplet vectors."""
        from src.processing.vectorizer import TripletVectorizer

        v = TripletVectorizer()
        pts_a = [
            make_minutia(0, 0), make_minutia(10, 0), make_minutia(5, 8),
        ]
        # Rotate by 90°: (x, y) → (-y, x)
        pts_b = [
            make_minutia(0, 0), make_minutia(0, 10), make_minutia(-8, 5),
        ]
        vec_a = v.to_vector(pts_a)
        vec_b = v.to_vector(pts_b)

        for i in range(9):
            assert abs(vec_a[i] - vec_b[i]) < 1e-3, f"Diff at index {i}: {vec_a[i]} vs {vec_b[i]}"

    def test_target_dim_fixed(self) -> None:
        from src.processing.vectorizer import TripletVectorizer
        v = TripletVectorizer(max_triangles=50)
        assert v.target_dim == 450

    def test_from_vector_produces_expected_count(self) -> None:
        from src.processing.vectorizer import TripletVectorizer
        v = TripletVectorizer(max_triangles=50)
        pts = v.from_vector(np.zeros(450, dtype=np.float32), expected_minutiae=5)
        assert len(pts) == 5

"""Tests for MinutiaNormalizer and RotationInvariantNormalizer."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
)


def make_minutia(x: int, y: int, angle: float = 0.0,
                 mtype: MinutiaType = MinutiaType.TERMINATION,
                 confidence: float = 1.0) -> MinutiaCandidate:
    """Helper to create a MinutiaCandidate."""
    return MinutiaCandidate(
        x=x, y=y, angle=angle,
        type=mtype, confidence=confidence,
        origin=AlgorithmOrigin.SKELETON,
    )


class TestMinutiaNormalizer:
    """Standard geometric normalizer."""

    def test_normalize_empty_minutiae(self) -> None:
        """Normalizing an empty list returns a fingerprint with empty minutiae."""
        from src.processing.normalization import MinutiaNormalizer

        normalizer = MinutiaNormalizer()
        result = normalizer.normalize([], (100, 200))

        assert isinstance(result, NormalizedFingerprint)
        assert result.minutiae == []
        assert result.width == 200
        assert result.height == 100

    def test_normalize_centers_minutiae(self) -> None:
        """Minutiae are centered around their centroid."""
        from src.processing.normalization import MinutiaNormalizer

        normalizer = MinutiaNormalizer()
        minutiae = [
            make_minutia(10, 10),
            make_minutia(30, 30),
        ]
        result = normalizer.normalize(minutiae, (100, 100))
        # Centroid is (20, 20). After centering: (10-20=-10, 10-20=-10), (30-20=10, 30-20=10)
        assert result.minutiae[0].x == -10
        assert result.minutiae[0].y == -10
        assert result.minutiae[1].x == 10
        assert result.minutiae[1].y == 10

    def test_apply_consensus_removes_duplicates(self) -> None:
        """Close minutiae are merged (highest confidence wins)."""
        from src.processing.normalization import MinutiaNormalizer

        normalizer = MinutiaNormalizer()
        minutiae = [
            make_minutia(10, 10, confidence=0.9),
            make_minutia(11, 10, confidence=0.5),  # within 5px → duplicate
        ]
        result = normalizer._apply_consensus(minutiae)
        assert len(result) == 1  # merged
        assert result[0].confidence == 0.9  # highest confidence survives

    def test_apply_consensus_empty(self) -> None:
        """Empty list returns empty list."""
        from src.processing.normalization import MinutiaNormalizer

        normalizer = MinutiaNormalizer()
        assert normalizer._apply_consensus([]) == []

    def test_canonical_sort_by_distance(self) -> None:
        """Minutiae are sorted by distance from origin."""
        from src.processing.normalization import MinutiaNormalizer

        normalizer = MinutiaNormalizer()
        minutiae = [
            make_minutia(10, 0),   # r² = 100
            make_minutia(1, 1),    # r² = 2  → should be first
            make_minutia(3, 4),    # r² = 25
        ]
        sorted_m = normalizer._canonical_sort(minutiae)
        distances = [m.x**2 + m.y**2 for m in sorted_m]
        assert distances == sorted(distances)


class TestRotationInvariantNormalizer:
    """PCA-based rotation invariant normalizer."""

    def test_fewer_than_3_minutiae_falls_back(self) -> None:
        """With < 3 minutiae, falls back to the base normalizer."""
        from src.processing.normalization import RotationInvariantNormalizer

        normalizer = RotationInvariantNormalizer()
        minutiae = [make_minutia(10, 10)]
        result = normalizer.normalize(minutiae, (100, 100))
        assert isinstance(result, NormalizedFingerprint)
        assert len(result.minutiae) == 1

    def test_normalizes_with_rotation(self) -> None:
        """With >= 3 minutiae, PCA rotation is applied."""
        from src.processing.normalization import RotationInvariantNormalizer

        normalizer = RotationInvariantNormalizer()
        # Create a horizontal line of minutiae
        minutiae = [
            make_minutia(0, 50),
            make_minutia(50, 50),
            make_minutia(100, 50),
        ]
        result = normalizer.normalize(minutiae, (100, 100))
        assert isinstance(result, NormalizedFingerprint)
        assert len(result.minutiae) == 3

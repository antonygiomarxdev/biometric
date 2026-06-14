"""Tests for MinutiaeVectorizer."""

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
    """Helper to create a MinutiaCandidate."""
    return MinutiaCandidate(
        x=x, y=y, angle=angle,
        type=mtype, confidence=1.0,
        origin=AlgorithmOrigin.SKELETON,
    )


class TestMinutiaeVectorizer:
    """Vectoriser converts minutiae lists to/from numpy arrays."""

    def test_to_vector_with_terminations(self) -> None:
        """Termination minutiae produce type_id=0."""
        from src.processing.vectorizer import MinutiaeVectorizer

        minutiae = [make_minutia(10, 20, 45.0, MinutiaType.TERMINATION)]
        vec = MinutiaeVectorizer.to_vector(minutiae)

        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        # [type=0, x=10, y=20, angle=45]
        assert vec[0] == 0.0
        assert vec[1] == 10.0
        assert vec[2] == 20.0
        assert vec[3] == 45.0

    def test_to_vector_with_bifurcations(self) -> None:
        """Bifurcation minutiae produce type_id=1."""
        from src.processing.vectorizer import MinutiaeVectorizer

        minutiae = [make_minutia(5, 15, 90.0, MinutiaType.BIFURCATION)]
        vec = MinutiaeVectorizer.to_vector(minutiae)

        assert vec[0] == 1.0  # bifurcation type
        assert vec[1] == 5.0
        assert vec[2] == 15.0
        assert vec[3] == 90.0

    def test_to_vector_multiple_minutiae(self) -> None:
        """Multiple minutiae are concatenated in the vector."""
        from src.processing.vectorizer import MinutiaeVectorizer

        minutiae = [
            make_minutia(1, 2, 10.0, MinutiaType.TERMINATION),
            make_minutia(3, 4, 20.0, MinutiaType.BIFURCATION),
        ]
        vec = MinutiaeVectorizer.to_vector(minutiae)
        assert len(vec) == 8  # 2 minutiae × 4 fields
        assert vec[4] == 1.0  # second minutia is bifurcation

    def test_to_vector_empty(self) -> None:
        """Empty minutiae list produces an empty array."""
        from src.processing.vectorizer import MinutiaeVectorizer

        vec = MinutiaeVectorizer.to_vector([])
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 0

    def test_from_vector_reconstructs(self) -> None:
        """Vector is reconstructable back to minutiae."""
        from src.processing.vectorizer import MinutiaeVectorizer

        vec = np.array([0.0, 10.0, 20.0, 45.0], dtype=np.float32)
        minutiae = MinutiaeVectorizer.from_vector(vec)

        assert len(minutiae) == 1
        assert minutiae[0].x == 10
        assert minutiae[0].y == 20
        assert minutiae[0].angle == 45.0
        assert minutiae[0].type == MinutiaType.TERMINATION

    def test_from_vector_bifurcation(self) -> None:
        """Type_id=1 is decoded as BIFURCATION."""
        from src.processing.vectorizer import MinutiaeVectorizer

        vec = np.array([1.0, 5.0, 10.0, 0.0], dtype=np.float32)
        minutiae = MinutiaeVectorizer.from_vector(vec)

        assert minutiae[0].type == MinutiaType.BIFURCATION

    def test_from_vector_handles_partial(self) -> None:
        """Partial data at end of vector is handled gracefully."""
        from src.processing.vectorizer import MinutiaeVectorizer

        vec = np.array([0.0, 10.0], dtype=np.float32)  # only 2 elements (< 4)
        minutiae = MinutiaeVectorizer.from_vector(vec)
        assert len(minutiae) == 0

    def test_pad_vector_truncates(self) -> None:
        """pad_vector truncates when vector is longer than target."""
        from src.processing.vectorizer import MinutiaeVectorizer

        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        padded = MinutiaeVectorizer.pad_vector(vec, 3)
        assert len(padded) == 3
        assert list(padded) == [1.0, 2.0, 3.0]

    def test_pad_vector_pads_with_zeros(self) -> None:
        """pad_vector pads with zeros when vector is shorter."""
        from src.processing.vectorizer import MinutiaeVectorizer

        vec = np.array([1.0, 2.0], dtype=np.float32)
        padded = MinutiaeVectorizer.pad_vector(vec, 5)
        assert len(padded) == 5
        assert list(padded[:2]) == [1.0, 2.0]
        assert list(padded[2:]) == [0.0, 0.0, 0.0]

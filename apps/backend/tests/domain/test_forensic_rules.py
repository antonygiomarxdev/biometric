"""Tests for forensic validation rules (Phase 10 - Strategy Pattern)."""
from __future__ import annotations

import pytest

from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
)
from src.domain.forensic_rules import (
    EnrollmentValidationStrategy,
    InsufficientFeaturesError,
    SearchValidationStrategy,
)


def _candidates(n: int) -> list[MinutiaCandidate]:
    return [
        MinutiaCandidate(
            x=i * 5, y=i * 5, angle=0.0,
            type=MinutiaType.TERMINATION,
            confidence=1.0,
            origin=AlgorithmOrigin.SKELETON,
        )
        for i in range(n)
    ]


class TestEnrollmentValidationStrategy:
    """A full ten-print must have at least 8 minutiae."""

    def test_passes_with_exactly_eight_minutiae(self) -> None:
        EnrollmentValidationStrategy().validate(_candidates(8))

    def test_passes_with_more_than_eight_minutiae(self) -> None:
        EnrollmentValidationStrategy().validate(_candidates(20))

    @pytest.mark.parametrize("count", [0, 1, 2, 7])
    def test_fails_with_fewer_than_eight_minutiae(self, count: int) -> None:
        with pytest.raises(InsufficientFeaturesError) as exc_info:
            EnrollmentValidationStrategy().validate(_candidates(count))
        assert "at least 8" in str(exc_info.value)

    def test_min_constant_is_eight(self) -> None:
        assert EnrollmentValidationStrategy.MIN_MINUTIAE == 8


class TestSearchValidationStrategy:
    """A latent fragment must have at least 2 minutiae to search."""

    def test_passes_with_exactly_two_minutiae(self) -> None:
        SearchValidationStrategy().validate(_candidates(2))

    def test_passes_with_many_minutiae(self) -> None:
        SearchValidationStrategy().validate(_candidates(50))

    @pytest.mark.parametrize("count", [0, 1])
    def test_fails_with_fewer_than_two_minutiae(self, count: int) -> None:
        with pytest.raises(InsufficientFeaturesError) as exc_info:
            SearchValidationStrategy().validate(_candidates(count))
        assert "at least 2" in str(exc_info.value)

    def test_min_constant_is_two(self) -> None:
        assert SearchValidationStrategy.MIN_MINUTIAE == 2

    def test_search_is_more_permissive_than_enrollment(self) -> None:
        """A 5-minutiae fragment should fail enrollment but pass search."""
        candidates = _candidates(5)
        with pytest.raises(InsufficientFeaturesError):
            EnrollmentValidationStrategy().validate(candidates)
        # Same candidates should be valid for search
        SearchValidationStrategy().validate(candidates)

"""Forensic Business Rules (Phase 10).

Strategy Pattern: encapsulates the validation thresholds that an expert
forensic examiner would apply to a fingerprint before treating it as
evidentially useful.

Per the expert domain knowledge:
  - ENROLL: a high-quality full ten-print must contain at least 8
    minutiae. Below this threshold, a print has no identificatory
    value and is rejected from the registry.
  - SEARCH: a latent fragment (crime scene lift) may contain as few
    as 2 minutiae. We tolerate the noise because the matcher is
    robust: noisy triplets simply will not match anything in the
    database, so they self-filter at retrieval time.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.types import MinutiaCandidate


class InsufficientFeaturesError(ValueError):
    """Raised when a fingerprint fails a forensic validation rule."""


@runtime_checkable
class IForensicValidationStrategy(Protocol):
    """Validates that a fingerprint meets the minimum forensic threshold
    for the operation being performed (enrollment or search)."""

    def validate(self, candidates: list[MinutiaCandidate]) -> None:
        """Raises InsufficientFeaturesError if the print is below the
        minimum number of minutiae for this operation."""
        ...


class EnrollmentValidationStrategy:
    """Validates that a print is suitable for enrollment in the registry.

    A full ten-print must have at least 8 minutiae to have
    identificatory value (expert forensic rule).
    """

    MIN_MINUTIAE: int = 8

    def validate(self, candidates: list[MinutiaCandidate]) -> None:
        if len(candidates) < self.MIN_MINUTIAE:
            raise InsufficientFeaturesError(
                f"Enrollment requires at least {self.MIN_MINUTIAE} minutiae "
                f"(had {len(candidates)})."
            )


class SearchValidationStrategy:
    """Validates that a latent fragment has enough data to search.

    A crime-scene lift may be a partial fragment; as few as 2
    minutiae can produce a single triangle, which the RAG matcher
    will attempt to match. Noise self-filters at retrieval time.
    """

    MIN_MINUTIAE: int = 2

    def validate(self, candidates: list[MinutiaCandidate]) -> None:
        if len(candidates) < self.MIN_MINUTIAE:
            raise InsufficientFeaturesError(
                f"Search requires at least {self.MIN_MINUTIAE} minutiae "
                f"(had {len(candidates)})."
            )

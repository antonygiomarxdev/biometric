"""Fingerprint domain entity."""

from dataclasses import dataclass
from typing import List

from .minutiae import Minutiae


@dataclass
class Fingerprint:
    """Represents a fingerprint with its extracted minutiae."""

    id: str
    minutiae_points: List[Minutiae]

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return f"Fingerprint(id={self.id})"
        
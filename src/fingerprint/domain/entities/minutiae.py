"""Minutiae domain entity."""

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass
class Minutiae:
    """Represents a minutia point in a fingerprint."""

    type: Literal["termination", "bifurcation"]
    position: Tuple[int, int]
    orientation: float

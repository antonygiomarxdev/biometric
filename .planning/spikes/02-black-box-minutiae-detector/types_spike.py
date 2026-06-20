"""Contract types for the black-box minutiae detector (Spike 02).

These types define the output of the proposed MinutiaeDetector interface.
They are intentionally separate from the production ``MinutiaCandidate`` —
the new type carries forensic context (zone, ridge trace, overlap flag)
that the matcher and the perito's report both need.

The implementation of these types lives in:
  - detector_spike.py       (capa 1, raw candidates)
  - validation_spike.py     (capa 2, contextual validation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from src.core.types import AlgorithmOrigin, MinutiaType


class Zone(Enum):
    """Where a minutia sits relative to the reliable region.

    Used by the matcher for weighting and by the report for
    spatial context. Border minutiae are demoted, near-singularity
    minutiae are usually anchors.
    """

    BORDER = auto()
    INTERIOR = auto()
    NEAR_CORE = auto()
    NEAR_DELTA = auto()


class SingularityKind(Enum):
    """A core or a delta in the orientation field."""

    CORE = "core"
    DELTA = "delta"


@dataclass(frozen=True, slots=True)
class ValidatedMinutia:
    """A minutia candidate after contextual validation.

    The forensic metadata (``zone``, ``ridge_trace_length``,
    ``is_overlap``, ``in_pattern_area``) is what separates this from
    the production ``MinutiaCandidate`` — it carries the context
    the matcher and the perito both need.
    """

    x: int
    y: int
    angle: float
    type: MinutiaType
    confidence: float
    origin: AlgorithmOrigin
    zone: Zone
    ridge_trace_length: int
    is_overlap: bool
    in_pattern_area: bool


@dataclass(frozen=True, slots=True)
class Singularity:
    """A core or delta singularity detected via Poincaré + DORIC."""

    x: int
    y: int
    kind: SingularityKind
    confidence: float
    poincare_value: float


@dataclass(frozen=True)
class QualityZone:
    """A spatial region with a quality score.

    Used for the perito report (e.g. "high quality in the core zone,
    low quality in the upper-left where the latente is fragmented").
    """

    bbox: tuple[int, int, int, int]
    quality_score: float
    n_minutiae: int


@dataclass(frozen=True)
class DetectionResult:
    """The full output of MinutiaeDetector.

    Consumers:
      - Matcher (Bozorth3 pairs): ``minutiae`` (with ``confidence`` as weight)
      - Perito report: ``cores``, ``deltas``, ``quality_zones``
      - Perito UI: ``minutiae`` with colour by ``zone``, size by ``confidence``
    """

    minutiae: list[ValidatedMinutia]
    cores: list[Singularity]
    deltas: list[Singularity]
    pattern_area_mask: np.ndarray | None
    quality_zones: list[QualityZone]
    skeleton: np.ndarray
    enhanced_image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_minutiae(self) -> int:
        return len(self.minutiae)

    @property
    def num_cores(self) -> int:
        return len(self.cores)

    @property
    def num_deltas(self) -> int:
        return len(self.deltas)


__all__ = [
    "DetectionResult",
    "QualityZone",
    "Singularity",
    "SingularityKind",
    "ValidatedMinutia",
    "Zone",
]

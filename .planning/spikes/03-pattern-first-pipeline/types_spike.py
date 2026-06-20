"""Contract types for the pattern-first pipeline (Spike 03).

Extends the spike 02 contract with:
  - ``PatternType`` enum: arch / loop / whorl (real classification, not
    inferred from count).
  - ``FacingDirection`` enum: slant left / right for loops.
  - ``PatternClassification`` dataclass: the result of the classifier
    with its confidence and supporting evidence.
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


class PatternType(Enum):
    """NBIS / Henry classification result.

    UNKNOWN is used when the classifier cannot decide (degraded
    prints, ambiguous topology).
    """

    PLAIN_ARCH = "plain_arch"
    TENTED_ARCH = "tented_arch"
    LOOP = "loop"
    WHORL = "whorl"
    UNKNOWN = "unknown"


class FacingDirection(Enum):
    """For loops, the direction the recurve opens.

    In a loop, the core sits on the inner side of the recurve and the
    delta on the outer side. The facing direction is the side the
    recurve opens to:
      - LEFT  : recurve opens leftward (delta on the right of core)
      - RIGHT : recurve opens rightward (delta on the left of core)
    Without knowing the hand (left/right), we cannot map this to
    radial/ulnar — only to the visual slant.
    """

    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


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
class PatternClassification:
    """Output of the pattern classifier.

    ``confidence`` ∈ [0, 1] — high means the classifier is sure
    (curvature and singularity count both support the same pattern).

    ``mean_curvature`` and ``singularity_count`` are exposed for
    inspection and debugging — they are the evidence the classifier
    used.
    """

    pattern_type: PatternType
    confidence: float
    mean_curvature: float
    singularity_count: tuple[int, int]
    facing: FacingDirection = FacingDirection.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)


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
      - Perito report: ``pattern_classification``, ``cores``, ``deltas``,
        ``quality_zones``
      - Perito UI: ``minutiae`` with colour by ``zone``, size by ``confidence``
    """

    minutiae: list[ValidatedMinutia]
    cores: list[Singularity]
    deltas: list[Singularity]
    pattern_classification: PatternClassification
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

    @property
    def pattern_type(self) -> PatternType:
        return self.pattern_classification.pattern_type


__all__ = [
    "DetectionResult",
    "FacingDirection",
    "PatternClassification",
    "PatternType",
    "QualityZone",
    "Singularity",
    "SingularityKind",
    "ValidatedMinutia",
    "Zone",
]

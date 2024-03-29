from typing import Any

from src.fingerprint_comparison.domain.entities.fingerprint import Fingerprint
from src.fingerprint_comparison.domain.types import Image

type BestScore = float
type Filename = str
type KeyPoints = list[Any]
type MatchPoints = Any

FingerprintComparisonStrategyComparisonResult = (
    BestScore,
    Image,
    Filename,
    KeyPoints,
    KeyPoints,
    MatchPoints,
)


class FingerprintComparisonStrategy:
    def compare[
        T
    ](
        self, sample_image: Fingerprint, fingerprint_image: Fingerprint
    ) -> FingerprintComparisonStrategyComparisonResult:
        raise NotImplementedError("Subclasses must implement compare method")

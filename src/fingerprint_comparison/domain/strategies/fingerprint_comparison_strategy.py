from typing import Any

type BestScore = int
type Image[T] = T
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
        self, sample_image: Image[T], fingerprint_image: Image[T]
    ) -> FingerprintComparisonStrategyComparisonResult:
        raise NotImplementedError("Subclasses must implement compare method")

import numpy as np

from src.fingerprint.domain.entities.minutiae import MinutiaeFeature
from src.fingerprint.domain.repositories.minutiae_extractor_repository import (
    MinutiaeExtractorRepository,
)


class MinutiaeExtractorService:
    def __init__(self, extractor_repository: MinutiaeExtractorRepository):
        self.extractor_repository = extractor_repository

    def extract_features(
        self, img: np.ndarray, spurious_threshold: int
    ) -> tuple[list[MinutiaeFeature], list[MinutiaeFeature]]:
        return self.extractor_repository.extract_minutiae_features(
            img, spurious_threshold
        )

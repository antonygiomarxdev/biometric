import numpy as np

from src.fingerprint.infrastructure.opencv.fingerprint_image_enhancer_impl import (
    FingerprintImageEnhancer,
)
from src.fingerprint.infrastructure.opencv.fingerprint_minutiae_extractor_impl import (
    MinutiaeExtractorImpl,
)


class ExtractMinutiaeUseCase:
    def __init__(
        self,
        minutiae_extractor_service: MinutiaeExtractorImpl,
        enhancer: FingerprintImageEnhancer,
    ):
        self.minutiae_extractor_service: MinutiaeExtractorImpl = (
            minutiae_extractor_service
        )
        self.enhancer: FingerprintImageEnhancer = enhancer

    def execute(
        self, img: np.ndarray, spurious_threshold: int = 10
    ) -> tuple[list, list]:
        # Mejora la imagen primero
        enhanced_img: np.ndarray = self.enhancer.enhance(img)

        # Luego, extrae las minutiae de la imagen mejorada
        return self.minutiae_extractor_service.extract_minutiae_features(
            enhanced_img, spurious_threshold
        )

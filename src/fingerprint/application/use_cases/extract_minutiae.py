import numpy as np

from src.fingerprint.domain.entities.minutiae import Minutiae

from src.fingerprint.infrastructure.opencv.fingerprint_image_enhancer_impl import (
    FingerprintImageEnhancerImpl,
)
from src.fingerprint.infrastructure.opencv.fingerprint_minutiae_extractor_impl import (
    FingerprintMinutiaeExtractorImpl,
)


class ExtractMinutiaeUseCase:
    """Use case to obtain minutiae from a fingerprint image."""

    def __init__(
        self,
        enhancer: FingerprintImageEnhancerImpl,
        extractor: FingerprintMinutiaeExtractorImpl,
    ) -> None:
        self.enhancer = enhancer
        self.extractor = extractor

    def execute(self, img: np.ndarray) -> list[Minutiae]:
        """Enhance the image and extract its minutiae."""

        enhanced_img = self.enhancer.enhance(img)
        skeleton = self.enhancer.skeletonize(enhanced_img)
        return self.extractor.extract_minutiae(skeleton)

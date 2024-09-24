import numpy as np

from src.fingerprint.application.services.minutiae_extractor_service import (
    MinutiaeExtractorService,
)
from src.fingerprint.infrastructure.opencv.fingerprint_minutiae_extractor_impl import (
    MinutiaeExtractorImpl,
)


def test_extract_minutiae():
    # Carga una imagen de ejemplo
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    # Crea la implementación del repositorio
    extractor_impl = MinutiaeExtractorImpl()

    # Crea el servicio
    minutiae_service = MinutiaeExtractorService(extractor_impl)

    # Ejecuta la extracción de características
    features_term, features_bif = minutiae_service.extract_features(img, 10)

    assert isinstance(features_term, list)
    assert isinstance(features_bif, list)

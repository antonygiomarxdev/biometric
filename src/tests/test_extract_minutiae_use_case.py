import os
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.fingerprint.application.use_cases.extract_minutiae import ExtractMinutiaeUseCase
from src.fingerprint.infrastructure.opencv import (
    FingerprintImageEnhancerImpl,
    FingerprintMinutiaeExtractorImpl,
)


def test_extract_minutiae_returns_list():
    img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
    enhancer = FingerprintImageEnhancerImpl()
    extractor = FingerprintMinutiaeExtractorImpl()
    use_case = ExtractMinutiaeUseCase(enhancer, extractor)

    minutiae = use_case.execute(img)

    assert isinstance(minutiae, list)

"""Real SOCOFing image fixtures for RAG matching tests.

Mark real-image tests with @pytest.mark.slow since the Gabor
enhancement pipeline takes seconds per image.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.config import test_config

SOCOFING_ROOT = test_config.socofing_real


def load_grayscale(path: Path) -> np.ndarray:
    """Load a grayscale image."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


@pytest.fixture
def socofing_image_path() -> Path:
    """Path to a single SOCOFing Real image."""
    return SOCOFING_ROOT / "100__M_Left_index_finger.BMP"


@pytest.fixture
def socofing_5_real_images() -> list[Path]:
    """List of 5 distinct real images for E2E enrollment."""
    if not SOCOFING_ROOT.exists():
        pytest.skip(f"SOCOFING_ROOT not found: {SOCOFING_ROOT}")
    return sorted(SOCOFING_ROOT.glob("*.BMP"))[:5]

"""Tests for AI fingerprint enhancement (U-Net MobileNetV2)."""

import numpy as np
import pytest

from src.processing.enhancers.ai import EnhancementEnhancer
from src.ai.enhancement import EnhancementProcessor


class MockModelManager:
    """Minimal ModelManager stub for enhancement tests."""

    def __init__(self) -> None:
        self._call_count = 0

    def run_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Return a dummy enhanced output of the same spatial shape."""
        self._call_count += 1
        # Simulate a (1, 1, H, W) output from the ONNX model
        batch, channels, height, width = image.shape
        # Dummy enhancement: slight blur of the input
        dummy = np.random.uniform(0.2, 0.8, (1, 1, height, width)).astype(np.float32)
        return dummy


@pytest.fixture
def mock_model_manager() -> MockModelManager:
    return MockModelManager()


@pytest.fixture
def sample_fingerprint() -> np.ndarray:
    """256x256 synthetic fingerprint-like image."""
    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(10, 250, 12):
        img[i:i + 4, 20:240] = 200
    noise = np.random.randint(0, 40, (256, 256), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ── EnhancementEnhancer tests ───────────────────────────────────────────

class TestEnhancementEnhancer:

    def test_enhance_output_shape(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """Enhance() returns ndarray with same H,W as input."""
        enhancer = EnhancementEnhancer(mock_model_manager)
        result = enhancer.enhance(sample_fingerprint)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_fingerprint.shape
        assert result.dtype == np.uint8

    def test_enhance_output_range(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """Enhancement output pixel values are in valid uint8 range (0-255)."""
        enhancer = EnhancementEnhancer(mock_model_manager)
        result = enhancer.enhance(sample_fingerprint)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_enhance_small_image(
        self,
        mock_model_manager: MockModelManager,
    ) -> None:
        """Small image (<64x64) does not crash."""
        small = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        enhancer = EnhancementEnhancer(mock_model_manager)
        result = enhancer.enhance(small)
        assert isinstance(result, np.ndarray)
        assert result.shape == (32, 32)
        assert result.dtype == np.uint8


# ── EnhancementProcessor unit tests ─────────────────────────────────────

class TestEnhancementProcessor:

    def test_preprocess_output_shape(self) -> None:
        """Preprocess output shape is (1, 1, 512, 512), dtype float32."""
        processor = EnhancementProcessor()
        img = np.random.randint(0, 256, (300, 400), dtype=np.uint8)
        tensor = processor.preprocess(img)
        assert tensor.shape == (1, 1, 512, 512)
        assert tensor.dtype == np.float32
        # Values in [0, 1]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

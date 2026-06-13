"""Tests for AI fingerprint segmentation (U-Net)."""

import numpy as np
import pytest

from src.processing.enhancers.ai import SegmentationEnhancer
from src.ai.segmentation import SegmentationProcessor


class MockModelManager:
    """Minimal ModelManager stub for segmentation tests."""

    def __init__(self) -> None:
        self._call_count = 0

    def run_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Return a dummy segmentation mask of the same spatial shape."""
        self._call_count += 1
        # Simulate a (1, 1, H, W) output from the ONNX model
        batch, channels, height, width = image.shape
        # Create a centred blob as the predicted mask
        mask = np.zeros((height, width), dtype=np.float32)
        cy, cx = height // 2, width // 2
        radius = min(height, width) // 4
        yy, xx = np.ogrid[:height, :width]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2] = 0.85
        return mask[np.newaxis, np.newaxis, :, :]


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


# ── SegmentationEnhancer tests ──────────────────────────────────────────

class TestSegmentationEnhancer:

    def test_enhance_returns_valid_shape(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """Enhance() returns ndarray with same H,W as input (within model bounds)."""
        enhancer = SegmentationEnhancer(mock_model_manager)
        result = enhancer.enhance(sample_fingerprint)
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_fingerprint.shape
        assert result.dtype == np.uint8

    def test_enhance_invalid_image(
        self,
        mock_model_manager: MockModelManager,
    ) -> None:
        """Blank image does not crash — returns zero mask result."""
        blank = np.zeros((256, 256), dtype=np.uint8)
        enhancer = SegmentationEnhancer(mock_model_manager)
        # Should not raise — model produces mask over background
        result = enhancer.enhance(blank)
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256)


# ── SegmentationProcessor unit tests ────────────────────────────────────

class TestSegmentationProcessor:

    def test_preprocess_output_shape(self) -> None:
        """Preprocess produces (1, 1, 512, 512) float32 tensor."""
        processor = SegmentationProcessor()
        img = np.random.randint(0, 256, (300, 400), dtype=np.uint8)
        tensor = processor.preprocess(img)
        assert tensor.shape == (1, 1, 512, 512)
        assert tensor.dtype == np.float32
        # Values in [0, 1]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_postprocess_binary_mask(self) -> None:
        """Postprocess returns uint8 binary (0 or 255) at original size."""
        processor = SegmentationProcessor()
        original_shape = (256, 256)
        # Dummy model output (1, 1, 512, 512) with random floats
        dummy_output = np.random.uniform(0, 1, (1, 1, 512, 512)).astype(np.float32)
        mask = processor.postprocess(dummy_output, original_shape)
        assert mask.shape == original_shape
        assert mask.dtype == np.uint8
        # Only binary values
        unique = set(np.unique(mask))
        assert unique.issubset({0, 255}), f"Unexpected values: {unique}"

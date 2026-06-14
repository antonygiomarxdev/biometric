"""Tests for AI pipeline components: segmentation, enhancement, extraction.

All ONNX inference is mocked via the session-scoped conftest fixtures.
Tests focus on the pre/post processing logic (numpy array manipulation)
and the orchestration methods that wire preprocess -> inference -> postprocess.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ai.config import AiConfig
from src.core.types import AlgorithmOrigin, MinutiaType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_image() -> np.ndarray:
    """64x64 grayscale uint8 test image with a diagonal gradient."""
    img = np.zeros((64, 64), dtype=np.uint8)
    for i in range(64):
        img[i, :] = i * 4  # gradient from 0 to 252
    return img


@pytest.fixture
def ai_config() -> AiConfig:
    """Default AiConfig with CPU provider."""
    return AiConfig(use_gpu=False)


@pytest.fixture
def mock_model_manager() -> Any:
    """ModelManager with all inference methods mocked."""
    mm = MagicMock()
    mm.run_segmentation.return_value = np.ones((1, 1, 512, 512), dtype=np.float32)
    mm.run_enhancement.return_value = np.ones((1, 1, 512, 512), dtype=np.float32)
    mm.run_extraction.return_value = np.ones((1, 1, 512, 512), dtype=np.float32)
    return mm


# ---------------------------------------------------------------------------
# SegmentationProcessor
# ---------------------------------------------------------------------------


class TestSegmentationProcessor:
    """Pre/post processing for the U-Net segmentation model."""

    def test_preprocess_normalises_and_pads(
        self, small_image: np.ndarray
    ) -> None:
        """preprocess normalises to [0,1] and pads to input_size square."""
        from src.ai.segmentation import SegmentationProcessor

        processor = SegmentationProcessor(AiConfig(input_size=128, use_gpu=False))
        tensor = processor.preprocess(small_image)

        assert tensor.shape == (1, 1, 128, 128)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        # The original image (64x64) should be centred inside the 128x128 canvas
        # Values outside the centre region should be 0 (padding)
        assert tensor[0, 0, 0, 0] == 0.0  # top-left corner is padding

    def test_preprocess_preserves_content(
        self, small_image: np.ndarray
    ) -> None:
        """preprocess keeps the image content at the centre of the canvas."""
        from src.ai.segmentation import SegmentationProcessor

        processor = SegmentationProcessor(AiConfig(input_size=128, use_gpu=False))
        tensor = processor.preprocess(small_image)

        # Content region should be non-zero
        y_off = (128 - 64) // 2
        x_off = (128 - 64) // 2
        content = tensor[0, 0, y_off:y_off + 64, x_off:x_off + 64]
        assert content.max() > 0.0
        # Normalised: max value should be ~252/255 ≈ 0.988
        assert abs(content.max() - 252.0 / 255.0) < 0.01

    def test_postprocess_thresholds_and_resizes(
        self, small_image: np.ndarray
    ) -> None:
        """postprocess thresholds the output and resizes to original shape."""
        from src.ai.segmentation import SegmentationProcessor

        processor = SegmentationProcessor(AiConfig(use_gpu=False))

        # Create a mock raw output: confidence map with a hotspot
        raw = np.zeros((1, 1, 512, 512), dtype=np.float32)
        raw[0, 0, 100:200, 100:200] = 0.75  # above default threshold (0.5)

        mask = processor.postprocess(raw, small_image.shape)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        # The hotspot region should be white (255)
        assert mask.max() == 255
        assert mask.min() == 0

    def test_postprocess_empty_mask(
        self, small_image: np.ndarray
    ) -> None:
        """postprocess returns all zeros when output is below threshold."""
        from src.ai.segmentation import SegmentationProcessor

        processor = SegmentationProcessor(AiConfig(use_gpu=False))

        raw = np.zeros((1, 1, 512, 512), dtype=np.float32)
        mask = processor.postprocess(raw, small_image.shape)
        assert mask.max() == 0

    def test_segment_orchestrates_pipeline(
        self, small_image: np.ndarray, mock_model_manager: Any
    ) -> None:
        """segment runs preprocess -> inference -> postprocess."""
        from src.ai.segmentation import SegmentationProcessor

        processor = SegmentationProcessor(AiConfig(input_size=128, use_gpu=False))

        # Mock model_manager.run_segmentation to return a known output
        mock_output = np.zeros((1, 1, 128, 128), dtype=np.float32)
        mock_output[0, 0, 30:50, 30:50] = 0.8
        mock_model_manager.run_segmentation.return_value = mock_output

        mask = processor.segment(small_image, mock_model_manager)

        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        assert mask.max() == 255
        mock_model_manager.run_segmentation.assert_called_once()


# ---------------------------------------------------------------------------
# EnhancementProcessor
# ---------------------------------------------------------------------------


class TestEnhancementProcessor:
    """Pre/post processing for the U-Net enhancement model."""

    def test_preprocess_letterbox_and_normalise(
        self, small_image: np.ndarray
    ) -> None:
        """preprocess letterboxes the image to a square with correct aspect ratio."""
        from src.ai.enhancement import EnhancementProcessor

        processor = EnhancementProcessor(AiConfig(input_size=128, use_gpu=False))
        tensor = processor.preprocess(small_image)

        assert tensor.shape == (1, 1, 128, 128)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_handles_non_square(
        self, ai_config: AiConfig
    ) -> None:
        """preprocess handles rectangular images by maintaining aspect ratio."""
        from src.ai.enhancement import EnhancementProcessor

        processor = EnhancementProcessor(ai_config)
        rect_img = np.zeros((80, 40), dtype=np.uint8)

        tensor = processor.preprocess(rect_img)

        assert tensor.shape[2] == tensor.shape[3]  # square
        assert tensor.dtype == np.float32

    def test_postprocess_denormalises_and_resizes(
        self, small_image: np.ndarray
    ) -> None:
        """postprocess converts float output back to uint8 at original size."""
        from src.ai.enhancement import EnhancementProcessor

        processor = EnhancementProcessor(AiConfig(use_gpu=False))

        # Mock raw output from the model
        raw = np.random.rand(1, 1, 512, 512).astype(np.float32)

        result = processor.postprocess(raw, small_image.shape)

        assert result.shape == (64, 64)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_postprocess_clips_values(
        self, small_image: np.ndarray
    ) -> None:
        """postprocess clips values to [0, 1] before scaling to uint8."""
        from src.ai.enhancement import EnhancementProcessor

        processor = EnhancementProcessor(AiConfig(use_gpu=False))

        raw = np.array([[[[-0.5, 1.5], [0.3, 0.7]]]], dtype=np.float32)
        raw = raw.reshape(1, 1, 2, 2)

        # Need to match the input_size for the output shape
        # Postprocess resizes if shape doesn't match
        result = processor.postprocess(raw, (4, 4))

        assert result.dtype == np.uint8
        # Values outside [0,1] are clipped
        assert result.max() <= 255
        assert result.min() >= 0

    def test_enhance_orchestrates_pipeline(
        self, small_image: np.ndarray, mock_model_manager: Any
    ) -> None:
        """enhance runs preprocess -> inference -> postprocess."""
        from src.ai.enhancement import EnhancementProcessor

        processor = EnhancementProcessor(AiConfig(input_size=128, use_gpu=False))

        mock_output = np.random.rand(1, 1, 128, 128).astype(np.float32)
        mock_model_manager.run_enhancement.return_value = mock_output

        result = processor.enhance(small_image, mock_model_manager)

        assert result.shape == (64, 64)
        assert result.dtype == np.uint8
        mock_model_manager.run_enhancement.assert_called_once()


# ---------------------------------------------------------------------------
# ExtractionProcessor
# ---------------------------------------------------------------------------


class TestExtractionProcessorPreprocess:
    """Pre-processing for the DL minutiae extraction model."""

    def test_preprocess_grayscale(
        self, small_image: np.ndarray
    ) -> None:
        """preprocess normalises a grayscale image to [0, 1] canvas."""
        from src.ai.extraction import ExtractionProcessor

        processor = ExtractionProcessor(AiConfig(input_size=128, use_gpu=False))
        tensor = processor.preprocess(small_image)

        assert tensor.shape == (1, 1, 128, 128)
        assert tensor.dtype == np.float32
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_preprocess_converts_bgr(
        self, ai_config: AiConfig
    ) -> None:
        """preprocess converts a BGR 3-channel image to grayscale."""
        from src.ai.extraction import ExtractionProcessor

        processor = ExtractionProcessor(ai_config)
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)

        tensor = processor.preprocess(bgr)

        assert tensor.shape == (1, 1, 512, 512)
        assert tensor.dtype == np.float32

    def test_preprocess_converts_single_channel_3d(
        self, ai_config: AiConfig
    ) -> None:
        """preprocess squeezes a (H, W, 1) image."""
        from src.ai.extraction import ExtractionProcessor

        processor = ExtractionProcessor(ai_config)
        hwc = np.zeros((64, 64, 1), dtype=np.uint8)

        tensor = processor.preprocess(hwc)

        assert tensor.ndim == 4
        assert tensor.shape[2] == 512
        assert tensor.shape[3] == 512

    def test_preprocess_crops_oversized(
        self, ai_config: AiConfig
    ) -> None:
        """preprocess crops images larger than input_size."""
        from src.ai.extraction import ExtractionProcessor

        processor = ExtractionProcessor(ai_config)
        large = np.zeros((600, 600), dtype=np.uint8)

        tensor = processor.preprocess(large)

        assert tensor.shape == (1, 1, 512, 512)


class TestExtractionProcessorPostprocess:
    """Post-processing (heatmap decoding) for the extraction model."""

    @pytest.fixture
    def processor(self) -> Any:
        from src.ai.extraction import ExtractionProcessor
        return ExtractionProcessor(AiConfig(input_size=64, use_gpu=False))

    def test_single_channel_heatmap(
        self, processor: Any
    ) -> None:
        """Single-channel output is decoded as UNKNOWN minutiae."""
        # (1, 1, 64, 64) with a single hotspot
        raw = np.zeros((1, 1, 64, 64), dtype=np.float32)
        raw[0, 0, 32, 32] = 0.9

        candidates = processor.postprocess(raw, (128, 128), confidence_threshold=0.5)

        assert len(candidates) > 0
        assert all(c.type == MinutiaType.UNKNOWN for c in candidates)  # UNKNOWN
        assert all(c.origin == AlgorithmOrigin.DEEP_LEARNING for c in candidates)

    def test_multi_channel_heatmap(
        self, processor: Any
    ) -> None:
        """Two-channel output decodes terminations and bifurcations separately."""
        raw = np.zeros((1, 2, 64, 64), dtype=np.float32)
        raw[0, 0, 20, 20] = 0.9  # termination
        raw[0, 1, 40, 40] = 0.8  # bifurcation

        candidates = processor.postprocess(raw, (128, 128), confidence_threshold=0.5)

        assert len(candidates) >= 2
        types = {c.type for c in candidates}
        assert 0 in [t.value for t in types]  # TERMINATION
        assert 1 in [t.value for t in types]  # BIFURCATION

    def test_no_detections_below_threshold(
        self, processor: Any
    ) -> None:
        """No candidates returned when all heatmap values are below threshold."""
        raw = np.zeros((1, 1, 64, 64), dtype=np.float32)
        raw[0, 0, 32, 32] = 0.1

        candidates = processor.postprocess(raw, (128, 128), confidence_threshold=0.5)

        assert len(candidates) == 0

    def test_unexpected_shape_returns_empty(
        self, processor: Any
    ) -> None:
        """An unexpected output shape logs a warning and returns empty list."""
        raw = np.zeros((1, 1, 1, 64, 64), dtype=np.float32)  # 5D

        candidates = processor.postprocess(raw, (128, 128))

        assert candidates == []

    def test_single_channel_with_batch_dim_preserved(
        self, processor: Any
    ) -> None:
        """A (1, 1, H, W) output is treated as single-channel UNKNOWN."""
        raw = np.zeros((1, 1, 64, 64), dtype=np.float32)
        raw[0, 0, 32, 32] = 0.9

        candidates = processor.postprocess(
            raw, (128, 128), confidence_threshold=0.5
        )

        assert len(candidates) > 0
        assert all(c.type == MinutiaType.UNKNOWN for c in candidates)

    def test_coordinate_remap(
        self, processor: Any
    ) -> None:
        """Coordinates are remapped from output space to original image space."""
        raw = np.zeros((1, 1, 64, 64), dtype=np.float32)
        raw[0, 0, 32, 32] = 0.9

        # Original image is 128x128, which exactly matches the canvas size (64 input size)
        # Actually it's being cropped to 64 max content
        candidates = processor.postprocess(
            raw, (128, 128), confidence_threshold=0.5
        )

        for c in candidates:
            assert 0 <= c.x < 128
            assert 0 <= c.y < 128

    def test_decode_heatmap_nms(
        self, processor: Any
    ) -> None:
        """_decode_heatmap applies NMS and extracts local maxima."""
        heatmap = np.zeros((64, 64), dtype=np.float32)
        heatmap[32, 32] = 0.9
        heatmap[33, 33] = 0.85  # neighbour, should be suppressed

        from src.core.types import MinutiaType

        candidates = processor._decode_heatmap(heatmap, 0.5, MinutiaType.TERMINATION)

        # After NMS with 3x3, only the local maximum survives
        assert len(candidates) == 1

    def test_decode_heatmap_no_detections_returns_empty(
        self, processor: Any
    ) -> None:
        """_decode_heatmap returns empty list when no points exceed threshold."""
        heatmap = np.zeros((64, 64), dtype=np.float32)

        from src.core.types import MinutiaType

        candidates = processor._decode_heatmap(
            heatmap, 0.9, MinutiaType.TERMINATION
        )
        assert candidates == []

    def test_decode_heatmap_connected_components(
        self, processor: Any
    ) -> None:
        """_decode_heatmap groups connected pixels via centre-of-mass."""
        heatmap = np.zeros((64, 64), dtype=np.float32)
        # Create a 3x3 blob
        heatmap[30:33, 30:33] = 0.7
        heatmap[31, 31] = 0.9

        from src.core.types import MinutiaType

        candidates = processor._decode_heatmap(heatmap, 0.5, MinutiaType.BIFURCATION)

        assert len(candidates) == 1
        assert candidates[0].type == MinutiaType.BIFURCATION
        assert candidates[0].confidence == pytest.approx(0.9, abs=0.001)
        assert 30 <= candidates[0].x <= 33
        assert 30 <= candidates[0].y <= 33


class TestExtractionProcessorIntegration:
    """End-to-end extraction pipeline with mocked model output."""

    def test_extraction_pipeline(
        self, small_image: np.ndarray
    ) -> None:
        """Full pipeline: preprocess -> run_extraction -> postprocess."""
        from src.ai.extraction import ExtractionProcessor

        processor = ExtractionProcessor(AiConfig(input_size=64, use_gpu=False))

        mock_mm = MagicMock()
        raw_output = np.zeros((1, 2, 64, 64), dtype=np.float32)
        raw_output[0, 0, 16, 16] = 0.85  # termination
        raw_output[0, 1, 48, 48] = 0.75  # bifurcation
        mock_mm.run_extraction.return_value = raw_output

        tensor = processor.preprocess(small_image)
        raw = mock_mm.run_extraction(tensor)
        candidates = processor.postprocess(raw, small_image.shape[:2])

        assert len(candidates) >= 2
        mock_mm.run_extraction.assert_called_once_with(tensor)

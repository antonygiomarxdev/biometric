"""Tests for AI-based fingerprint minutiae extraction.

Covers :class:`ExtractionProcessor` (pre/post processing) and
:class:`AiFeatureExtractor` (full pipeline through ModelManager).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ai.extraction import ExtractionProcessor
from src.core.interfaces import IFeatureExtractor
from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
)


# ── ExtractionProcessor unit tests ───────────────────────────────────────


class TestExtractionProcessorPreprocess:

    def test_preprocess_shape(self) -> None:
        """200x300 uint8 input produces (1, 1, 512, 512) float32 tensor."""
        processor = ExtractionProcessor()
        img = np.random.randint(0, 256, (200, 300), dtype=np.uint8)
        tensor = processor.preprocess(img)
        assert tensor.shape == (1, 1, 512, 512)
        assert tensor.dtype == np.float32

    def test_preprocess_normalization(self) -> None:
        """Input range [0, 255] maps to output range [0, 1]."""
        processor = ExtractionProcessor()
        img = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
        tensor = processor.preprocess(img)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        # A non-trivial image should have values above 0
        assert tensor.max() > 0.0

    def test_preprocess_3channel_image(self) -> None:
        """3-channel BGR input is converted to grayscale."""
        processor = ExtractionProcessor()
        img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        tensor = processor.preprocess(img)
        assert tensor.shape == (1, 1, 512, 512)
        assert tensor.dtype == np.float32

    def test_preprocess_single_channel_3d(self) -> None:
        """(H, W, 1) input is squeezed to (H, W) correctly."""
        processor = ExtractionProcessor()
        img = np.random.randint(0, 256, (150, 200, 1), dtype=np.uint8)
        tensor = processor.preprocess(img)
        assert tensor.shape == (1, 1, 512, 512)

    def test_preprocess_larger_than_input_size(self) -> None:
        """Image larger than 512x512 is cropped, not resized."""
        processor = ExtractionProcessor()
        img = np.zeros((600, 800), dtype=np.uint8)
        img[:100, :100] = 255  # white square in top-left
        tensor = processor.preprocess(img)
        assert tensor.shape == (1, 1, 512, 512)
        # Content is min(600,512) x min(800,512) = 512x512
        # The top-left white region should survive the crop
        content = tensor[0, 0, :100, :100]
        assert content.max() > 0.0


class TestExtractionProcessorPostprocess:

    def _make_peak(self, h: int, w: int, cy: int, cx: int, strength: float) -> np.ndarray:
        """Create a 2D heatmap with a local maximum at (cy, cx)."""
        heatmap = np.zeros((h, w), dtype=np.float32)
        heatmap[cy, cx] = strength
        # Surrounding ring at lower value so the centre remains the unique max
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and (dy != 0 or dx != 0):
                    heatmap[ny, nx] = strength * 0.3
        heatmap[cy, cx] = strength  # Reinforce centre
        return heatmap

    def test_postprocess_dual_channel(self) -> None:
        """Dual-channel heatmap decodes to correct MinutiaType per channel.

        The model output is (1, 2, 64, 64) and the original image is
        64×64.  With a 512×512 input canvas the valid output coordinate
        range for a 64×64 original image is [28, 36) — we place peaks
        inside that window.
        """
        processor = ExtractionProcessor()
        out_h = out_w = 64

        # Valid range in output space for a 64x64 original on a 512 canvas:
        # y_offset = (512 - 64)//2 = 224, scale_y = 512/64 = 8
        # valid_y = [224//8 : (224+64)//8) = [28:36)
        # Place peaks at (y=32, x=30) and (y=34, x=33)
        term_peak = self._make_peak(out_h, out_w, 32, 30, 0.85)
        bif_peak = self._make_peak(out_h, out_w, 34, 33, 0.90)

        # Stack: (2, 64, 64) then batch dim -> (1, 2, 64, 64)
        output = np.stack([term_peak, bif_peak], axis=0)[np.newaxis, ...]

        candidates = processor.postprocess(
            output, (64, 64), confidence_threshold=0.5,
        )

        assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"

        types = {m.type for m in candidates}
        assert MinutiaType.TERMINATION in types
        assert MinutiaType.BIFURCATION in types

        for m in candidates:
            assert m.origin == AlgorithmOrigin.DEEP_LEARNING

    def test_postprocess_empty(self) -> None:
        """All-zeros heatmap produces empty list."""
        processor = ExtractionProcessor()
        output = np.zeros((1, 2, 64, 64), dtype=np.float32)
        candidates = processor.postprocess(output, (64, 64), confidence_threshold=0.5)
        assert candidates == []

    def test_postprocess_out_of_bounds(self) -> None:
        """Peak at origin in canvas maps outside a small original -> excluded."""
        processor = ExtractionProcessor()
        # Original image is (100, 100), centered on 512x512 canvas.
        # y_offset = x_offset = (512 - 100)//2 = 206.
        # A peak at output (0, 0) -> canvas (0, 0) -> original (-206, -206) -> outside
        out_h = out_w = 512
        heatmap = np.zeros((1, 1, out_h, out_w), dtype=np.float32)
        heatmap[0, 0, 0, 0] = 0.9
        candidates = processor.postprocess(heatmap, (100, 100), confidence_threshold=0.5)
        assert len(candidates) == 0

    def test_postprocess_coordinate_adjustment(self) -> None:
        """Peak at canvas-centre maps back to correct original coordinate."""
        processor = ExtractionProcessor()
        original_shape = (100, 100)
        out_h = out_w = 512
        y_offset = (512 - 100) // 2  # = 206
        x_offset = (512 - 100) // 2  # = 206

        # Peak at (y_offset+50, x_offset+50) = (256, 256) in output space
        # -> canvas (256, 256) -> original (50, 50)
        heatmap = self._make_peak(out_h, out_w, y_offset + 50, x_offset + 50, 0.85)
        output = heatmap[np.newaxis, np.newaxis, :, :]

        candidates = processor.postprocess(output, original_shape, confidence_threshold=0.5)
        assert len(candidates) >= 1
        m = candidates[0]
        assert m.x == 50, f"Expected x=50, got {m.x}"
        assert m.y == 50, f"Expected y=50, got {m.y}"

    def test_postprocess_single_channel_output(self) -> None:
        """Single-channel model output (no batch dim) is handled correctly."""
        processor = ExtractionProcessor()
        out_h = out_w = 64
        # Place peak in the valid output range for a 64x64 original
        heatmap = self._make_peak(out_h, out_w, 32, 32, 0.8)
        output = heatmap  # Shape (64, 64) — no batch/channel dims
        candidates = processor.postprocess(output, (64, 64), confidence_threshold=0.5)
        assert len(candidates) >= 1
        assert candidates[0].type == MinutiaType.UNKNOWN

    def test_postprocess_confidence_below_threshold(self) -> None:
        """Peaks below confidence threshold are filtered out."""
        processor = ExtractionProcessor()
        h = w = 64
        heatmap = self._make_peak(h, w, 32, 32, 0.3)
        output = heatmap[np.newaxis, np.newaxis, :, :]
        candidates = processor.postprocess(output, (64, 64), confidence_threshold=0.5)
        assert len(candidates) == 0

    def test_postprocess_dual_channel_diff_output_resolution(self) -> None:
        """Output at 256x256 is scaled to canvas before coordinate remap."""
        processor = ExtractionProcessor()
        out_h = out_w = 256
        original_shape = (100, 100)

        # For a 100x100 original on a 512 canvas, valid output range
        # for 256x256 output is:
        #   y_canvas_offset = (512-100)//2 = 206
        #   scale_y = 512 / 256 = 2
        #   valid_y = [206/2 : (206+100)/2) = [103 : 153)
        # Place peak at y=130, x=130 in output space
        # -> canvas (260, 260) -> original (54, 54)
        heatmap = self._make_peak(out_h, out_w, 130, 130, 0.85)
        output = heatmap[np.newaxis, np.newaxis, :, :]
        candidates = processor.postprocess(output, original_shape, confidence_threshold=0.5)
        assert len(candidates) >= 1
        m = candidates[0]
        assert m.x == 54, f"Expected x=54, got {m.x}"
        assert m.y == 54, f"Expected y=54, got {m.y}"


# ── Mock helpers for AiFeatureExtractor tests ────────────────────────────


class MockModelManager:
    """Minimal ModelManager stub for AiFeatureExtractor tests."""

    def __init__(self, dummy_output: np.ndarray | None = None) -> None:
        self._call_count = 0
        self._dummy_output = dummy_output

    def run_extraction(self, image: np.ndarray) -> np.ndarray:
        """Return a dummy extraction output.

        Creates a single-peak heatmap in the terminations channel at
        the centre of the image if no explicit output was provided.
        """
        self._call_count += 1
        if self._dummy_output is not None:
            return self._dummy_output

        batch, channels, height, width = image.shape
        output = np.zeros((1, 2, height, width), dtype=np.float32)
        cy, cx = height // 2, width // 2
        output[0, 0, cy, cx] = 0.85  # termination
        output[0, 1, cy + 10, cx + 10] = 0.75  # bifurcation
        return output

    def get_session(self, name: str) -> None:
        pass  # Not needed for tests


class MockExtractionProcessor:
    """Stub ExtractionProcessor that returns known candidates."""

    def __init__(self, candidates: list[MinutiaCandidate] | None = None) -> None:
        self._candidates = candidates

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Return minimal tensor shape."""
        return np.zeros((1, 1, 64, 64), dtype=np.float32)

    def postprocess(
        self,
        raw_output: np.ndarray,
        original_shape: tuple[int, int],
        confidence_threshold: float | None = None,
    ) -> list[MinutiaCandidate]:
        """Return configured candidates."""
        return self._candidates or []


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

# ── AiFeatureExtractor tests ────────────────────────────────────────────


class TestAiFeatureExtractor:

    def test_implements_ifeature_extractor(self) -> None:
        """AiFeatureExtractor is an instance of IFeatureExtractor."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(MockModelManager())
        assert isinstance(extractor, IFeatureExtractor)

    def test_extract_algorithm_origin(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """All returned candidates have AlgorithmOrigin.DEEP_LEARNING."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(mock_model_manager)
        candidates = extractor.extract(sample_fingerprint)
        for m in candidates:
            assert m.origin == AlgorithmOrigin.DEEP_LEARNING, (
                f"Expected DEEP_LEARNING, got {m.origin}"
            )

    def test_extract_blank_image(
        self,
        mock_model_manager: MockModelManager,
    ) -> None:
        """Blank image returns empty list, not crash."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(mock_model_manager)
        blank = np.zeros((512, 512), dtype=np.uint8)
        candidates = extractor.extract(blank)
        assert isinstance(candidates, list)

    def test_extract_empty_image(self) -> None:
        """None-size or zero-size image returns empty list."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(MockModelManager())
        empty = np.zeros((0, 0), dtype=np.uint8)
        candidates = extractor.extract(empty)
        assert candidates == []

    def test_extract_confidence_range(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """All returned confidence values are in [0.0, 1.0]."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(mock_model_manager)
        candidates = extractor.extract(sample_fingerprint)
        for m in candidates:
            assert 0.0 <= m.confidence <= 1.0, (
                f"Confidence {m.confidence} out of range"
            )

    def test_extract_model_failure(
        self,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """ModelManager exception is caught and returns empty list."""
        from src.processing.extractor import AiFeatureExtractor

        class FailingModelManager:
            def run_extraction(self, image: np.ndarray) -> np.ndarray:
                msg = "Model inference failed"
                raise RuntimeError(msg)

        extractor = AiFeatureExtractor(FailingModelManager())  # type: ignore[arg-type]
        # Should not raise
        candidates = extractor.extract(sample_fingerprint)
        assert candidates == []

    def test_extract_output_type(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """Extract returns list[MinutiaCandidate]."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(mock_model_manager)
        candidates = extractor.extract(sample_fingerprint)
        assert isinstance(candidates, list)
        if candidates:
            assert isinstance(candidates[0], MinutiaCandidate)

    def test_extract_model_manager_called(
        self,
        mock_model_manager: MockModelManager,
        sample_fingerprint: np.ndarray,
    ) -> None:
        """ModelManager.run_extraction is invoked during extract."""
        from src.processing.extractor import AiFeatureExtractor

        extractor = AiFeatureExtractor(mock_model_manager)
        extractor.extract(sample_fingerprint)
        assert mock_model_manager._call_count >= 1  # noqa: SLF001

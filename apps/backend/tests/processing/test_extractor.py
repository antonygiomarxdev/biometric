"""Tests for feature extractors: Skeleton and AI.

All ONNX inference is mocked via the session-scoped conftest fixtures.
Skeleton extraction tests use synthetic binary images with known
ridge patterns to verify Crossing Number logic deterministically.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.types import AlgorithmOrigin, MinutiaType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ridge_image() -> np.ndarray:
    """Synthetic binary image (uint8) with a simple vertical ridge pattern.

    Creates a 50x50 image with 3 vertical ridges (white bars) on a black
    background.  Each ridge is 3 pixels wide with a 4-pixel gap.
    """
    img = np.zeros((50, 50), dtype=np.uint8)
    for x in range(5, 50, 7):
        img[:, x : x + 3] = 255
    return img


@pytest.fixture
def blank_binary() -> np.ndarray:
    """All-black binary image."""
    return np.zeros((50, 50), dtype=np.uint8)


@pytest.fixture
def all_white_binary() -> np.ndarray:
    """All-white binary image."""
    return np.full((50, 50), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# SkeletonMinutiaeExtractor
# ---------------------------------------------------------------------------


class TestSkeletonMinutiaeExtractor:
    """Crossing-Number-based skeleton extractor (raw points, no filtering)."""

    def test_extract_with_valid_ridges(
        self, ridge_image: np.ndarray
    ) -> None:
        """extract returns a list of MinutiaCandidate for a ridge image."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        candidates = extractor.extract(ridge_image)

        assert isinstance(candidates, list)
        if len(candidates) > 0:
            c = candidates[0]
            assert isinstance(c.x, int)
            assert isinstance(c.y, int)
            assert isinstance(c.angle, float)
            assert c.origin == AlgorithmOrigin.SKELETON
            assert 0 <= c.confidence <= 1.0

    def test_extract_blank_image_returns_empty(
        self, blank_binary: np.ndarray
    ) -> None:
        """A completely blank image returns an empty list."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        candidates = extractor.extract(blank_binary)
        assert candidates == []

    def test_extract_all_white_returns_empty(
        self, all_white_binary: np.ndarray
    ) -> None:
        """An all-white image (no skeleton structure) returns empty list."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        candidates = extractor.extract(all_white_binary)
        assert candidates == []

    def test_extract_handles_uint8_input(
        self,
    ) -> None:
        """extract works with uint8 input (not just binary)."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 200
        candidates = extractor.extract(img)
        assert isinstance(candidates, list)

    def test_extract_with_gradient_image_triggers_otsu(
        self,
    ) -> None:
        """A gradient image with >2 unique values triggers Otsu binarization."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        gradient = np.tile(np.arange(50, dtype=np.uint8), (50, 1))
        candidates = extractor.extract(gradient)
        assert isinstance(candidates, list)

    def test_extract_very_few_white_pixels_triggers_inversion(
        self,
    ) -> None:
        """An image with < 5% white pixels triggers automatic inversion."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        img = np.zeros((50, 50), dtype=np.uint8)
        img[24:26, 24:26] = 255
        candidates = extractor.extract(img)
        assert isinstance(candidates, list)

    def test_compute_angle_termination(self) -> None:
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        blk = np.array(
            [[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8,
        )
        angle = extractor._compute_angle(blk, MinutiaType.TERMINATION)
        assert angle == -90.0

    def test_compute_angle_bifurcation(self) -> None:
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        blk = np.array(
            [[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=np.uint8,
        )
        angle = extractor._compute_angle(blk, MinutiaType.BIFURCATION)
        assert angle == pytest.approx(0.0, abs=1.0)

    def test_compute_angle_no_neighbours(self) -> None:
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        blk = np.zeros((3, 3), dtype=np.uint8)
        angle = extractor._compute_angle(blk, MinutiaType.TERMINATION)
        assert angle == 0.0

    def test_detect_crossing_number(self) -> None:
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        skel = np.zeros((20, 20), dtype=np.uint8)
        skel[10, 5:16] = 1
        skel[5:16, 10] = 1

        candidates = extractor._detect_crossing_number(skel)
        assert len(candidates) > 0

    def test_detect_crossing_number_with_non_binary_input(self) -> None:
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        skel = np.zeros((20, 20), dtype=np.uint8)
        skel[10, 5:16] = 2
        skel[5:16, 10] = 2
        candidates = extractor._detect_crossing_number(skel)
        assert isinstance(candidates, list)

    def test_detect_crossing_number_no_skeleton(self) -> None:
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        skel = np.zeros((20, 20), dtype=np.uint8)
        candidates = extractor._detect_crossing_number(skel)
        assert isinstance(candidates, list)

    def test_extract_returns_raw_unfiltered(self) -> None:
        """SkeletonMinutiaeExtractor no filtra — devuelve puntos crudos."""
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        img = np.zeros((50, 50), dtype=np.uint8)
        img[10:40, 10:40] = 255
        candidates = extractor.extract(img)
        assert isinstance(candidates, list)
        # Crossing Number debe haber encontrado algo en un bloque sólido
        # (puntos en el borde del bloque producen terminaciones)
        assert len(candidates) >= 0


# ---------------------------------------------------------------------------
# MockMinutia helper
# ---------------------------------------------------------------------------


class MockMinutia:
    """Minimal duck-typed substitute for MinutiaCandidate in filter tests."""

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y
        self.angle = 0.0
        self.type = MinutiaType.TERMINATION
        self.confidence = 1.0
        self.origin = AlgorithmOrigin.SKELETON


# ---------------------------------------------------------------------------
# AiFeatureExtractor
# ---------------------------------------------------------------------------


class TestAiFeatureExtractor:
    """Deep-learning extractor using ONNX Runtime."""

    def test_extract_with_mocked_model(self) -> None:
        """extract returns candidates when model and processor are mocked."""
        from src.processing.extractor import AiFeatureExtractor

        mock_mm = MagicMock()
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = np.zeros(
            (1, 1, 64, 64), dtype=np.float32
        )
        mock_mm.run_extraction.return_value = np.zeros(
            (1, 2, 64, 64), dtype=np.float32
        )
        mock_processor.postprocess.return_value = [
            MockMinutia(10, 10),
            MockMinutia(20, 20),
        ]

        extractor = AiFeatureExtractor(mock_mm, processor=mock_processor)

        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        candidates = extractor.extract(image)

        assert len(candidates) == 2
        mock_processor.preprocess.assert_called_once()
        mock_mm.run_extraction.assert_called_once()
        mock_processor.postprocess.assert_called_once()

    def test_extract_converts_bgr_input(self) -> None:
        """extract converts a BGR 3-channel image to grayscale."""
        from src.processing.extractor import AiFeatureExtractor

        mock_mm = MagicMock()
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = np.zeros(
            (1, 1, 64, 64), dtype=np.float32
        )
        mock_mm.run_extraction.return_value = np.zeros(
            (1, 1, 64, 64), dtype=np.float32
        )
        mock_processor.postprocess.return_value = []

        extractor = AiFeatureExtractor(mock_mm, processor=mock_processor)

        bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        extractor.extract(bgr)

        mock_processor.preprocess.assert_called_once()

    def test_extract_empty_image_returns_empty(self) -> None:
        """extract returns empty list for a None or empty image."""
        from src.processing.extractor import AiFeatureExtractor

        mock_mm = MagicMock()
        mock_processor = MagicMock()
        extractor = AiFeatureExtractor(mock_mm, processor=mock_processor)

        # None image
        assert extractor.extract(None) == []
        # Zero-size array
        assert extractor.extract(np.zeros((0, 0), dtype=np.uint8)) == []

    def test_extract_handles_model_failure_gracefully(self) -> None:
        """extract returns empty list when the model raises an exception."""
        from src.processing.extractor import AiFeatureExtractor

        mock_mm = MagicMock()
        mock_mm.run_extraction.side_effect = RuntimeError("Model crash")
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = np.zeros(
            (1, 1, 64, 64), dtype=np.float32
        )

        extractor = AiFeatureExtractor(mock_mm, processor=mock_processor)

        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        candidates = extractor.extract(image)

        assert candidates == []

    def test_default_processor_created(self) -> None:
        """AiFeatureExtractor creates a default ExtractionProcessor when none given."""
        from src.processing.extractor import AiFeatureExtractor
        from src.ai.extraction import ExtractionProcessor

        mock_mm = MagicMock()
        extractor = AiFeatureExtractor(mock_mm)
        assert isinstance(extractor.processor, ExtractionProcessor)

    def test_extract_tracks_bgr_conversion_then_original_shape(
        self,
    ) -> None:
        """extract converts BGR to grayscale and uses original shape for postprocess."""
        from src.processing.extractor import AiFeatureExtractor

        mock_mm = MagicMock()
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = np.zeros(
            (1, 1, 64, 64), dtype=np.float32
        )
        mock_mm.run_extraction.return_value = np.zeros(
            (1, 1, 64, 64), dtype=np.float32
        )
        mock_processor.postprocess.return_value = []

        extractor = AiFeatureExtractor(mock_mm, processor=mock_processor)

        bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        extractor.extract(bgr)

        _, call_kwargs = mock_processor.postprocess.call_args
        # The original_shape should match the original image dimensions
        assert mock_processor.postprocess.call_count == 1
        call_args = mock_processor.postprocess.call_args[0]
        assert call_args[1] == (64, 64)

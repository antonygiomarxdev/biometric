"""Tests for CPU-based fingerprint enhancer.

This module tests the deterministic CPU enhancement pipeline
(normalization, orientation, frequency estimation, Gabor filtering)
with synthetic images.  No ONNX models are involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.processing.enhancers.base import EnhancerConfig
from src.processing.enhancers.cpu import CpuEnhancer


@pytest.fixture
def enhancer() -> CpuEnhancer:
    """Default CPU enhancer with standard configuration."""
    return CpuEnhancer(EnhancerConfig())


@pytest.fixture
def ridge_image() -> np.ndarray:
    """60x60 synthetic image with a vertical ridge pattern."""
    img = np.zeros((60, 60), dtype=np.uint8)
    for x in range(5, 60, 8):
        img[:, x : x + 3] = 200
    return img


class TestCpuEnhancer:
    """CPU enhancer pipeline with deterministic image processing."""

    def test_enhance_returns_uint8(
        self, enhancer: CpuEnhancer, ridge_image: np.ndarray
    ) -> None:
        """enhance returns a uint8 image."""
        result = enhancer.enhance(ridge_image, resize=False)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_enhance_resizes_when_requested(
        self, enhancer: CpuEnhancer, ridge_image: np.ndarray
    ) -> None:
        """enhance resizes to 350px height when resize=True."""
        result = enhancer.enhance(ridge_image, resize=True)
        assert result.shape[0] == 350

    def test_enhance_no_resize_preserves_shape(
        self, enhancer: CpuEnhancer, ridge_image: np.ndarray
    ) -> None:
        """enhance keeps original dimensions when resize=False."""
        result = enhancer.enhance(ridge_image, resize=False)
        assert result.shape == ridge_image.shape

    def test_enhance_handles_uniform_image(
        self, enhancer: CpuEnhancer
    ) -> None:
        """enhance handles a uniform (flat) image without crashing."""
        img = np.full((40, 40), 128, dtype=np.uint8)
        result = enhancer.enhance(img, resize=False)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8

    def test_normalize_zero_std(
        self, enhancer: CpuEnhancer
    ) -> None:
        """_normalize returns the original image when std is zero."""
        img = np.full((20, 20), 100, dtype=np.uint8)
        result = enhancer._normalize(img)
        assert np.array_equal(result, img)

    def test_normalize_standardizes(
        self, enhancer: CpuEnhancer
    ) -> None:
        """_normalize produces zero-mean, unit-variance output."""
        img = np.random.randint(0, 256, (30, 30)).astype(np.uint8)
        result = enhancer._normalize(img)
        assert abs(result.mean()) < 1.0
        assert abs(result.std() - 1.0) < 0.1

    def test_ridge_orient_returns_orientation_map(
        self, enhancer: CpuEnhancer, ridge_image: np.ndarray
    ) -> None:
        """_ridge_orient returns an orientation field."""
        norm = enhancer._normalize(ridge_image)
        orient = enhancer._ridge_orient(norm)
        assert orient.shape == ridge_image.shape
        assert orient.min() >= 0
        assert orient.max() <= np.pi

    def test_ridge_freq_returns_positive_float(
        self, enhancer: CpuEnhancer, ridge_image: np.ndarray
    ) -> None:
        """_ridge_freq returns a positive frequency value."""
        norm = enhancer._normalize(ridge_image)
        orient = enhancer._ridge_orient(norm)
        freq = enhancer._ridge_freq(norm, orient)
        assert isinstance(freq, float)
        assert freq > 0
        assert np.isfinite(freq)

    def test_ridge_filter_returns_binary(
        self, enhancer: CpuEnhancer, ridge_image: np.ndarray
    ) -> None:
        """_ridge_filter returns a boolean (binary) image."""
        norm = enhancer._normalize(ridge_image)
        orient = enhancer._ridge_orient(norm)
        freq = enhancer._ridge_freq(norm, orient)
        binary = enhancer._ridge_filter(norm, orient, freq)
        assert binary.dtype == np.bool_

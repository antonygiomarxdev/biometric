"""
Tests for pre-processing hooks (QualityMasker, BinarizationHook).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.processing.pre_hooks import QualityMasker, BinarizationHook
from src.core.interfaces import IPreProcessor


class TestQualityMasker:
    """QualityMasker should mask low-variance (blank/scar) regions."""

    def test_implements_protocol(self) -> None:
        assert isinstance(QualityMasker(), IPreProcessor)

    def test_blank_image_returns_all_false(self) -> None:
        img = np.zeros((64, 64), dtype=np.uint8)
        result = QualityMasker(block_size=16).process(img)
        assert result.quality_mask is not None
        assert result.quality_mask.sum() == 0

    def test_full_ridge_image_returns_all_true(self) -> None:
        img = np.tile(np.arange(64, dtype=np.uint8).reshape(1, 64), (64, 1))
        result = QualityMasker(block_size=16, min_variance=1.0).process(img)
        assert result.quality_mask is not None
        assert result.quality_mask.sum() > 0

    def test_mask_shape_matches_image(self) -> None:
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = QualityMasker().process(img)
        assert result.quality_mask is not None
        assert result.quality_mask.shape == img.shape

    def test_passthrough_image_unchanged(self) -> None:
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = QualityMasker().process(img)
        np.testing.assert_array_equal(result.image, img)


class TestBinarizationHook:
    """BinarizationHook should produce a clean binary image."""

    def test_implements_protocol(self) -> None:
        assert isinstance(BinarizationHook(), IPreProcessor)

    def test_binary_output_has_two_values(self) -> None:
        img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = BinarizationHook().process(img)
        unique = np.unique(result.image)
        assert len(unique) <= 2

    def test_invert_flag_flips_output(self) -> None:
        img = np.ones((20, 20), dtype=np.uint8) * 200
        not_inverted = BinarizationHook(invert=False).process(img)
        inverted = BinarizationHook(invert=True).process(img)
        assert not_inverted.image.mean() != inverted.image.mean()

    def test_grayscale_conversion_from_color(self) -> None:
        rgb = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        result = BinarizationHook().process(rgb)
        assert len(result.image.shape) == 2

    def test_quality_mask_is_none(self) -> None:
        img = np.zeros((10, 10), dtype=np.uint8)
        result = BinarizationHook().process(img)
        assert result.quality_mask is None

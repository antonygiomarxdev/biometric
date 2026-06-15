"""
Tests for pre-processing hooks (BinarizationHook, OrientationFieldAnalyzer,
QualityMasker, SingularityDetector). All hooks use the v2
PipelineContext API.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.interfaces import IPipelineStep, PipelineContext
from src.processing.pre_hooks import (
    BinarizationHook,
    OrientationFieldAnalyzer,
    QualityMasker,
    SingularityDetector,
)


# ---------------------------------------------------------------------------
# BinarizationHook
# ---------------------------------------------------------------------------


class TestBinarizationHook:
    def test_implements_protocol(self) -> None:
        assert isinstance(BinarizationHook(), IPipelineStep)

    def test_binary_output_has_two_values(self) -> None:
        ctx = PipelineContext(raw_image=np.random.randint(0, 256, (50, 50), dtype=np.uint8))
        BinarizationHook().process(ctx)
        assert ctx.preprocessed_image is not None
        unique = np.unique(ctx.preprocessed_image)
        assert len(unique) <= 2

    def test_invert_flag_flips_output(self) -> None:
        img = np.ones((20, 20), dtype=np.uint8) * 200
        ctx_a = PipelineContext(raw_image=img)
        BinarizationHook(invert=False).process(ctx_a)
        ctx_b = PipelineContext(raw_image=img)
        BinarizationHook(invert=True).process(ctx_b)
        assert ctx_a.preprocessed_image.mean() != ctx_b.preprocessed_image.mean()

    def test_grayscale_conversion_from_color(self) -> None:
        rgb = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
        ctx = PipelineContext(raw_image=rgb)
        BinarizationHook().process(ctx)
        assert ctx.preprocessed_image.ndim == 2


# ---------------------------------------------------------------------------
# OrientationFieldAnalyzer (Hong-Wan-Jain 1998)
# ---------------------------------------------------------------------------


class TestOrientationFieldAnalyzer:
    def test_implements_protocol(self) -> None:
        assert isinstance(OrientationFieldAnalyzer(), IPipelineStep)

    def test_blank_image_mostly_invalid(self) -> None:
        ctx = PipelineContext(raw_image=np.zeros((64, 64), dtype=np.uint8))
        OrientationFieldAnalyzer(block_size=16).process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.sum() == 0

    def test_uniform_horizontal_ridges_pass(self) -> None:
        img = np.tile(np.arange(64, dtype=np.uint8).reshape(1, 64), (64, 1))
        ctx = PipelineContext(raw_image=img)
        OrientationFieldAnalyzer(block_size=16, coherence_threshold=0.1).process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.sum() > 0

    def test_mask_shape_matches_image(self) -> None:
        img = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        ctx = PipelineContext(raw_image=img)
        OrientationFieldAnalyzer().process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.shape == img.shape

    def test_orientation_field_shape(self) -> None:
        ctx = PipelineContext(raw_image=np.random.randint(0, 256, (64, 64), dtype=np.uint8))
        OrientationFieldAnalyzer(block_size=16).process(ctx)
        assert ctx.orientation_field is not None
        assert ctx.orientation_field.shape == (4, 4)

    def test_handles_color_image(self) -> None:
        rgb = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        ctx = PipelineContext(raw_image=rgb)
        OrientationFieldAnalyzer(block_size=16).process(ctx)
        assert ctx.preprocessed_image.ndim == 2
        assert ctx.preprocessed_image.shape == (32, 32)


# ---------------------------------------------------------------------------
# QualityMasker (legacy)
# ---------------------------------------------------------------------------


class TestQualityMasker:
    def test_blank_image_returns_all_false(self) -> None:
        ctx = PipelineContext(raw_image=np.zeros((64, 64), dtype=np.uint8))
        QualityMasker(block_size=16).process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.sum() == 0

    def test_full_ridge_image_returns_all_true(self) -> None:
        img = np.tile(np.arange(64, dtype=np.uint8).reshape(1, 64), (64, 1))
        ctx = PipelineContext(raw_image=img)
        QualityMasker(block_size=16, min_variance=1.0).process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.sum() > 0

    def test_mask_shape_matches_image(self) -> None:
        ctx = PipelineContext(raw_image=np.random.randint(0, 256, (100, 100), dtype=np.uint8))
        QualityMasker().process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.shape == (100, 100)


# ---------------------------------------------------------------------------
# SingularityDetector
# ---------------------------------------------------------------------------


class TestSingularityDetector:
    def test_implements_protocol(self) -> None:
        assert isinstance(SingularityDetector(), IPipelineStep)

    def test_no_orientation_passes_through(self) -> None:
        ctx = PipelineContext(raw_image=np.zeros((64, 64), dtype=np.uint8))
        SingularityDetector().process(ctx)
        assert ctx.quality_mask is not None
        assert ctx.quality_mask.all()

    def test_combines_with_existing_mask(self) -> None:
        """If another hook already set quality_mask, the detector intersects it."""
        ctx = PipelineContext(raw_image=np.zeros((64, 64), dtype=np.uint8))
        ctx.quality_mask = np.zeros((64, 64), dtype=bool)
        # Provide an orientation field so the detector runs
        ctx.orientation_field = np.zeros((4, 4), dtype=np.float32)
        ctx.coherence_field = np.ones((4, 4), dtype=np.float32)
        SingularityDetector(roi_radius=10).process(ctx)
        # Intersection of two all-False masks is still all-False
        assert ctx.quality_mask is not None
        assert not ctx.quality_mask.any()

    def test_writes_roi_mask(self) -> None:
        """roi_mask is always populated (all-True if no Core found)."""
        ctx = PipelineContext(raw_image=np.zeros((64, 64), dtype=np.uint8))
        SingularityDetector().process(ctx)
        assert ctx.roi_mask is not None
        assert ctx.roi_mask.shape == (64, 64)
        assert ctx.roi_mask.all()

    def test_roi_mask_independent_of_quality_mask(self) -> None:
        """When an upstream hook makes quality_mask all-False, roi_mask stays all-True."""
        ctx = PipelineContext(raw_image=np.zeros((64, 64), dtype=np.uint8))
        ctx.orientation_field = np.zeros((4, 4), dtype=np.float32)
        ctx.coherence_field = np.ones((4, 4), dtype=np.float32)
        ctx.quality_mask = np.zeros((64, 64), dtype=bool)  # all invalid
        SingularityDetector(roi_radius=10).process(ctx)
        # quality_mask stays all-False (intersection)
        assert not ctx.quality_mask.any()
        # roi_mask is the pure ROI disc — independent
        assert ctx.roi_mask.any()

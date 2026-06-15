"""
Tests for post-processing hooks using the v2 PipelineContext API.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.interfaces import IPipelineStep, PipelineContext
from src.core.types import AlgorithmOrigin, MinutiaCandidate, MinutiaType
from src.processing.post_hooks import (
    BorderMaskCleaner,
    BrokenRidgeHealer,
    EnsembleFusionFilter,
    LowConfidenceFilter,
    OrientationRefiner,
    QualityFilter,
    SpurRemover,
)


def _m(
    x: int,
    y: int,
    typ: MinutiaType = MinutiaType.TERMINATION,
    conf: float = 0.9,
) -> MinutiaCandidate:
    return MinutiaCandidate(
        x=x, y=y, angle=0.0, type=typ, confidence=conf, origin=AlgorithmOrigin.SKELETON,
    )


def _ctx_with(candidates, mask=None, enhanced=None, groups=None) -> PipelineContext:
    ctx = PipelineContext(raw_image=np.zeros((100, 100), dtype=np.uint8))
    ctx.candidate_groups = groups or []
    ctx.candidates = list(candidates)
    ctx.quality_mask = mask
    ctx.enhanced_image = enhanced
    return ctx


# ---------------------------------------------------------------------------
# QualityFilter
# ---------------------------------------------------------------------------


class TestQualityFilter:
    def test_implements_protocol(self) -> None:
        assert isinstance(QualityFilter(), IPipelineStep)

    def test_passes_all_when_no_mask(self) -> None:
        pts = [_m(10, 10), _m(20, 20)]
        ctx = _ctx_with(pts)
        QualityFilter().process(ctx)
        assert ctx.candidates == pts

    def test_drops_points_outside_mask(self) -> None:
        pts = [_m(5, 5), _m(50, 50)]
        mask = np.zeros((64, 64), dtype=bool)
        mask[50, 50] = True
        ctx = _ctx_with(pts, mask=mask)
        QualityFilter().process(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 50

    def test_empty_candidates(self) -> None:
        ctx = _ctx_with([], mask=np.ones((10, 10), dtype=bool))
        QualityFilter().process(ctx)
        assert ctx.candidates == []


# ---------------------------------------------------------------------------
# BorderMaskCleaner
# ---------------------------------------------------------------------------


class TestBorderMaskCleaner:
    def test_implements_protocol(self) -> None:
        assert isinstance(BorderMaskCleaner(), IPipelineStep)

    def test_drops_points_near_mask_border(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        pts = [_m(21, 21, typ=MinutiaType.BIFURCATION), _m(50, 50)]
        ctx = _ctx_with(pts, mask=mask)
        BorderMaskCleaner(border_px=15).process(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 50

    def test_keeps_inner_points(self) -> None:
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        pts = [_m(50, 50), _m(60, 60)]
        ctx = _ctx_with(pts, mask=mask)
        BorderMaskCleaner(border_px=10).process(ctx)
        assert len(ctx.candidates) == 2

    def test_no_mask_passes_all(self) -> None:
        pts = [_m(1, 1), _m(99, 99)]
        ctx = _ctx_with(pts)
        BorderMaskCleaner(border_px=15).process(ctx)
        assert len(ctx.candidates) == 2

    def test_invalid_roi_mode_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="roi_mode must be one of"):
            BorderMaskCleaner(roi_mode="bogus")  # type: ignore[arg-type]

    def test_invalid_border_px_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="border_px must be"):
            BorderMaskCleaner(border_px=-1)

    def test_roi_mode_none_passes_all(self) -> None:
        """In none mode, BorderMaskCleaner is a pass-through regardless of mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        pts = [_m(1, 1), _m(99, 99), _m(50, 50)]
        ctx = _ctx_with(pts, mask=mask)
        BorderMaskCleaner(roi_mode="none").process(ctx)
        assert len(ctx.candidates) == 3

    def test_roi_mode_core_uses_roi_mask(self) -> None:
        """In core mode, the roi_mask determines the valid region
        regardless of quality_mask."""
        qmask = np.ones((100, 100), dtype=bool)  # everything "valid"
        # ROI disc centred at (50, 50) with radius 20
        roi = np.zeros((100, 100), dtype=bool)
        yy, xx = np.mgrid[0:100, 0:100]
        roi = (yy - 50) ** 2 + (xx - 50) ** 2 <= 20 ** 2
        # Point at (10, 10) is inside qmask but outside ROI
        pts = [_m(10, 10), _m(50, 50)]
        ctx = _ctx_with(pts, mask=qmask)
        ctx.roi_mask = roi
        BorderMaskCleaner(border_px=2, roi_mode="core").process(ctx)
        # (10,10) outside ROI → dropped
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 50

    def test_roi_mode_core_falls_back_to_quality_mask(self) -> None:
        """If roi_mask is None, core mode falls back to quality_mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        pts = [_m(21, 21), _m(50, 50)]
        ctx = _ctx_with(pts, mask=mask)
        # roi_mask is None (default)
        assert ctx.roi_mask is None
        BorderMaskCleaner(border_px=15, roi_mode="core").process(ctx)
        # Falls back to quality_mask: (21,21) is at the border → dropped
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 50

    def test_roi_mode_bbox_legacy_behaviour(self) -> None:
        """bbox mode ignores roi_mask and uses quality_mask's bbox."""
        qmask = np.zeros((100, 100), dtype=bool)
        qmask[20:80, 20:80] = True
        roi = np.ones((100, 100), dtype=bool)  # ROI everywhere
        pts = [_m(21, 21), _m(50, 50)]
        ctx = _ctx_with(pts, mask=qmask)
        ctx.roi_mask = roi
        # In bbox mode, roi_mask is ignored — uses quality_mask bbox
        BorderMaskCleaner(border_px=15, roi_mode="bbox").process(ctx)
        # (21,21) is at the border of qmask bbox → dropped
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 50


# ---------------------------------------------------------------------------
# OrientationRefiner
# ---------------------------------------------------------------------------


class TestOrientationRefiner:
    def test_implements_protocol(self) -> None:
        assert isinstance(OrientationRefiner(), IPipelineStep)

    def test_no_image_passes_all(self) -> None:
        pts = [_m(50, 50)]
        ctx = _ctx_with(pts)
        OrientationRefiner().process(ctx)
        assert len(ctx.candidates) == 1

    def test_drops_low_coherence_points(self) -> None:
        image = np.zeros((100, 100), dtype=np.uint8)
        for x in range(30, 70, 4):
            image[:, x:x+2] = 200
        pts = [_m(50, 50), _m(2, 2)]
        ctx = _ctx_with(pts, enhanced=image)
        OrientationRefiner(window=16, coherence_threshold=0.35).process(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 50


# ---------------------------------------------------------------------------
# LowConfidenceFilter
# ---------------------------------------------------------------------------


class TestLowConfidenceFilter:
    def test_implements_protocol(self) -> None:
        assert isinstance(LowConfidenceFilter(), IPipelineStep)

    def test_drops_low_confidence(self) -> None:
        ctx = _ctx_with([_m(10, 10, conf=0.9), _m(20, 20, conf=0.3)])
        LowConfidenceFilter(threshold=0.6).process(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].x == 10

    def test_no_mask_required(self) -> None:
        ctx = _ctx_with([_m(5, 5)])
        LowConfidenceFilter().process(ctx)
        assert len(ctx.candidates) == 1


# ---------------------------------------------------------------------------
# SpurRemover
# ---------------------------------------------------------------------------


class TestSpurRemover:
    def test_implements_protocol(self) -> None:
        assert isinstance(SpurRemover(), IPipelineStep)

    def test_removes_close_bif_term_pair(self) -> None:
        bif = _m(10, 10, typ=MinutiaType.BIFURCATION)
        term = _m(13, 14, typ=MinutiaType.TERMINATION)
        ctx = _ctx_with([bif, term])
        SpurRemover(max_distance=8.0).process(ctx)
        assert ctx.candidates == []

    def test_keeps_far_bif_term_pair(self) -> None:
        bif = _m(10, 10, typ=MinutiaType.BIFURCATION)
        term = _m(100, 100, typ=MinutiaType.TERMINATION)
        ctx = _ctx_with([bif, term])
        SpurRemover(max_distance=8.0).process(ctx)
        assert len(ctx.candidates) == 2

    def test_empty_candidates(self) -> None:
        ctx = _ctx_with([])
        SpurRemover().process(ctx)
        assert ctx.candidates == []


# ---------------------------------------------------------------------------
# BrokenRidgeHealer
# ---------------------------------------------------------------------------


class TestBrokenRidgeHealer:
    def test_implements_protocol(self) -> None:
        assert isinstance(BrokenRidgeHealer(), IPipelineStep)

    def test_removes_close_term_pair(self) -> None:
        a = _m(10, 10, typ=MinutiaType.TERMINATION)
        b = _m(13, 14, typ=MinutiaType.TERMINATION)
        ctx = _ctx_with([a, b])
        BrokenRidgeHealer(max_distance=6.0).process(ctx)
        assert ctx.candidates == []

    def test_keeps_far_term_pair(self) -> None:
        a = _m(10, 10, typ=MinutiaType.TERMINATION)
        b = _m(100, 100, typ=MinutiaType.TERMINATION)
        ctx = _ctx_with([a, b])
        BrokenRidgeHealer(max_distance=6.0).process(ctx)
        assert len(ctx.candidates) == 2

    def test_ignores_bifurcations(self) -> None:
        bif = _m(10, 10, typ=MinutiaType.BIFURCATION)
        term = _m(12, 12, typ=MinutiaType.TERMINATION)
        ctx = _ctx_with([bif, term])
        BrokenRidgeHealer(max_distance=6.0).process(ctx)
        assert len(ctx.candidates) == 2


# ---------------------------------------------------------------------------
# EnsembleFusionFilter
# ---------------------------------------------------------------------------


class TestEnsembleFusionFilter:
    def test_implements_protocol(self) -> None:
        assert isinstance(EnsembleFusionFilter(), IPipelineStep)

    def test_keeps_agreed_points(self) -> None:
        g1 = [_m(10, 10)]
        g2 = [_m(10, 10)]
        ctx = _ctx_with([], groups=[g1, g2])
        EnsembleFusionFilter(radius=5.0, min_votes=2).process(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates == ctx.candidates

    def test_empty_groups(self) -> None:
        ctx = _ctx_with([], groups=[[], []])
        EnsembleFusionFilter().process(ctx)
        assert ctx.candidates == []
        assert ctx.candidates == []

    def test_no_agreement(self) -> None:
        g1 = [_m(10, 10)]
        g2 = [_m(99, 99)]
        ctx = _ctx_with([], groups=[g1, g2])
        EnsembleFusionFilter(radius=5.0, min_votes=2).process(ctx)
        assert ctx.candidates == []

    def test_three_groups_majority_vote(self) -> None:
        g1 = [_m(10, 10), _m(99, 99)]
        g2 = [_m(10, 10), _m(99, 99)]
        g3 = [_m(10, 10), _m(50, 50)]
        ctx = _ctx_with([], groups=[g1, g2, g3])
        EnsembleFusionFilter(radius=5.0, min_votes=2).process(ctx)
        assert len(ctx.candidates) == 2

    def test_no_groups(self) -> None:
        ctx = _ctx_with([])
        EnsembleFusionFilter().process(ctx)
        assert ctx.candidates == []

    def test_merge_averages_coordinates(self) -> None:
        g1 = [_m(10, 10)]
        g2 = [_m(14, 14)]
        ctx = _ctx_with([], groups=[g1, g2])
        EnsembleFusionFilter(radius=8.0, min_votes=2).process(ctx)
        assert len(ctx.candidates) == 1
        # Average of 10 and 14 = 12
        assert ctx.candidates[0].x == 12
        assert ctx.candidates[0].y == 12

    def test_merge_does_not_duplicate(self) -> None:
        g1 = [_m(10, 10), _m(99, 99)]
        g2 = [_m(12, 12), _m(99, 99)]
        ctx = _ctx_with([], groups=[g1, g2])
        EnsembleFusionFilter(radius=8.0, min_votes=2).process(ctx)
        assert len(ctx.candidates) == 2
        for p in ctx.candidates:
            assert isinstance(p.x, int)
            assert isinstance(p.y, int)

    def test_merge_majority_type(self) -> None:
        a = MinutiaCandidate(x=10, y=10, angle=0, type=MinutiaType.TERMINATION,
                             confidence=0.9, origin=AlgorithmOrigin.SKELETON)
        b = MinutiaCandidate(x=11, y=10, angle=0, type=MinutiaType.BIFURCATION,
                             confidence=0.7, origin=AlgorithmOrigin.GABOR)
        ctx = _ctx_with([], groups=[[a], [b]])
        EnsembleFusionFilter(radius=8.0, min_votes=2).process(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].type == MinutiaType.TERMINATION

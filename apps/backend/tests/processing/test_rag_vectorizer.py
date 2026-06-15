"""Tests for the RAG Triplet Vectorizer (Phase 10)."""
from __future__ import annotations

import math
import numpy as np
import pytest

from src.core.interfaces import PipelineContext
from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
)
from src.processing.vectorizer import RagTripletVectorizer


def _candidate(x: int, y: int, mtype: MinutiaType = MinutiaType.TERMINATION) -> MinutiaCandidate:
    return MinutiaCandidate(
        x=x, y=y, angle=0.0, type=mtype, confidence=1.0, origin=AlgorithmOrigin.SKELETON,
    )


def _make_context(candidates: list[MinutiaCandidate], core: tuple[int, int] | None = None) -> PipelineContext:
    return PipelineContext(raw_image=np.zeros((100, 100), dtype=np.uint8), candidates=candidates, core=core)


def test_vectorize_returns_empty_for_insufficient_minutiae() -> None:
    ctx = _make_context([_candidate(10, 10), _candidate(20, 20)])
    vectorizer = RagTripletVectorizer()
    assert vectorizer.vectorize(ctx) == []


def test_vectorize_returns_chunks_with_correct_feature_dim() -> None:
    candidates = [_candidate(x, y) for x, y in [(10, 10), (40, 20), (25, 50), (60, 60), (80, 30)]]
    ctx = _make_context(candidates, core=(50, 50))
    vectorizer = RagTripletVectorizer()
    chunks = vectorizer.vectorize(ctx)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.features) == RagTripletVectorizer.FEATURE_DIM
        assert isinstance(chunk.weight, float)


def test_weight_decay_triangles_closer_to_core_have_higher_weight() -> None:
    # 4 candidates: 2 close to core, 2 far from core
    candidates = [
        _candidate(50, 50),  # Core
        _candidate(60, 50),  # Close
        _candidate(55, 60),  # Close
        _candidate(250, 250),  # Far
        _candidate(260, 260),  # Far
    ]
    ctx = _make_context(candidates, core=(50, 50))
    vectorizer = RagTripletVectorizer(sigma=80.0)
    chunks = vectorizer.vectorize(ctx)

    # Multiple triangles produced. Verify weight spread.
    weights = [c.weight for c in chunks]
    assert len(weights) >= 2
    assert max(weights) > min(weights), f"Expected weight spread, got {weights}"
    # All weights should be in (0, 1]
    for w in weights:
        assert 0.0 < w <= 1.0
    # There should be at least one chunk with low weight (far triangles)
    assert min(weights) < 0.5, f"Expected some low-weight chunks, got {weights}"


def test_fallback_no_core_assigns_uniform_weight() -> None:
    candidates = [_candidate(x, y) for x, y in [(10, 10), (40, 20), (25, 50), (60, 60)]]
    ctx = _make_context(candidates, core=None)
    vectorizer = RagTripletVectorizer()
    chunks = vectorizer.vectorize(ctx)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.weight == 1.0


def test_triangle_invariants_rotation_invariance() -> None:
    """
    Triangles produce the same invariant features regardless of 
    rotation around the origin. (Sides and angles don't change).
    """
    pts = np.array([[0, 0], [10, 0], [5, 8]], dtype=np.float64)
    feats_original = RagTripletVectorizer._triangle_invariants(pts, np.array([1, 0, 0]))

    # Rotate 90 degrees
    theta = math.pi / 2
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts_rot = pts @ rot.T
    feats_rot = RagTripletVectorizer._triangle_invariants(pts_rot, np.array([1, 0, 0]))

    # First 6 features (sides + angles) should be equal
    np.testing.assert_allclose(feats_original[:6], feats_rot[:6], atol=1e-5)


def test_triangle_invariants_translation_invariance() -> None:
    """Sides are invariant to translation."""
    pts = np.array([[0, 0], [10, 0], [5, 8]], dtype=np.float64)
    feats = RagTripletVectorizer._triangle_invariants(pts, np.array([0, 0, 0]))

    # Translate
    pts_t = pts + np.array([100, 100])
    feats_t = RagTripletVectorizer._triangle_invariants(pts_t, np.array([0, 0, 0]))

    np.testing.assert_allclose(feats[:3], feats_t[:3], atol=1e-5)


def test_vectorizer_uses_provided_context_core() -> None:
    """Ensure the vectorizer reads core from context, not hardcoded."""
    # Triangle far from (10, 10) and close to (200, 200)
    candidates = [_candidate(200, 200), _candidate(210, 200), _candidate(205, 210)]
    ctx_far = _make_context(candidates, core=(10, 10))
    ctx_close = _make_context(candidates, core=(205, 203))

    vz = RagTripletVectorizer(sigma=50.0)
    weights_far = [c.weight for c in vz.vectorize(ctx_far)]
    weights_close = [c.weight for c in vz.vectorize(ctx_close)]

    # When core is close, weights should be near 1.0
    # When core is far, weights should be much lower
    assert all(w < 0.5 for w in weights_far)
    assert all(w > 0.5 for w in weights_close)

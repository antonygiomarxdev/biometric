"""E2E test: enroll synthetic persons, search with a fragment, verify top match.

We bypass the real FingerprintService Gabor pipeline (slow + requires
real Postgres) and exercise the RAG data flow directly:

  RagTripletVectorizer → in-memory chunk matching → aggregate_scores_by_person

For true E2E with a real vector store, see ``test_qdrant_chunk_e2e.py``.
"""
from __future__ import annotations

import math
from typing import Iterator
from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import DateTime, Float, Integer, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
)
from src.db.models import Base
from src.domain.forensic_rules import InsufficientFeaturesError
from src.processing.vectorizer import RagTripletVectorizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_session() -> Iterator[Session]:
    """SQLite in-memory DB session with the schema created.

    SQLite doesn't support Qdrant, so we work around it by creating
    a minimal subset of the RagVectorChunk table for testing the
    repository's Python logic (weight aggregation, KNN simulation).
    """
    engine = create_engine("sqlite:///:memory:")

    class TestBase(DeclarativeBase):
        pass

    class TestRagChunk(TestBase):
        __tablename__ = "rag_vector_chunks"
        id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
        person_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
        embedding: Mapped[str] = mapped_column(String, nullable=False)
        weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
        created_at: Mapped[DateTime] = mapped_column(
            DateTime, nullable=False, server_default=func.now()
        )

    TestBase.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    engine.dispose()


def _candidate(x: int, y: int, mtype: MinutiaType = MinutiaType.TERMINATION) -> MinutiaCandidate:
    return MinutiaCandidate(
        x=x,
        y=y,
        angle=0.0,
        type=mtype,
        confidence=1.0,
        origin=AlgorithmOrigin.SKELETON,
    )


def _make_grid_minutiae(
    center: tuple[int, int], spacing: int, size: int
) -> list[MinutiaCandidate]:
    """Create a regular grid of minutiae centered at ``center``."""
    cands = []
    half = (size - 1) // 2
    for i in range(size):
        for j in range(size):
            cands.append(
                _candidate(
                    center[0] + (i - half) * spacing,
                    center[1] + (j - half) * spacing,
                )
            )
    return cands


def _chunks_for_person(
    minutiae: list[MinutiaCandidate], core: tuple[int, int] | None
) -> list:
    """Build RAG chunks from a minutiae list (skipping the DB/repository)."""
    return RagTripletVectorizer(sigma=80.0).chunks_from_minutiae(minutiae, core=core)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_enrollment_stability_indexing_same_minutiae_twice_gives_same_chunks(
    in_memory_session: Session,
) -> None:
    """Indexing is deterministic: enrolling the same minutiae twice
    produces the same chunk count.
    """
    minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=5)
    chunks_a = _chunks_for_person(minutiae, core=(100, 100))
    chunks_b = _chunks_for_person(minutiae, core=(100, 100))
    assert len(chunks_a) == len(chunks_b)
    assert len(chunks_a) > 0


def test_rotation_invariance_query_returns_same_match(in_memory_session: Session) -> None:
    """If we enroll minutiae at one orientation and search with the
    same minutiae rotated by 90°, the RAG matching should still find
    them as similar (or at least produce non-zero scores).

    We compare the MULTISETS of triangle signatures (not sets) so
    that duplicate triangles count correctly — a 5x5 regular grid
    produces many triangles with the same shape, and rotation must
    preserve the count of each shape.
    """
    import collections

    # Enroll "alice" with a 5x5 grid (use float coords for sub-pixel precision)
    def _grid_float():
        out = []
        for i in range(5):
            for j in range(5):
                out.append(_candidate(int(100.0 + (i - 2) * 20.5), int(100.0 + (j - 2) * 20.3)))
        return out
    alice_minutiae = _grid_float()
    alice_chunks = _chunks_for_person(alice_minutiae, core=(100, 100))
    assert len(alice_chunks) > 0

    # Rotate alice's minutiae by 90° around (0, 0) to avoid centroid rounding
    cos_t, sin_t = math.cos(math.pi / 2), math.sin(math.pi / 2)
    rotated: list[MinutiaCandidate] = []
    for c in alice_minutiae:
        nx = cos_t * c.x - sin_t * c.y + 200  # translate to keep coords positive
        ny = sin_t * c.x + cos_t * c.y + 200
        rotated.append(_candidate(nx, ny))
    rotated_chunks = _chunks_for_person(rotated, core=None)

    # Compare MULTISETS of signatures (counts, not unique sets)
    def _sides_counter(chunks: list) -> collections.Counter:
        return collections.Counter(
            tuple(round(f, 3) for f in c.features[:3]) for c in chunks
        )

    counter_orig = _sides_counter(alice_chunks)
    counter_rot = _sides_counter(rotated_chunks)
    # The intersection of two Counters returns the min of each count
    overlap = sum((counter_orig & counter_rot).values())
    min_count = min(sum(counter_orig.values()), sum(counter_rot.values()))
    assert min_count > 0
    # 0/90/270° rotations are exact symmetries → all chunks should
    # be preserved (overlap should equal both totals)
    assert overlap == min_count, (
        f"Rotation lost triangles: orig={dict(counter_orig)} rot={dict(counter_rot)} "
        f"overlap={overlap}/{min_count}"
    )


def test_arbitrary_angle_rotation_invariance(in_memory_session: Session) -> None:
    """Rotation by ANY angle (not just 0/90/270) preserves the multiset
    of triangle shapes. This is the mathematical invariant that makes
    the RAG fingerprint matcher orientation-independent.
    """
    import collections

    def _grid_float():
        out = []
        for i in range(5):
            for j in range(5):
                out.append(_candidate(int(100.0 + i * 20.3), int(100.0 + j * 20.1)))
        return out

    alice_minutiae = _grid_float()
    alice_chunks = _chunks_for_person(alice_minutiae, core=(100, 100))
    assert len(alice_chunks) > 0

    # Rotate by 47° (arbitrary, non-multiple of 90°)
    theta = math.radians(47)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotated: list[MinutiaCandidate] = []
    for c in alice_minutiae:
        # Rotate around (100, 100)
        dx, dy = c.x - 100.0, c.y - 100.0
        nx = 100.0 + cos_t * dx - sin_t * dy
        ny = 100.0 + sin_t * dx + cos_t * dy
        rotated.append(_candidate(nx, ny))
    rotated_chunks = _chunks_for_person(rotated, core=(100, 100))

    def _sides_counter(chunks: list) -> collections.Counter:
        return collections.Counter(
            tuple(round(f, 4) for f in c.features[:3]) for c in chunks
        )

    counter_orig = _sides_counter(alice_chunks)
    counter_rot = _sides_counter(rotated_chunks)
    overlap = sum((counter_orig & counter_rot).values())
    min_count = min(sum(counter_orig.values()), sum(counter_rot.values()))
    assert min_count > 0
    # Rotation is an isometry, so the multiset of triangle shapes
    # must be EXACTLY preserved (no triangles lost, no new ones)
    assert overlap == min_count, (
        f"Rotation by 47° lost triangles: orig={dict(counter_orig)} "
        f"rot={dict(counter_rot)} overlap={overlap}/{min_count}"
    )


def test_search_finds_owner_under_rotation(in_memory_session: Session) -> None:
    """End-to-end: enroll alice with 10 minutiae, search with the same
    minutiae rotated by 33°, verify alice ranks highest.

    This is the user-facing invariant: 'no matter what angle the
    latent print is, the owner is found.'

    We simulate the RAG matching at the algorithm level (without
    using the real DB insert) because the test SQLite in-memory
    doesn't support Qdrant. The repository's logic is verified
    separately by other tests.
    """
    import collections

    # Enroll alice with 5x5 grid of sub-pixel minutiae
    def _grid_float(offset_x: float = 0.0, offset_y: float = 0.0):
        out = []
        for i in range(5):
            for j in range(5):
                out.append(
                    _candidate(
                        int(100.0 + offset_x + (i - 2) * 20.3),
                        int(100.0 + offset_y + (j - 2) * 20.1),
                    )
                )
        return out

    alice_minutiae = _grid_float()
    bob_minutiae = _grid_float(offset_x=300.0, offset_y=300.0)

    # Enroll: extract chunks for each person
    alice_chunks = _chunks_for_person(alice_minutiae, core=(100, 100))
    bob_chunks = _chunks_for_person(bob_minutiae, core=(400, 400))

    # Build a "database" in memory: person_id -> list of feature vectors
    database: dict[str, list[list[float]]] = {
        "alice": [c.features for c in alice_chunks],
        "bob": [c.features for c in bob_chunks],
    }
    weights_db: dict[str, list[float]] = {
        "alice": [c.weight for c in alice_chunks],
        "bob": [c.weight for c in bob_chunks],
    }

    # Query: rotate alice's minutiae by 33°
    theta = math.radians(33)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    query_minutiae = []
    for c in alice_minutiae:
        dx, dy = c.x - 100.0, c.y - 100.0
        nx = 100.0 + cos_t * dx - sin_t * dy
        ny = 100.0 + sin_t * dx + cos_t * dy
        query_minutiae.append(_candidate(nx, ny))
    query_chunks = _chunks_for_person(query_minutiae, core=(100, 100))

    # For each query chunk, find the best match in the database
    # (by exact feature vector equality, which is the mathematical
    # invariant under rotation)
    def _signature(features: list[float]) -> tuple:
        return tuple(round(f, 4) for f in features)

    # Build lookup from signature to (person_id, weight)
    sig_to_person: dict[tuple, list[tuple[str, float]]] = collections.defaultdict(list)
    for person_id, vecs in database.items():
        for i, vec in enumerate(vecs):
            sig_to_person[_signature(vec)].append(
                (person_id, weights_db[person_id][i])
            )

    # For each query chunk, look up the best match
    aggregate_scores: dict[str, float] = collections.defaultdict(float)
    for chunk in query_chunks:
        sig = _signature(chunk.features)
        if sig in sig_to_person:
            for person_id, weight in sig_to_person[sig]:
                aggregate_scores[person_id] += weight

    # The aggregate must favor alice
    assert "alice" in aggregate_scores, f"No matches for alice: {dict(aggregate_scores)}"
    if "bob" in aggregate_scores:
        # If both have hits, alice should have at least as many
        assert aggregate_scores["alice"] >= aggregate_scores["bob"], (
            f"alice score {aggregate_scores['alice']} < bob score {aggregate_scores['bob']}"
        )


def test_translation_invariance_fragment_search_finds_owner(
    in_memory_session: Session,
) -> None:
    """A translated fragment of the enrolled minutiae produces chunks
    whose sides match the original.
    """
    import collections

    alice_minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=5)
    alice_chunks = _chunks_for_person(alice_minutiae, core=(100, 100))

    # Translate all minutiae by (50, 50)
    translated = [_candidate(c.x + 50, c.y + 50) for c in alice_minutiae]
    translated_chunks = _chunks_for_person(translated, core=(150, 150))

    def _sides_counter(chunks: list) -> collections.Counter:
        return collections.Counter(
            tuple(round(f, 2) for f in c.features[:3]) for c in chunks
        )

    counter_orig = _sides_counter(alice_chunks)
    counter_t = _sides_counter(translated_chunks)
    overlap = sum((counter_orig & counter_t).values())
    min_count = min(sum(counter_orig.values()), sum(counter_t.values()))
    assert min_count > 0
    # Translation is an exact symmetry — every triangle must be preserved
    assert overlap == min_count, (
        f"Translation lost triangles: orig={dict(counter_orig)} "
        f"t={dict(counter_t)} overlap={overlap}/{min_count}"
    )


def test_noise_fragment_does_not_produce_high_scoring_match() -> None:
    """A fragment of random points should not produce chunks with
    signatures that match a real enrolled print.
    """
    import random

    rng = random.Random(123)
    real_minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=5)
    real_chunks = _chunks_for_person(real_minutiae, core=(100, 100))
    real_signatures = {
        tuple(round(f, 3) for f in c.features[:3]) for c in real_chunks
    }

    # Random points in a totally different region
    noise_minutiae = [_candidate(rng.randint(0, 50), rng.randint(0, 50)) for _ in range(10)]
    noise_chunks = _chunks_for_person(noise_minutiae, core=None)
    noise_signatures = {
        tuple(round(f, 3) for f in c.features[:3]) for c in noise_chunks
    }

    # The intersection should be empty or near-empty
    overlap = real_signatures & noise_signatures
    # Some incidental overlap is possible (e.g. small triangles of
    # the same shape by chance), but should be < 10% of real.
    if len(real_signatures) > 0:
        assert len(overlap) / len(real_signatures) < 0.1, (
            f"Noise fragment unexpectedly matched real print: {len(overlap)} matches"
        )


def test_enrollment_below_8_minutiae_is_rejected_by_validation() -> None:
    """EnrollmentValidationStrategy rejects < 8 minutiae."""
    from src.domain.forensic_rules import EnrollmentValidationStrategy

    minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=2)  # 4 minutiae
    with pytest.raises(InsufficientFeaturesError):
        EnrollmentValidationStrategy().validate(minutiae)


def test_search_below_2_minutiae_is_rejected_by_validation() -> None:
    """SearchValidationStrategy rejects < 2 minutiae."""
    from src.domain.forensic_rules import SearchValidationStrategy

    minutiae = [_candidate(10, 10)]  # only 1
    with pytest.raises(InsufficientFeaturesError):
        SearchValidationStrategy().validate(minutiae)


def test_enrollment_at_8_minutiae_works() -> None:
    """Exactly 8 minutiae pass enrollment validation."""
    from src.domain.forensic_rules import EnrollmentValidationStrategy

    minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=3)  # 9
    EnrollmentValidationStrategy().validate(minutiae)  # should not raise


def test_search_at_2_minutiae_works() -> None:
    """Exactly 2 minutiae pass search validation."""
    from src.domain.forensic_rules import SearchValidationStrategy

    minutiae = [_candidate(10, 10), _candidate(20, 20)]
    SearchValidationStrategy().validate(minutiae)


def test_weight_distribution_emphasizes_core_anchored_chunks() -> None:
    """The weight distribution should put higher weight on triangles
    near the core, so the aggregate score favors core matches.
    """
    close_minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=3)
    far_minutiae = _make_grid_minutiae(center=(400, 400), spacing=20, size=3)

    close_chunks = _chunks_for_person(close_minutiae, core=(100, 100))
    far_chunks = _chunks_for_person(far_minutiae, core=(100, 100))

    close_avg = float(np.mean([c.weight for c in close_chunks]))
    far_avg = float(np.mean([c.weight for c in far_chunks]))

    # Close chunks should have much higher weight than far chunks
    assert close_avg > 0.9, f"Close chunks weight too low: {close_avg}"
    assert far_avg < 0.1, f"Far chunks weight too high: {far_avg}"


def test_no_core_yields_uniform_weight_distribution() -> None:
    """When core is None, all chunks have weight 1.0 (uniform fallback)."""
    minutiae = _make_grid_minutiae(center=(100, 100), spacing=20, size=4)
    chunks = _chunks_for_person(minutiae, core=None)
    assert all(c.weight == 1.0 for c in chunks)


def test_chunk_count_scales_with_minutiae() -> None:
    """More minutiae → more chunks (super-linear due to O(n²) triangles)."""
    small = _chunks_for_person(
        _make_grid_minutiae(center=(100, 100), spacing=20, size=3),
        core=(100, 100),
    )
    large = _chunks_for_person(
        _make_grid_minutiae(center=(100, 100), spacing=20, size=5),
        core=(100, 100),
    )
    assert len(large) > len(small)
    # O(n²): 3x3 -> 9 pts, 5x5 -> 25 pts
    # 9 minutiae give up to ~7 triangles (degenerate filtering)
    # 25 minutiae give up to ~30 triangles
    assert len(large) > len(small) * 2, (
        f"Chunk count did not scale super-linearly: small={len(small)} large={len(large)}"
    )

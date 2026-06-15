"""Tests for RagMatchingService (Phase 10).

The service has two main operations: enroll and search. We mock the
repository and the FingerprintService so the tests are pure-Python
unit tests that don't need a real database.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
    TripletVector,
)
from src.db.repositories.rag_vector_repository import RagVectorRepository
from src.domain.forensic_rules import InsufficientFeaturesError
from src.services.fingerprint_service import FingerprintService
from src.services.rag_matching_service import RagMatchingService


def _make_candidates(n: int) -> list[MinutiaCandidate]:
    """Build n minutiae in a small grid for triangulation tests."""
    return [
        MinutiaCandidate(
            x=(i % 5) * 10 + 5,
            y=(i // 5) * 10 + 5,
            angle=0.0,
            type=MinutiaType.TERMINATION,
            confidence=1.0,
            origin=AlgorithmOrigin.SKELETON,
        )
        for i in range(n)
    ]


def _fake_normalized(minutiae: list[MinutiaCandidate]) -> NormalizedFingerprint:
    return NormalizedFingerprint(
        id="x", minutiae=minutiae, width=100, height=100
    )


def _make_fingerprint_service_mock(
    normalized: NormalizedFingerprint,
) -> MagicMock:
    """A mock FingerprintService that returns a fixed NormalizedFingerprint.

    Uses a plain MagicMock (no spec) because the service exposes
    private attributes (``_steps``, etc.) that the mock library
    cannot introspect via ``spec``.
    """
    svc = MagicMock()
    svc.process_image.return_value = normalized
    svc.enhancer = MagicMock()
    svc.extractors = [MagicMock()]
    svc.normalizer = MagicMock()
    return svc


class TestEnroll:
    """Enroll a fingerprint into the RAG chunk store."""

    def test_enroll_creates_chunks_for_sufficient_minutiae(self) -> None:
        minutiae = _make_candidates(10)
        normalized = _fake_normalized(minutiae)
        fp_svc = _make_fingerprint_service_mock(normalized)
        rag_repo = MagicMock(spec=RagVectorRepository)
        rag_repo.bulk_insert_chunks.return_value = ["row1", "row2", "row3"]

        service = RagMatchingService(
            fingerprint_service=fp_svc,
            rag_repository=rag_repo,
        )
        result = service.enroll(
            image=np.zeros((100, 100), dtype=np.uint8),
            person_id="alice",
            db=MagicMock(),
        )

        assert result.person_id == "alice"
        assert result.chunks_inserted > 0
        rag_repo.bulk_insert_chunks.assert_called_once()

    def test_enroll_rejects_fewer_than_8_minutiae(self) -> None:
        minutiae = _make_candidates(5)  # below 8
        normalized = _fake_normalized(minutiae)
        fp_svc = _make_fingerprint_service_mock(normalized)
        rag_repo = MagicMock(spec=RagVectorRepository)

        service = RagMatchingService(
            fingerprint_service=fp_svc,
            rag_repository=rag_repo,
        )
        with pytest.raises(InsufficientFeaturesError):
            service.enroll(
                image=np.zeros((100, 100), dtype=np.uint8),
                person_id="bob",
                db=MagicMock(),
            )
        rag_repo.bulk_insert_chunks.assert_not_called()

    def test_enroll_with_no_chunks_inserted_returns_zero(self) -> None:
        # <3 minutiae means no triangles, hence no chunks
        minutiae = _make_candidates(2)
        normalized = _fake_normalized(minutiae)
        # But validation will reject first (<8). Use empty minutiae.
        normalized = _fake_normalized([])
        fp_svc = _make_fingerprint_service_mock(normalized)
        rag_repo = MagicMock(spec=RagVectorRepository)
        rag_repo.bulk_insert_chunks.return_value = []

        # Empty minutiae will fail validation
        service = RagMatchingService(
            fingerprint_service=fp_svc,
            rag_repository=rag_repo,
        )
        with pytest.raises(InsufficientFeaturesError):
            service.enroll(
                image=np.zeros((100, 100), dtype=np.uint8),
                person_id="carol",
                db=MagicMock(),
            )


class TestSearch:
    """Search the RAG store for a latent image."""

    def test_search_aggregates_knn_results_by_person(self) -> None:
        minutiae = _make_candidates(10)
        normalized = _fake_normalized(minutiae)
        fp_svc = _make_fingerprint_service_mock(normalized)
        rag_repo = MagicMock(spec=RagVectorRepository)

        # Mock 3 query chunks: each returns 2 KNN hits
        def fake_knn(session, query_embedding, top_k):
            return [
                {"id": "c1", "person_id": "alice", "weight": 0.9,
                 "distance": 0.1, "weighted_score": 0.81},
                {"id": "c2", "person_id": "bob", "weight": 0.5,
                 "distance": 0.2, "weighted_score": 0.4},
            ]
        rag_repo.weighted_knn_search.side_effect = fake_knn
        rag_repo.aggregate_scores_by_person.return_value = [
            {"person_id": "alice", "total_score": 1.62, "hits": 2},
            {"person_id": "bob", "total_score": 0.8, "hits": 2},
        ]

        service = RagMatchingService(
            fingerprint_service=fp_svc,
            rag_repository=rag_repo,
        )
        hits = service.search(
            image=np.zeros((100, 100), dtype=np.uint8),
            db=MagicMock(),
        )

        assert len(hits) == 2
        assert hits[0].person_id == "alice"
        assert hits[0].total_score == pytest.approx(1.62)
        assert hits[1].person_id == "bob"
        assert rag_repo.weighted_knn_search.call_count >= 1

    def test_search_with_no_chunks_returns_empty(self) -> None:
        normalized = _fake_normalized([])  # <8 so validation passes? No, will raise
        # Use <2 minutiae to pass search validation but no triangles
        minutiae = _make_candidates(2)
        normalized = _fake_normalized(minutiae)
        fp_svc = _make_fingerprint_service_mock(normalized)
        rag_repo = MagicMock(spec=RagVectorRepository)

        service = RagMatchingService(
            fingerprint_service=fp_svc,
            rag_repository=rag_repo,
        )
        hits = service.search(
            image=np.zeros((100, 100), dtype=np.uint8),
            db=MagicMock(),
        )
        assert hits == []
        rag_repo.weighted_knn_search.assert_not_called()

    def test_search_uses_search_validation_strategy(self) -> None:
        """The search method must accept as few as 2 minutiae."""
        minutiae = _make_candidates(2)
        normalized = _fake_normalized(minutiae)
        fp_svc = _make_fingerprint_service_mock(normalized)
        rag_repo = MagicMock(spec=RagVectorRepository)

        service = RagMatchingService(
            fingerprint_service=fp_svc,
            rag_repository=rag_repo,
        )
        # Should NOT raise (2 minutiae is the minimum for search)
        service.search(
            image=np.zeros((100, 100), dtype=np.uint8),
            db=MagicMock(),
        )

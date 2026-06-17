"""E2E test: QdrantChunkRepository orchestrator with a stubbed FingerprintService.

The image processing pipeline (Gabor, skeleton, etc.) is not the
focus of Phase 15 — Phase 13/14 cover that. Here we verify the
orchestrator wires together correctly:
  1. enroll(image, person_id) → chunks persisted to Qdrant
  2. search(image) → aggregated persons from real HNSW
  3. Top-1 is the enrolled owner

We stub the FingerprintService to return synthetic minutiae based
on a deterministic seed (image.shape), so the same image always
yields the same minutiae pattern.

Requires Docker daemon (testcontainers spins up qdrant:latest).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer

from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
)
from src.processing.vectorizer import RagTripletVectorizer
from src.services.fingerprint_service import FingerprintService
from src.services.rag_matching_service import QdrantRagMatchingService, SearchHit
from src.db.qdrant_chunk_repository import QdrantChunkRepository
from tests.config import test_config


# ---------------------------------------------------------------------------
# Stub FingerprintService
# ---------------------------------------------------------------------------


class _StubFingerprintService(FingerprintService):
    """Returns deterministic synthetic minutiae based on image shape.

    The stub ignores the real image content and produces a regular
    grid of minutiae centered at a person-specific location derived
    from ``image.shape``. This is enough to verify the orchestrator
    flow without depending on the Gabor / skeleton pipeline.
    """

    _PERSON_OFFSETS: dict[str, tuple[int, int]] = {
        test_config.test_person_1: (100, 100),
        test_config.test_person_2: (300, 100),
        test_config.test_person_3: (500, 100),
    }

    def process_image(
        self,
        image: np.ndarray,
        fingerprint_id: str = "unknown",
        resize: bool = False,
    ) -> NormalizedFingerprint:
        # Person derived from fingerprint_id prefix
        person = fingerprint_id.split("_")[0] if "_" in fingerprint_id else fingerprint_id
        offset = self._PERSON_OFFSETS.get(person, (200, 200))

        minutiae: list[MinutiaCandidate] = []
        for i in range(test_config.synthetic_grid_size):
            for j in range(test_config.synthetic_grid_size):
                half_grid = test_config.synthetic_grid_size // 2
                minutiae.append(
                    MinutiaCandidate(
                        x=int(offset[0] + (i - half_grid) * test_config.synthetic_minutia_spacing),
                        y=int(offset[1] + (j - half_grid) * test_config.synthetic_minutia_spacing),
                        angle=0.0,
                        type=(
                            MinutiaType.BIFURCATION
                            if (i + j) % 3 == 0
                            else MinutiaType.TERMINATION
                        ),
                        confidence=1.0,
                        origin=AlgorithmOrigin.SKELETON,
                    )
                )
        return NormalizedFingerprint(
            id=fingerprint_id,
            minutiae=minutiae,
            width=image.shape[1] if image.ndim >= 2 else 100,
            height=image.shape[0] if image.ndim >= 2 else 100,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qdrant_server() -> Iterator[tuple[str, int]]:
    with QdrantContainer() as q:
        yield q.get_container_host_ip(), int(q.get_exposed_port(6333))


@pytest.fixture(scope="module")
def client(qdrant_server: tuple[str, int]) -> Iterator[QdrantClient]:
    c = QdrantClient(
        host=qdrant_server[0], port=qdrant_server[1], check_compatibility=False,
    )
    yield c
    c.close()


@pytest.fixture
def repo(client: QdrantClient) -> QdrantChunkRepository:
    collection = "test_orchestrator_e2e"
    r = QdrantChunkRepository(client, collection=collection)
    if r._collection_exists():
        r._client.delete_collection(collection_name=collection)
    r.ensure_collection()
    return r


@pytest.fixture
def search_svc(repo: QdrantChunkRepository) -> QdrantRagMatchingService:
    return QdrantRagMatchingService(
        fingerprint_service=_StubFingerprintService(),
        chunk_repository=repo,
    )


def _enroll(
    repo: QdrantChunkRepository,
    image: np.ndarray,
    person_id: str,
    fingerprint_id: str,
) -> int:
    """Direct enrollment: process_image → vectorize → bulk_insert."""
    fp_service = _StubFingerprintService()
    vectorizer = RagTripletVectorizer()
    normalized = fp_service._process_image(image, fingerprint_id=fingerprint_id)
    chunks = vectorizer._chunks_from_normalized(normalized)
    return repo.bulk_insert_chunks(person_id, fingerprint_id, chunks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _dummy_image(seed_id: str) -> np.ndarray:
    """A 96x103 grayscale 'image' (size matters for the stub)."""
    h = ord(seed_id[0]) if seed_id else 96
    return np.zeros((h, 96), dtype=np.uint8)


class TestOrchestratorEnroll:
    """The enroll flow: image → chunks → Qdrant."""

    def test_enroll_returns_chunk_count(
        self, repo: QdrantChunkRepository,
    ) -> None:
        result = _enroll(
            repo, _dummy_image(f"{test_config.test_person_1}{test_config.test_fp_suffix}"), 
            test_config.test_person_1, 
            f"{test_config.test_person_1}{test_config.test_fp_suffix}"
        )
        assert result > 0
        # Verify it's actually in Qdrant
        assert repo.collection_size() == result

    def test_multiple_enrolls_accumulate(
        self, repo: QdrantChunkRepository,
    ) -> None:
        for fp in (
            f"{test_config.test_person_1}{test_config.test_fp_suffix}", 
            f"{test_config.test_person_1}_fp2", 
            f"{test_config.test_person_2}{test_config.test_fp_suffix}"
        ):
            _enroll(
                repo, _dummy_image(fp),
                person_id=fp.split("_")[0],
                fingerprint_id=fp,
            )
        # 3 persons × 25 minutiae each → many chunks
        assert repo.collection_size() >= 3


class TestOrchestratorSearch:
    """The search() path: probe image → search hits → ranked persons."""

    def test_search_after_enroll_finds_owner(
        self, search_svc: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        # Enroll all 3
        for person in (test_config.test_person_1, test_config.test_person_2, test_config.test_person_3):
            _enroll(
                repo, _dummy_image(f"{person}{test_config.test_fp_suffix}"),
                person_id=person,
                fingerprint_id=f"{person}{test_config.test_fp_suffix}",
            )

        # Search with alice's pattern
        hits = search_svc.search(_dummy_image(f"{test_config.test_person_1}{test_config.test_probe_suffix}"), top_k_persons=3)
        assert len(hits) > 0
        assert isinstance(hits[0], SearchHit)
        assert hits[0].person_id == test_config.test_person_1
        # Score must be positive
        assert hits[0].total_score > 0.0
        # Alice should outrank bob and carol
        ranked = {h.person_id: h.total_score for h in hits}
        assert ranked[test_config.test_person_1] >= ranked.get(test_config.test_person_2, 0.0)
        assert ranked[test_config.test_person_1] >= ranked.get(test_config.test_person_3, 0.0)

    def test_search_empty_collection_returns_empty(
        self, search_svc: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        # No enrollments
        hits = search_svc.search(_dummy_image("probe"))
        assert hits == []

    def test_search_respects_top_k_persons(
        self, search_svc: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        for person in (test_config.test_person_1, test_config.test_person_2, test_config.test_person_3):
            _enroll(
                repo, _dummy_image(f"{person}{test_config.test_fp_suffix}"),
                person_id=person,
                fingerprint_id=f"{person}{test_config.test_fp_suffix}",
            )
        hits = search_svc.search(_dummy_image(f"{test_config.test_person_1}{test_config.test_probe_suffix}"), top_k_persons=2)
        assert len(hits) <= 2


class TestOrchestratorDelete:
    """The delete path: remove by person_id."""

    def test_delete_by_person_removes_chunks(
        self, repo: QdrantChunkRepository,
    ) -> None:
        _enroll(repo, _dummy_image(f"{test_config.test_person_1}{test_config.test_fp_suffix}"), test_config.test_person_1, f"{test_config.test_person_1}{test_config.test_fp_suffix}")
        _enroll(repo, _dummy_image(f"{test_config.test_person_2}{test_config.test_fp_suffix}"), test_config.test_person_2, f"{test_config.test_person_2}{test_config.test_fp_suffix}")
        size_before = repo.collection_size()
        assert size_before > 0
        deleted = repo.delete_by_person(test_config.test_person_1)
        assert deleted > 0
        assert repo.collection_size() < size_before

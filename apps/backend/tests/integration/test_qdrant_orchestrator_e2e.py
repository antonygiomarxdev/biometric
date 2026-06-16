"""E2E test: QdrantRagMatchingService orchestrator with a stubbed FingerprintService.

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
from src.services.rag_matching_service import (
    EnrollmentResult,
    QdrantRagMatchingService,
    SearchHit,
)
from src.db.qdrant_chunk_repository import QdrantChunkRepository


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
        "alice": (100, 100),
        "bob": (300, 100),
        "carol": (500, 100),
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
        for i in range(5):
            for j in range(5):
                minutiae.append(
                    MinutiaCandidate(
                        x=offset[0] + (i - 2) * 20.0,
                        y=offset[1] + (j - 2) * 20.0,
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
def service(repo: QdrantChunkRepository) -> QdrantRagMatchingService:
    return QdrantRagMatchingService(
        fingerprint_service=_StubFingerprintService(),
        chunk_repository=repo,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _dummy_image(seed_id: str) -> np.ndarray:
    """A 96x103 grayscale 'image' (size matters for the stub)."""
    h = ord(seed_id[0]) if seed_id else 96
    return np.zeros((h, 96), dtype=np.uint8)


class TestOrchestratorEnroll:
    """The enroll() path: image → chunks → Qdrant."""

    def test_enroll_returns_enrollment_result(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        result = service.enroll(
            _dummy_image("alice_fp1"), person_id="alice", fingerprint_id="alice_fp1",
        )
        assert isinstance(result, EnrollmentResult)
        assert result.person_id == "alice"
        assert result.chunks_inserted > 0
        assert result.total_weight > 0.0
        # Verify it's actually in Qdrant
        assert repo.collection_size() == result.chunks_inserted

    def test_enroll_default_fingerprint_id(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        result = service.enroll(_dummy_image("alice"), person_id="alice")
        assert result.person_id == "alice"
        assert result.chunks_inserted > 0

    def test_multiple_enrolls_accumulate(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        for fp in ("alice_fp1", "alice_fp2", "bob_fp1"):
            service.enroll(
                _dummy_image(fp),
                person_id=fp.split("_")[0],
                fingerprint_id=fp,
            )
        # 3 persons × 25 minutiae each → many chunks
        assert repo.collection_size() >= 3


class TestOrchestratorSearch:
    """The search() path: probe image → search hits → ranked persons."""

    def test_search_after_enroll_finds_owner(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        # Enroll all 3
        for person in ("alice", "bob", "carol"):
            service.enroll(
                _dummy_image(f"{person}_fp1"),
                person_id=person,
                fingerprint_id=f"{person}_fp1",
            )

        # Search with alice's pattern
        hits = service.search(_dummy_image("alice_probe"), top_k_persons=3)
        assert len(hits) > 0
        assert isinstance(hits[0], SearchHit)
        assert hits[0].person_id == "alice"
        # Score must be positive
        assert hits[0].total_score > 0.0
        # Alice should outrank bob and carol
        ranked = {h.person_id: h.total_score for h in hits}
        assert ranked["alice"] >= ranked.get("bob", 0.0)
        assert ranked["alice"] >= ranked.get("carol", 0.0)

    def test_search_empty_collection_returns_empty(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        # No enrollments
        hits = service.search(_dummy_image("probe"))
        assert hits == []

    def test_search_respects_top_k_persons(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        for person in ("alice", "bob", "carol"):
            service.enroll(
                _dummy_image(f"{person}_fp1"),
                person_id=person,
                fingerprint_id=f"{person}_fp1",
            )
        hits = service.search(_dummy_image("alice_probe"), top_k_persons=2)
        assert len(hits) <= 2


class TestOrchestratorDelete:
    """The delete path: remove by person_id."""

    def test_delete_by_person_removes_chunks(
        self, service: QdrantRagMatchingService, repo: QdrantChunkRepository,
    ) -> None:
        service.enroll(_dummy_image("alice_fp1"), "alice", "alice_fp1")
        service.enroll(_dummy_image("bob_fp1"), "bob", "bob_fp1")
        size_before = repo.collection_size()
        assert size_before > 0
        # QdrantRagMatchingService doesn't expose delete_by_person directly;
        # use the repo's method.
        deleted = repo.delete_by_person("alice")
        assert deleted > 0
        assert repo.collection_size() < size_before

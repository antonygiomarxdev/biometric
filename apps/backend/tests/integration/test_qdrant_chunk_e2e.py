"""E2E test: full chunk search round-trip against a real Qdrant server.

This test exercises the data flow that matters in production:
1. HNSW index creation
2. Bulk insert of multiple persons × multiple chunks
3. Weighted KNN search with cosine distance
4. Payload filtering (person_id, chunk_type)
5. Score aggregation per person
6. Top-1 retrieval

The image processing pipeline is bypassed (synthetic minutiae)
because Phase 13/14 work on tiny SOCOFing images is a separate
issue. The test focuses on the *chunk store* contract.

Requires Docker daemon (testcontainers spins up qdrant:latest).
"""
from __future__ import annotations

import collections
import time
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
    TripletVector,
)
from src.db.qdrant_chunk_repository import QdrantChunkRepository
from src.processing.vectorizer import RagTripletVectorizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qdrant_server() -> Iterator[tuple[str, int]]:
    """Spin up a real Qdrant container for the entire module."""
    with QdrantContainer() as q:
        host = q.get_container_host_ip()
        port = int(q.get_exposed_port(6333))
        yield host, port


@pytest.fixture(scope="module")
def client(qdrant_server: tuple[str, int]) -> Iterator[QdrantClient]:
    """A module-scoped client (avoids file-handle exhaustion)."""
    host, port = qdrant_server
    c = QdrantClient(host=host, port=port, check_compatibility=False)
    yield c
    c.close()


@pytest.fixture
def repo(client: QdrantClient) -> QdrantChunkRepository:
    """Module-scoped collection, cleared per test for isolation."""
    collection = "test_e2e_chunk_repo"
    r = QdrantChunkRepository(client, collection=collection)
    r.ensure_collection()
    # Clean slate for each test
    r._client.delete_collection(collection_name=collection)
    r.ensure_collection()
    return r


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _candidate(x: float, y: float, mtype: MinutiaType = MinutiaType.TERMINATION) -> MinutiaCandidate:
    return MinutiaCandidate(
        x=x,
        y=y,
        angle=0.0,
        type=mtype,
        confidence=1.0,
        origin=AlgorithmOrigin.SKELETON,
    )


def _grid_minutiae(
    center: tuple[float, float], spacing: float, size: int, offset_x: float = 0.0, offset_y: float = 0.0
) -> list[MinutiaCandidate]:
    """Create a regular grid of minutiae centered at ``center``."""
    out: list[MinutiaCandidate] = []
    half = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            out.append(
                _candidate(
                    center[0] + offset_x + (i - half) * spacing,
                    center[1] + offset_y + (j - half) * spacing,
                )
            )
    return out


def _chunks_for(minutiae: list[MinutiaCandidate], core: tuple[int, int] | None) -> list[TripletVector]:
    return RagTripletVectorizer(sigma=80.0).chunks_from_minutiae(minutiae, core=core)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQdrantChunkE2E:
    """E2E round-trip: enroll N persons, query with one, verify top-1."""

    def test_ensure_collection_creates_real_index(self, client: QdrantClient, repo: QdrantChunkRepository) -> None:
        """The collection exists in the server, not just in-memory."""
        info = client.get_collection(repo._collection)
        assert info is not None
        assert info.config.params.vectors.size == 9
        assert info.config.params.vectors.distance.name == "COSINE"

    def test_enroll_5_persons_top1_is_self(self, repo: QdrantChunkRepository) -> None:
        """Enroll 5 persons, query with one's minutiae, expect self at top-1."""
        # 5 persons at distinct positions, each with 5x5 grid of minutiae
        person_centers = {
            "alice": (100.0, 100.0),
            "bob": (300.0, 100.0),
            "carol": (500.0, 100.0),
            "dave": (100.0, 300.0),
            "eve": (300.0, 300.0),
        }
        for pid, (cx, cy) in person_centers.items():
            minutiae = _grid_minutiae((cx, cy), spacing=20.0, size=5)
            chunks = _chunks_for(minutiae, core=(int(cx), int(cy)))
            assert len(chunks) > 0
            inserted = repo.bulk_insert_chunks(pid, f"{pid}_fp1", chunks)
            assert inserted == len(chunks)

        # Real Qdrant, real count
        assert repo.collection_size() == sum(
            len(_chunks_for(_grid_minutiae(c, 20.0, 5), None)) for c in person_centers.values()
        )

        # Query with alice's minutiae (rotated 33° to test rotation invariance)
        theta = np.deg2rad(33)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cx, cy = person_centers["alice"]
        query_minutiae = []
        for c in _grid_minutiae((cx, cy), 20.0, 5):
            dx, dy = c.x - cx, c.y - cy
            nx = cx + cos_t * dx - sin_t * dy
            ny = cy + sin_t * dx + cos_t * dy
            query_minutiae.append(_candidate(nx, ny))
        query_chunks = _chunks_for(query_minutiae, core=(int(cx), int(cy)))
        assert len(query_chunks) > 0

        hits = repo.weighted_knn_search(query_chunks, top_k_per_chunk=5)
        persons = repo.aggregate_scores_by_person(hits)

        assert len(persons) >= 1
        assert persons[0].person_id == "alice", (
            f"Expected alice at top-1, got {persons[0].person_id} "
            f"with score {persons[0].total_score:.3f}"
        )

    def test_translation_invariance_via_real_qdrant(self, repo: QdrantChunkRepository) -> None:
        """Translated minutiae produce the same top-1 via real HNSW."""
        minutiae = _grid_minutiae((100.0, 100.0), 20.0, 5)
        chunks = _chunks_for(minutiae, core=(100, 100))
        repo.bulk_insert_chunks("alice", "alice_fp1", chunks)

        # Translate by (50, 50)
        translated = [_candidate(c.x + 50, c.y + 50) for c in minutiae]
        translated_chunks = _chunks_for(translated, core=(150, 150))

        hits = repo.weighted_knn_search(translated_chunks, top_k_per_chunk=5)
        persons = repo.aggregate_scores_by_person(hits)
        assert persons[0].person_id == "alice"

    def test_chunk_type_payload_filter(self, repo: QdrantChunkRepository) -> None:
        """Payload filter on chunk_type works server-side."""
        delaunay = [TripletVector(features=[0.1] * 9, weight=1.0)]
        mcc = [TripletVector(features=[0.1] * 9, weight=1.0)]
        repo.bulk_insert_chunks("p1", "fp1", delaunay, chunk_type="delaunay")
        repo.bulk_insert_chunks("p1", "fp1", mcc, chunk_type="mcc")

        delaunay_hits = repo.weighted_knn_search(
            [TripletVector(features=[0.1] * 9, weight=1.0)],
            top_k_per_chunk=5,
            chunk_type="delaunay",
        )
        mcc_hits = repo.weighted_knn_search(
            [TripletVector(features=[0.1] * 9, weight=1.0)],
            top_k_per_chunk=5,
            chunk_type="mcc",
        )
        assert all(h.chunk_type == "delaunay" for h in delaunay_hits)
        assert all(h.chunk_type == "mcc" for h in mcc_hits)

    def test_pagination_scroll_returns_all(self, repo: QdrantChunkRepository) -> None:
        """scroll_all() with limit respects pagination server-side."""
        # Insert 25 chunks (more than a single page)
        chunks = [TripletVector(features=[i / 25.0] * 9, weight=1.0) for i in range(25)]
        repo.bulk_insert_chunks("p1", "fp1", chunks)
        assert repo.collection_size() == 25
        all_chunks = repo.scroll_all(limit=10)
        assert len(all_chunks) == 25

    def test_delete_by_person_real_index(self, repo: QdrantChunkRepository) -> None:
        """Server-side delete via payload filter, count matches."""
        repo.bulk_insert_chunks("alice", "fp1", _chunks_for(
            _grid_minutiae((100, 100), 20.0, 4), core=(100, 100),
        ))
        repo.bulk_insert_chunks("bob", "fp1", _chunks_for(
            _grid_minutiae((300, 300), 20.0, 4), core=(300, 300),
        ))
        assert repo.collection_size() == sum(
            len(_chunks_for(_grid_minutiae(c, 20.0, 4), core=c))
            for c in [(100, 100), (300, 300)]
        )
        deleted = repo.delete_by_person("alice")
        assert deleted > 0
        assert repo.collection_size() > 0  # bob's chunks remain

    def test_top_k_limit_respected(self, repo: QdrantChunkRepository) -> None:
        """top_k_per_chunk=2 returns at most 2 per probe chunk."""
        chunks = _chunks_for(_grid_minutiae((100, 100), 20.0, 4), core=(100, 100))
        repo.bulk_insert_chunks("p1", "fp1", chunks)

        hits = repo.weighted_knn_search(
            [TripletVector(features=[0.5] * 9, weight=1.0)],
            top_k_per_chunk=2,
        )
        assert len(hits) <= 2

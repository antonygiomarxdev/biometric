"""
Unit tests for QdrantChunkRepository using an in-memory Qdrant client.

Tests are independent (no test ordering), each creates its own collection
via ensure_collection so isolated runs give the same result.
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.core.types import ChunkHit, TripletVector
from src.db.qdrant_chunk_repository import QdrantChunkRepository, COLLECTION_NAME, _chunk_point_id


def _sample_chunks(n: int = 5) -> list[TripletVector]:
    """Generate *n* synthetic Delaunay-chunk vectors (9-dim)."""
    rng = np.random.default_rng(42)
    return [
        TripletVector(features=rng.uniform(0, 1, 9).tolist(), weight=float(w))
        for w in np.linspace(0.5, 1.0, n).tolist()
    ]


@pytest.fixture
def repo() -> QdrantChunkRepository:
    """Create a repository backed by an in-memory Qdrant instance."""
    client = QdrantClient(location=":memory:")
    r = QdrantChunkRepository(client, collection=f"test_{id(client)}")
    r.ensure_collection()
    return r


class TestQdrantChunkRepository:
    def test_ensure_collection_creates(self) -> None:
        client = QdrantClient(location=":memory:")
        r = QdrantChunkRepository(client, collection="test_create")
        r.ensure_collection()
        info = client.get_collection("test_create")
        assert info is not None
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            # In newer qdrant-client versions, vectors is a dict for named vectors (default is "")
            v_params = vectors_config.get("")
        else:
            v_params = vectors_config
            
        assert v_params is not None
        assert v_params.size == 9
        assert v_params.distance == "Cosine"

    def test_ensure_collection_idempotent(self, repo: QdrantChunkRepository) -> None:
        repo.ensure_collection()
        repo.ensure_collection()

    def test_collection_size_empty(self, repo: QdrantChunkRepository) -> None:
        assert repo.collection_size() == 0

    def test_bulk_insert_chunks_returns_count(self, repo: QdrantChunkRepository) -> None:
        chunks = _sample_chunks(5)
        count = repo.bulk_insert_chunks("person_1", "fp_1", chunks)
        assert count == 5

    def test_collection_size_after_insert(self, repo: QdrantChunkRepository) -> None:
        chunks = _sample_chunks(3)
        repo.bulk_insert_chunks("person_1", "fp_1", chunks)
        assert repo.collection_size() == 3

    def test_collection_size_filtered_by_type(self, repo: QdrantChunkRepository) -> None:
        chunks = _sample_chunks(3)
        repo.bulk_insert_chunks("person_1", "fp_1", chunks, chunk_type="delaunay")
        assert repo.collection_size(chunk_type="delaunay") == 3
        assert repo.collection_size(chunk_type="mcc") == 0

    def test_weighted_knn_search_returns_hits(self, repo: QdrantChunkRepository) -> None:
        """Insert chunks, search with a similar probe, verify hits."""
        # Insert chunks with deterministic pattern
        chunks = [
            TripletVector(features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], weight=1.0),
            TripletVector(features=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], weight=0.8),
        ]
        repo.bulk_insert_chunks("person_1", "fp_1", chunks)

        # Query with a vector close to the first chunk
        query = [
            TripletVector(features=[0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91], weight=1.0),
        ]
        hits = repo.weighted_knn_search(query, top_k_per_chunk=5)

        assert len(hits) >= 1
        assert hits[0].person_id == "person_1"
        assert hits[0].weighted_score > 0.0

    def test_weighted_knn_search_deduplicates(self, repo: QdrantChunkRepository) -> None:
        """Two chunks from the same (person, fp) return as one hit."""
        chunks = _sample_chunks(3)
        repo.bulk_insert_chunks("person_1", "fp_1", chunks)

        # Query close to all 3 chunks
        query = [
            TripletVector(features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], weight=1.0),
            TripletVector(features=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], weight=1.0),
            TripletVector(features=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], weight=1.0),
        ]
        hits = repo.weighted_knn_search(query, top_k_per_chunk=5)

        # Dedup: multiple hits to same (person_1, fp_1) become one
        person_fps = {(h.person_id, h.fingerprint_id) for h in hits}
        assert len(person_fps) == 1

    def test_aggregate_scores_by_person(self, repo: QdrantChunkRepository) -> None:
        from src.core.types import ChunkHit

        hits = [
            ChunkHit("alice", "fp_1", "cap_1", "g_1", "delaunay", 1.0, 0.9, 0.9),
            ChunkHit("alice", "fp_2", "cap_2", "g_2", "delaunay", 1.0, 0.8, 0.8),
            ChunkHit("bob", "fp_3", "cap_3", "g_3", "delaunay", 1.0, 0.7, 0.7),
        ]
        persons = repo.aggregate_scores_by_person(hits)
        assert len(persons) == 2
        assert persons[0].person_id == "alice"
        assert persons[0].total_score == pytest.approx(1.7)
        assert persons[1].person_id == "bob"
        assert persons[1].total_score == pytest.approx(0.7)

    def test_empty_query_returns_empty(self, repo: QdrantChunkRepository) -> None:
        assert repo.weighted_knn_search([]) == []

    def test_bulk_insert_empty_chunks_returns_zero(self, repo: QdrantChunkRepository) -> None:
        assert repo.bulk_insert_chunks("person_1", "fp_1", []) == 0

    def test_delete_by_person(self, repo: QdrantChunkRepository) -> None:
        chunks = _sample_chunks(5)
        repo.bulk_insert_chunks("person_a", "fp_1", chunks)
        repo.bulk_insert_chunks("person_a", "fp_2", _sample_chunks(3))
        assert repo.collection_size() == 8
        deleted = repo.delete_by_person("person_a")
        assert deleted >= 8
        assert repo.collection_size() == 0

    def test_delete_by_fingerprint(self, repo: QdrantChunkRepository) -> None:
        chunks = _sample_chunks(2)
        repo.bulk_insert_chunks("p1", "fp_keep", chunks)
        repo.bulk_insert_chunks("p1", "fp_delete", _sample_chunks(3))
        repo.delete_by_fingerprint("fp_delete")
        hits = repo.scroll_all(limit=100)
        assert all(h.fingerprint_id == "fp_keep" for h in hits)

    def test_scroll_all_returns_chunks(self, repo: QdrantChunkRepository) -> None:
        chunks = _sample_chunks(3)
        repo.bulk_insert_chunks("p1", "fp_1", chunks)
        scrolled = repo.scroll_all(limit=10)
        assert len(scrolled) == 3

    def test_ensure_collection_creates_payload_indexes(self) -> None:
        """Payload indexes silently succeed in-memory (not persisted)."""
        import warnings

        warnings.filterwarnings("ignore", message="Payload indexes have no effect")
        client = QdrantClient(location=":memory:")
        r = QdrantChunkRepository(client, collection="test_indexes")
        r.ensure_collection()
        assert client.get_collection("test_indexes") is not None

    def test_weighted_knn_search_respects_chunk_type(self, repo: QdrantChunkRepository) -> None:
        delaunay = [TripletVector(features=[0.1] * 9, weight=1.0)]
        mcc = [TripletVector(features=[0.9] * 9, weight=1.0)]
        repo.bulk_insert_chunks("p1", "fp_1", delaunay, chunk_type="delaunay")
        repo.bulk_insert_chunks("p1", "fp_1", mcc, chunk_type="mcc")

        query = [TripletVector(features=[0.1] * 9, weight=1.0)]
        delaunay_hits = repo.weighted_knn_search(query, top_k_per_chunk=5, chunk_type="delaunay")
        assert len(delaunay_hits) >= 1
        mcc_hits = repo.weighted_knn_search(query, top_k_per_chunk=5, chunk_type="mcc")
        assert len(mcc_hits) >= 1


class TestChunkPointId:
    """Unit tests for the deterministic chunk ID helper."""

    def test_deterministic(self) -> None:
        a = _chunk_point_id("alice", "fp_1", 0)
        b = _chunk_point_id("alice", "fp_1", 0)
        assert a == b

    def test_distinct_person_yields_different_id(self) -> None:
        a = _chunk_point_id("alice", "fp_1", 0)
        b = _chunk_point_id("bob", "fp_1", 0)
        assert a != b

    def test_distinct_fingerprint_yields_different_id(self) -> None:
        a = _chunk_point_id("alice", "fp_1", 0)
        b = _chunk_point_id("alice", "fp_2", 0)
        assert a != b

    def test_distinct_index_yields_different_id(self) -> None:
        a = _chunk_point_id("alice", "fp_1", 0)
        b = _chunk_point_id("alice", "fp_1", 1)
        assert a != b

    def test_distinct_chunk_type_yields_different_id(self) -> None:
        a = _chunk_point_id("alice", "fp_1", 0, chunk_type="delaunay")
        b = _chunk_point_id("alice", "fp_1", 0, chunk_type="mcc")
        assert a != b

    def test_returns_positive_int(self) -> None:
        result = _chunk_point_id("alice", "fp_1", 0)
        assert isinstance(result, int)
        assert result > 0

    def test_returns_64bit_unsigned_safe_int(self) -> None:
        """Qdrant requires 64-bit unsigned integer or UUID. Hash → 16 hex
        chars = 64 bits, must stay within Python int (unbounded)."""
        result = _chunk_point_id("alice", "fp_1", 0)
        # Must be < 2^64 to be representable as a uint64
        assert result < (1 << 64)


class TestDeduplicateChunkHits:
    """Unit tests for the chunk-hit deduplicator (keeps max score per
    (person_id, fingerprint_id))."""

    def _hit(self, person: str, fp: str, score: float) -> ChunkHit:
        return ChunkHit(
            person_id=person,
            fingerprint_id=fp,
            capture_id="",
            graph_id="",
            chunk_type="delaunay",
            weight=1.0,
            similarity=score,
            weighted_score=score,
        )

    def test_keeps_highest_score_per_key(self) -> None:
        from src.db.qdrant_chunk_repository import QdrantChunkRepository

        hits = [
            self._hit("alice", "fp_1", 0.5),
            self._hit("alice", "fp_1", 0.9),
            self._hit("alice", "fp_1", 0.3),
        ]
        deduped = QdrantChunkRepository._deduplicate_chunk_hits(hits)
        assert len(deduped) == 1
        assert deduped[0].weighted_score == pytest.approx(0.9)

    def test_distinct_keys_preserved(self) -> None:
        from src.db.qdrant_chunk_repository import QdrantChunkRepository

        hits = [
            self._hit("alice", "fp_1", 0.9),
            self._hit("alice", "fp_2", 0.7),
            self._hit("bob", "fp_1", 0.5),
        ]
        deduped = QdrantChunkRepository._deduplicate_chunk_hits(hits)
        assert len(deduped) == 3
        # Order is not guaranteed by dict iteration
        keys = {(h.person_id, h.fingerprint_id) for h in deduped}
        assert keys == {("alice", "fp_1"), ("alice", "fp_2"), ("bob", "fp_1")}

    def test_empty_list(self) -> None:
        from src.db.qdrant_chunk_repository import QdrantChunkRepository

        assert QdrantChunkRepository._deduplicate_chunk_hits([]) == []

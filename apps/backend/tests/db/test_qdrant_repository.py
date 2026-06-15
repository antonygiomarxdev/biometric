"""Tests for QdrantRepository using Qdrant in-memory mode.

These tests use ``QdrantClient(location=":memory:")`` so they require
NO Docker or running infrastructure — they are pure unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.db.qdrant_repository import COLLECTION_NAME, EMBEDDING_DIM, QdrantRepository


@pytest.fixture
def repo() -> QdrantRepository:
    """Create a QdrantRepository backed by an in-memory Qdrant instance."""
    r = QdrantRepository.__new__(QdrantRepository)
    r._client = QdrantClient(location=":memory:")
    return r


class TestQdrantRepository:
    def test_ensure_collection_creates_collection(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        info = repo._client.get_collection(COLLECTION_NAME)
        assert info.status == "green"
        assert info.config.params.vectors.size == EMBEDDING_DIM

    def test_ensure_collection_idempotent(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        repo.ensure_collection()  # should not raise
        assert repo.collection_size() == 0

    def test_insert_and_search_roundtrip(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        vec_a = [1.0] * EMBEDDING_DIM
        vec_b = [-1.0] * EMBEDDING_DIM

        repo.insert(fingerprint_id="print_a", vector=vec_a, payload={"person": "Alice"})
        repo.insert(fingerprint_id="print_b", vector=vec_b, payload={"person": "Bob"})

        assert repo.collection_size() == 2

        # Search with vec_a — should return print_a first
        results = repo.search(vector=vec_a, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "print_a"
        assert results[0][1] > 0.99

    def test_search_returns_top_k(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        for i in range(10):
            vec = [float(i)] * EMBEDDING_DIM
            repo.insert(fingerprint_id=f"p_{i}", vector=vec)

        results = repo.search(vector=[5.0] * EMBEDDING_DIM, top_k=3)
        assert len(results) == 3

    def test_search_with_unknown_vector(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        repo.insert(
            fingerprint_id="only_one",
            vector=[0.5] * EMBEDDING_DIM,
            payload={"name": "sole"},
        )
        query = [1.0] * EMBEDDING_DIM  # different from 0.5
        results = repo.search(vector=query, top_k=5)
        assert len(results) == 1

    def test_insert_with_payload(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        payload = {"person_id": "P001", "image_path": "/data/print.bmp"}
        repo.insert(fingerprint_id="has_payload", vector=[0.1] * EMBEDDING_DIM, payload=payload)

        # Search for the same vector — verify payload comes back
        results = repo.search(vector=[0.1] * EMBEDDING_DIM, top_k=1)
        assert len(results) == 1
        assert results[0][0] == "has_payload"

    def test_insert_overwrite_same_id(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        repo.insert(fingerprint_id="same_id", vector=[0.1] * EMBEDDING_DIM, payload={"v": 1})
        repo.insert(fingerprint_id="same_id", vector=[0.9] * EMBEDDING_DIM, payload={"v": 2})

        assert repo.collection_size() == 1
        results = repo.search(vector=[0.9] * EMBEDDING_DIM, top_k=1)
        assert results[0][0] == "same_id"

    def test_delete_removes_point(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        repo.insert(fingerprint_id="delete_me", vector=[0.5] * EMBEDDING_DIM)
        repo.insert(fingerprint_id="keep_me", vector=[0.5] * EMBEDDING_DIM)
        assert repo.collection_size() == 2

        repo.delete("delete_me")
        assert repo.collection_size() == 1

        results = repo.search(vector=[0.5] * EMBEDDING_DIM, top_k=5)
        ids = [r[0] for r in results]
        assert "delete_me" not in ids
        assert "keep_me" in ids

    def test_collection_size_empty(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        assert repo.collection_size() == 0

    def test_embedding_dim_mismatch_raises(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        wrong_vec = [1.0, 2.0, 3.0]  # only 3 dims instead of EMBEDDING_DIM
        with pytest.raises(Exception):
            repo.insert(fingerprint_id="bad", vector=wrong_vec)


class TestQdrantRepositoryRealData:
    """Integration-style tests using real RidgeGraph embeddings from SOCOFing."""

    @pytest.fixture
    def repo(self) -> QdrantRepository:
        r = QdrantRepository.__new__(QdrantRepository)
        r._client = QdrantClient(location=":memory:")
        r.ensure_collection()
        return r

    @pytest.fixture
    def sample_embeddings(self) -> dict[str, list[float]]:
        from pathlib import Path

        from src.processing.graph_embedder import embed_graph
        from tests.processing.test_graph_extractor import _load_graph

        fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "socofing_real"
        if not fixtures.exists():
            pytest.skip("SOCOFing fixtures not found")
        paths = sorted(fixtures.glob("*.BMP"))[:5]
        result: dict[str, list[float]] = {}
        for p in paths:
            graph = _load_graph(p)
            emb = embed_graph(graph).tolist()
            result[p.stem] = emb
        return result

    def test_index_real_embeddings(self, repo: QdrantRepository, sample_embeddings: dict) -> None:
        for fid, emb in sample_embeddings.items():
            repo.insert(fingerprint_id=fid, vector=emb, payload={"fingerprint_id": fid})

        assert repo.collection_size() == len(sample_embeddings)

    def test_search_returns_self_as_top_match(
        self, repo: QdrantRepository, sample_embeddings: dict
    ) -> None:
        for fid, emb in sample_embeddings.items():
            repo.insert(fingerprint_id=fid, vector=emb, payload={"fingerprint_id": fid})

        for fid, emb in sample_embeddings.items():
            results = repo.search(vector=emb, top_k=3)
            assert results[0][0] == fid, f"{fid} should match itself first"
            assert results[0][1] > 0.99, f"Self-similarity for {fid} too low: {results[0][1]}"

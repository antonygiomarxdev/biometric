"""Tests for QdrantRepository — ICoarseMatcher adapter.

Uses ``QdrantClient(location=":memory:")`` so no Docker or running
infrastructure is required.  Tests target the public interface only;
the Qdrant SDK is implementation detail.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.core.types import CoarseMatch, GraphEmbedding
from src.db.qdrant_repository import QdrantRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _empty_embedding() -> GraphEmbedding:
    """Return an all-zeros GraphEmbedding (used as a stand-in vector)."""
    return GraphEmbedding(
        degree_0_ratio=0.0, degree_1_ratio=0.0, degree_2_ratio=0.0,
        degree_3_ratio=0.0, degree_4plus_ratio=0.0,
        edge_len_p10=0.0, edge_len_p25=0.0, edge_len_p50=0.0,
        edge_len_p75=0.0, edge_len_p90=0.0,
        edge_len_mean=0.0, edge_len_std=0.0,
        weight_p10=0.0, weight_p50=0.0, weight_p90=0.0,
        weight_mean=0.0, weight_std=0.0,
        log_num_nodes=0.0, log_num_edges=0.0,
        cutoff_ratio=0.0, avg_degree=0.0, density=0.0,
    )


@pytest.fixture
def repo() -> QdrantRepository:
    """A QdrantRepository backed by an in-memory Qdrant instance."""
    return QdrantRepository(QdrantClient(location=":memory:"))


def _load_socofing_embeddings() -> dict[str, GraphEmbedding]:
    """Build real GraphEmbeddings from the SOCOFing fixtures."""
    from src.processing.graph_embedder import embed_graph
    from tests.processing.test_graph_extractor import _load_graph

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "socofing_real"
    if not fixtures.exists():
        pytest.skip("SOCOFing fixtures not found")
    paths = sorted(fixtures.glob("*.BMP"))[:5]
    if not paths:
        pytest.skip("No SOCOFing .BMP fixtures")
    return {p.stem: embed_graph(_load_graph(p)) for p in paths}


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    def test_ensure_collection_creates_collection(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        # The collection is queryable through the public adapter.
        assert repo.collection_size() == 0

    def test_ensure_collection_is_idempotent(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        repo.ensure_collection()  # must not raise
        assert repo.collection_size() == 0

    def test_collection_size_starts_at_zero(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        assert repo.collection_size() == 0


# ---------------------------------------------------------------------------
# Write + search
# ---------------------------------------------------------------------------


class TestUpsertAndSearch:
    def test_roundtrip_with_graph_embedding(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()

        emb_a = GraphEmbedding(
            degree_0_ratio=0.0, degree_1_ratio=1.0, degree_2_ratio=0.0,
            degree_3_ratio=0.0, degree_4plus_ratio=0.0,
            edge_len_p10=10.0, edge_len_p25=10.0, edge_len_p50=10.0,
            edge_len_p75=10.0, edge_len_p90=10.0,
            edge_len_mean=10.0, edge_len_std=0.0,
            weight_p10=1.0, weight_p50=1.0, weight_p90=1.0,
            weight_mean=1.0, weight_std=0.0,
            log_num_nodes=float(np.log1p(2)),
            log_num_edges=float(np.log1p(1)),
            cutoff_ratio=0.0, avg_degree=1.0, density=1.0,
        )
        emb_b = GraphEmbedding(
            degree_0_ratio=0.0, degree_1_ratio=1.0, degree_2_ratio=0.0,
            degree_3_ratio=0.0, degree_4plus_ratio=0.0,
            edge_len_p10=-10.0, edge_len_p25=-10.0, edge_len_p50=-10.0,
            edge_len_p75=-10.0, edge_len_p90=-10.0,
            edge_len_mean=-10.0, edge_len_std=0.0,
            weight_p10=0.0, weight_p50=0.0, weight_p90=0.0,
            weight_mean=0.0, weight_std=0.0,
            log_num_nodes=float(np.log1p(2)),
            log_num_edges=float(np.log1p(1)),
            cutoff_ratio=0.0, avg_degree=1.0, density=1.0,
        )

        repo.upsert("print_a", emb_a, metadata={"person": "Alice"})
        repo.upsert("print_b", emb_b, metadata={"person": "Bob"})

        assert repo.collection_size() == 2
        results = repo.search(emb_a, top_k=2)
        assert [r.fingerprint_id for r in results] == ["print_a", "print_b"]
        assert results[0].score > 0.99
        assert results[0].metadata == {"person": "Alice"}

    def test_roundtrip_with_raw_ndarray(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        vec = np.zeros(GraphEmbedding.EMBEDDING_DIM, dtype=np.float32)
        vec[0] = 1.0  # degree_0_ratio = 1.0
        repo.upsert("only_one", vec)
        results = repo.search(vec, top_k=1)
        assert results[0].fingerprint_id == "only_one"

    def test_upsert_overwrites_same_id(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        emb1 = replace(_empty_embedding(), degree_1_ratio=0.1)
        emb2 = replace(_empty_embedding(), degree_1_ratio=0.9)

        repo.upsert("same_id", emb1, metadata={"v": 1})
        repo.upsert("same_id", emb2, metadata={"v": 2})
        assert repo.collection_size() == 1
        results = repo.search(emb2, top_k=1)
        assert results[0].fingerprint_id == "same_id"
        assert results[0].metadata == {"v": 2}

    def test_search_with_only_one_match(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        emb = _empty_embedding()
        repo.upsert("only_one", emb)
        results = repo.search(emb, top_k=5)
        assert len(results) == 1
        assert results[0].fingerprint_id == "only_one"

    def test_top_k_respects_limit(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        for i in range(10):
            emb = replace(_empty_embedding(), degree_1_ratio=float(i))
            repo.upsert(f"p_{i}", emb)
        results = repo.search(_empty_embedding(), top_k=3)
        assert len(results) == 3

    def test_fingerprint_id_field_is_always_set(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        emb = _empty_embedding()
        repo.upsert("stable_id", emb, metadata={"person_id": "P1"})
        results = repo.search(emb, top_k=1)
        assert results[0].fingerprint_id == "stable_id"
        # metadata exposes everything except the auto-injected id
        assert results[0].metadata == {"person_id": "P1"}

    def test_user_payload_fields_round_trip(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        emb = _empty_embedding()
        full_payload = {
            "person_id": "P-007",
            "image_path": "/data/fingerprints/james_007.bmp",
            "enrolled_at": "2026-06-15",
            "quality_score": 0.93,
        }
        repo.upsert("user_payload", emb, metadata=full_payload)
        results = repo.search(emb, top_k=1)
        assert results[0].fingerprint_id == "user_payload"
        for key, value in full_payload.items():
            assert results[0].metadata[key] == value


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_removes_point(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        repo.upsert("delete_me", _empty_embedding())
        repo.upsert("keep_me", _empty_embedding())
        assert repo.collection_size() == 2

        repo.delete("delete_me")
        assert repo.collection_size() == 1

        results = repo.search(_empty_embedding(), top_k=5)
        ids = [r.fingerprint_id for r in results]
        assert "delete_me" not in ids
        assert "keep_me" in ids


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_upsert_rejects_wrong_dim_ndarray(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        with pytest.raises(ValueError, match="shape"):
            repo.upsert("bad", np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_upsert_rejects_invalid_type(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        with pytest.raises(TypeError):
            repo.upsert("bad", "not an embedding")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Real SOCOFing data
# ---------------------------------------------------------------------------


class TestRealDataRoundtrip:
    def test_index_and_self_match(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        embeddings = _load_socofing_embeddings()
        for fid, emb in embeddings.items():
            repo.upsert(fid, emb, metadata={"fingerprint_id": fid})

        assert repo.collection_size() == len(embeddings)

        for fid, emb in embeddings.items():
            results = repo.search(emb, top_k=3)
            assert results[0].fingerprint_id == fid
            assert results[0].score > 0.99

    def test_returns_typed_coarse_match(self, repo: QdrantRepository) -> None:
        repo.ensure_collection()
        embeddings = _load_socofing_embeddings()
        for fid, emb in embeddings.items():
            repo.upsert(fid, emb)
        results = repo.search(next(iter(embeddings.values())), top_k=1)
        assert isinstance(results[0], CoarseMatch)
        assert isinstance(results[0].score, float)
        assert isinstance(results[0].fingerprint_id, str)

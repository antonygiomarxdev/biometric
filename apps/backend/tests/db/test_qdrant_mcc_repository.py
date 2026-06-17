"""Unit tests for QdrantMccRepository (Phase 21)."""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.core.types import MccCylinderHit


@pytest.fixture
def repo() -> QdrantMccRepository:
    client = QdrantClient(location=":memory:")
    r = QdrantMccRepository(client, collection="test_mcc_unit")
    r.ensure_collection()
    return r


def test_ensure_collection_is_idempotent(repo: QdrantMccRepository) -> None:
    repo.ensure_collection()
    repo.ensure_collection()


def test_bulk_insert_returns_count(repo: QdrantMccRepository) -> None:
    vectors = [np.random.rand(144).astype(np.float32) for _ in range(3)]
    n = repo.bulk_insert_cylinders("p1", "f1", "c1", vectors)
    assert n == 3


def test_knn_search_returns_hits_with_cosine_in_unit_range(repo: QdrantMccRepository) -> None:
    v = np.zeros(144, dtype=np.float32)
    v[0] = 1.0
    repo.bulk_insert_cylinders("p1", "f1", "c1", [v])
    hits = repo.knn_search([v], top_k_per_vector=1)
    assert len(hits) == 1
    assert 0.0 <= hits[0].similarity <= 1.0
    assert hits[0].person_id == "p1"


def test_aggregate_scores_by_person_normalizes(repo: QdrantMccRepository) -> None:
    hits = [
        MccCylinderHit(person_id="p1", fingerprint_id="f1", capture_id="c1", similarity=0.9),
        MccCylinderHit(person_id="p1", fingerprint_id="f1", capture_id="c1", similarity=0.8),
        MccCylinderHit(person_id="p2", fingerprint_id="f2", capture_id="c2", similarity=0.5),
    ]
    enrolled = {"p1": 10, "p2": 5}
    persons = repo.aggregate_scores_by_person(hits, enrolled)
    assert len(persons) == 2
    p1 = next(p for p in persons if p.person_id == "p1")
    assert p1.total_score == pytest.approx((0.9 + 0.8) / 10, abs=1e-5)
    p2 = next(p for p in persons if p.person_id == "p2")
    assert p2.total_score == pytest.approx(0.5 / 5, abs=1e-5)


def test_delete_by_person_removes_only_target(repo: QdrantMccRepository) -> None:
    v = np.ones(144, dtype=np.float32)
    repo.bulk_insert_cylinders("p1", "f1", "c1", [v])
    repo.bulk_insert_cylinders("p2", "f2", "c2", [v])
    removed = repo.delete_by_person("p1")
    assert removed == 1
    assert repo.count_by_person("p1") == 0
    assert repo.count_by_person("p2") == 1

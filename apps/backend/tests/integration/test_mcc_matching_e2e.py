"""E2E test: MCC matching round-trip on a real Qdrant server (Phase 21)."""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pytest
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer

from src.db.qdrant_mcc_repository import QdrantMccRepository


@pytest.fixture(scope="module")
def qdrant_server() -> Iterator[tuple[str, int]]:
    with QdrantContainer() as q:
        host = q.get_container_host_ip()
        port = int(q.get_exposed_port(6333))
        yield host, port


@pytest.fixture(scope="module")
def client(qdrant_server: tuple[str, int]) -> Iterator[QdrantClient]:
    host, port = qdrant_server
    c = QdrantClient(host=host, port=port, check_compatibility=False)
    yield c
    c.close()


@pytest.fixture
def repo(client: QdrantClient) -> QdrantMccRepository:
    collection = "test_e2e_mcc"
    r = QdrantMccRepository(client, collection=collection, vector_size=144)
    r.ensure_collection()
    r._client.delete_collection(collection_name=collection)
    r.ensure_collection()
    return r


def test_enroll_and_search_round_trip(repo: QdrantMccRepository) -> None:
    rng = np.random.default_rng(42)
    enrolled: list[tuple[str, np.ndarray]] = []
    for person_id in range(3):
        base = rng.standard_normal(144).astype(np.float32)
        base /= np.linalg.norm(base)
        cylinders = []
        for _ in range(10):
            noise = rng.standard_normal(144).astype(np.float32) * 0.05
            v = base + noise
            v /= np.linalg.norm(v)
            cylinders.append(v)
        n = repo.bulk_insert_cylinders(
            person_id=f"p{person_id}",
            fingerprint_id=f"f{person_id}",
            capture_id=f"c{person_id}",
            vectors=cylinders,
        )
        assert n == 10
        enrolled.append((f"p{person_id}", base))

    query_vecs = [enrolled[0][1]]
    hits = repo.knn_search(query_vecs, top_k_per_vector=5)
    assert hits, "Expected at least one hit"

    persons = repo.aggregate_scores_by_person(
        hits,
        enrolled_counts={"p0": 10, "p1": 10, "p2": 10},
    )
    assert persons, "Expected at least one person hit"
    assert persons[0].person_id == "p0"


def test_search_throughput_under_one_second(repo: QdrantMccRepository) -> None:
    rng = np.random.default_rng(7)
    for person_id in range(10):
        cylinders = []
        for _ in range(20):
            v = rng.standard_normal(144).astype(np.float32)
            v /= np.linalg.norm(v)
            cylinders.append(v)
        repo.bulk_insert_cylinders(
            person_id=f"p{person_id}",
            fingerprint_id=f"f{person_id}",
            capture_id=f"c{person_id}",
            vectors=cylinders,
        )

    query = [rng.standard_normal(144).astype(np.float32) for _ in range(5)]
    t0 = time.monotonic()
    hits = repo.knn_search(query, top_k_per_vector=5)
    elapsed = time.monotonic() - t0

    assert hits
    assert elapsed < 1.0, f"Search took {elapsed:.2f}s (expected <1s)"

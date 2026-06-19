"""E2E test: NIST Bozorth3 pair matching round-trip on a real Qdrant server (Phase 27)."""
from __future__ import annotations

import time
from typing import Iterator

import pytest
from qdrant_client import QdrantClient
from testcontainers.qdrant import QdrantContainer

from src.db.qdrant_pair_repository import QdrantPairRepository


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
def repo(client: QdrantClient) -> QdrantPairRepository:
    collection = "test_e2e_pairs"
    r = QdrantPairRepository(client)
    r._client.delete_collection(collection_name=collection)
    r.ensure_collection()
    return r


def _make_pair(i: int, j: int, dx: float = 0.1, dy: float = 0.2, dtheta: float = 0.3, distance: float = 0.5) -> dict:
    return {
        "i": i,
        "j": j,
        "mi_x": 0.1, "mi_y": 0.2, "mi_angle": 0.0,
        "mj_x": 0.3, "mj_y": 0.4, "mj_angle": 0.1,
        "dx": dx, "dy": dy, "dtheta": dtheta, "distance": distance,
        "type_pair": 11,
    }


def test_enroll_and_search_round_trip(repo: QdrantPairRepository) -> None:
    """Enroll pairs for 3 persons, then KNN-search and verify hits."""
    n_per = 10
    for person_id in range(3):
        pairs = [_make_pair(i, i + 1, dx=0.1 * person_id) for i in range(n_per)]
        n = repo.bulk_insert_pairs(
            person_id=f"p{person_id}",
            fingerprint_id=f"f{person_id}",
            capture_id=f"c{person_id}",
            pair_dicts=pairs,
        )
        assert n == n_per

    query = [_make_pair(0, 1, dx=0.0)]
    hits = repo.knn_search_pairs(query, top_k_per_vector=5)
    assert hits, "Expected at least one hit"
    assert all("person_id" in h for h in hits)


def test_search_throughput_under_one_second(repo: QdrantPairRepository) -> None:
    """Enroll many pairs and verify KNN search is fast."""
    for person_id in range(10):
        pairs = [_make_pair(i, i + 1) for i in range(50)]
        repo.bulk_insert_pairs(
            person_id=f"p{person_id}",
            fingerprint_id=f"f{person_id}",
            capture_id=f"c{person_id}",
            pair_dicts=pairs,
        )

    query = [_make_pair(0, 1) for _ in range(5)]
    t0 = time.monotonic()
    hits = repo.knn_search_pairs(query, top_k_per_vector=10)
    elapsed = time.monotonic() - t0

    assert hits
    assert elapsed < 1.0, f"Search took {elapsed:.2f}s (expected <1s)"

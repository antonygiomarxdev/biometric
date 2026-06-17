"""Protocol conformance test for IMccMatcher (Phase 21)."""

from __future__ import annotations

import numpy as np

from src.core.interfaces import IMccMatcher
from src.core.types import MccCylinderHit, MccPersonHit


class _FakeAdapter:
    """Minimal in-memory implementation satisfying IMccMatcher."""

    def ensure_collection(self) -> None: ...
    def bulk_insert_cylinders(self, person_id, fingerprint_id, capture_id, vectors): return len(vectors)
    def knn_search(self, query_vectors, top_k_per_vector=5): return []
    def aggregate_scores_by_person(self, hits, enrolled_counts): return []
    def delete_by_person(self, person_id): return 0


def test_fake_adapter_satisfies_protocol() -> None:
    adapter: IMccMatcher = _FakeAdapter()
    assert hasattr(adapter, "ensure_collection")
    assert hasattr(adapter, "bulk_insert_cylinders")
    assert hasattr(adapter, "knn_search")
    assert hasattr(adapter, "aggregate_scores_by_person")
    assert hasattr(adapter, "delete_by_person")

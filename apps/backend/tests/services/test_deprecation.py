"""Test that the Delaunay path is marked deprecated (Phase 21)."""

from __future__ import annotations

from src.services.rag_matching_service import QdrantRagMatchingService


def test_qdrant_rag_matching_service_has_deprecated_flag() -> None:
    assert QdrantRagMatchingService.__deprecated__ is True

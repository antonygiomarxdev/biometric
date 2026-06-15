"""
QdrantRepository — persistence gateway for graph embedding vectors.

Clean Architecture: depends only on Qdrant client (infra), never on
domain models.  Collection creation, point insertion, and vector
search are the three public operations.

Uses deterministic integer point IDs (SHA-256 hash of fingerprint_id)
so both in-memory (:memory:) and server modes work identically.
"""

from __future__ import annotations

import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

COLLECTION_NAME: str = "ridge_graphs"
EMBEDDING_DIM: int = 22


def _point_id(fingerprint_id: str) -> int:
    """Deterministic integer point ID derived from fingerprint_id."""
    return int(hashlib.sha256(fingerprint_id.encode()).hexdigest()[:16], 16)


class QdrantRepository:
    """Persistence gateway for graph embedding vectors.

    Usage::

        repo = QdrantRepository(host="localhost", port=6333)
        repo.ensure_collection()
        repo.insert(fingerprint_id="abc123", vector=emb, payload={...})
        results = repo.search(vector=query_emb, top_k=100)
    """

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        self._client = QdrantClient(host=host, port=port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, collection: str = COLLECTION_NAME) -> None:
        """Create the collection if it does not already exist."""
        try:
            self._client.create_collection(
                collection_name=collection,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
        except (ValueError, UnexpectedResponse):
            pass

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert(
        self,
        fingerprint_id: str,
        vector: list[float],
        payload: dict[str, Any] | None = None,
        collection: str = COLLECTION_NAME,
    ) -> None:
        """Insert (or overwrite) a single embedding point.

        Args:
            fingerprint_id: Unique identifier for the fingerprint (stored
                in payload, used to derive the integer point ID).
            vector: The dense embedding vector.
            payload: Optional metadata dict (e.g. person_id, image_path).
            collection: Target collection name.
        """
        p = dict(payload) if payload else {}
        p["fingerprint_id"] = fingerprint_id

        self._client.upsert(
            collection_name=collection,
            points=[
                qdrant_models.PointStruct(
                    id=_point_id(fingerprint_id),
                    vector=vector,
                    payload=p,
                )
            ],
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        vector: list[float],
        top_k: int = 100,
        collection: str = COLLECTION_NAME,
    ) -> list[tuple[str, float]]:
        """Return the *top_k* most similar embedding IDs and their scores.

        Returns:
            List of ``(fingerprint_id, cosine_similarity)`` tuples,
            sorted descending by score.
        """
        hits = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points
        results: list[tuple[str, float]] = []
        for hit in hits:
            fid: str = hit.payload.get("fingerprint_id", str(hit.id))  # type: ignore[union-attr]
            results.append((fid, hit.score))
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, fingerprint_id: str, collection: str = COLLECTION_NAME) -> None:
        """Remove a single embedding point by fingerprint_id."""
        self._client.delete(
            collection_name=collection,
            points_selector=qdrant_models.PointIdsList(
                points=[_point_id(fingerprint_id)],
            ),
        )

    def collection_size(self, collection: str = COLLECTION_NAME) -> int:
        """Return the number of points in the collection."""
        info = self._client.get_collection(collection)
        return info.points_count or 0

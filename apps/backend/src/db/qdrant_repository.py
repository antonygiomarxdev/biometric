"""
QdrantRepository — adapter for the ICoarseMatcher port.

Clean Architecture: this is an *infrastructure adapter* (right side of
the hexagon).  Services depend on :class:`~src.core.interfaces.ICoarseMatcher`,
never on this class directly.  Swap this for a Weaviate or Milvus adapter
without touching domain code.

Uses deterministic integer point IDs (SHA-256 hash of fingerprint_id)
so both in-memory (``:memory:``) and server modes work identically.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.interfaces import ICoarseMatcher
from src.core.types import CoarseMatch, GraphEmbedding

COLLECTION_NAME: str = "ridge_graphs"
_DEFAULT_HOST: str = "localhost"
_DEFAULT_PORT: int = 6333
_VECTOR_DIM: int = GraphEmbedding.EMBEDDING_DIM


def _point_id(fingerprint_id: str) -> int:
    """Deterministic integer point ID derived from *fingerprint_id*."""
    return int(hashlib.sha256(fingerprint_id.encode()).hexdigest()[:16], 16)


class QdrantRepository(ICoarseMatcher):
    """Adapter for ICoarseMatcher backed by Qdrant.

    Args:
        client: An already-constructed :class:`QdrantClient`.  Injected
            so tests can pass ``QdrantClient(location=":memory:")``
            without going through the default host/port.
        collection: Collection name (overridable for multi-tenant setups).
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str = COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_host(
        cls,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        collection: str = COLLECTION_NAME,
    ) -> QdrantRepository:
        """Construct from a host/port pair (production convenience)."""
        return cls(QdrantClient(host=host, port=port), collection=collection)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self._collection_exists():
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=_VECTOR_DIM,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def _collection_exists(self) -> bool:
        """Return True iff the collection is present in the server."""
        try:
            self._client.get_collection(self._collection)
        except (ValueError, UnexpectedResponse):
            return False
        return True

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        fingerprint_id: str,
        embedding: np.ndarray | GraphEmbedding,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Insert or update one fingerprint embedding.

        Args:
            fingerprint_id: Unique identifier.  Stored in the payload
                automatically; *metadata* is merged on top of it.
            embedding: Dense vector.  Accepts a :class:`GraphEmbedding`
                (preferred) or a raw :class:`np.ndarray` of size
                :data:`GraphEmbedding.EMBEDDING_DIM`.
            metadata: Optional user metadata (person_id, image_path, …).
        """
        vector = self._coerce_vector(embedding)
        payload: dict[str, Any] = dict(metadata or {})
        payload["fingerprint_id"] = fingerprint_id

        self._client.upsert(
            collection_name=self._collection,
            points=[
                qdrant_models.PointStruct(
                    id=_point_id(fingerprint_id),
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    @staticmethod
    def _coerce_vector(embedding: np.ndarray | GraphEmbedding) -> list[float]:
        """Accept either a GraphEmbedding or a raw ndarray."""
        if isinstance(embedding, GraphEmbedding):
            return embedding.to_vector().tolist()  # type: ignore[no-any-return]
        if isinstance(embedding, np.ndarray):
            arr = embedding.astype(np.float32, copy=False)
            if arr.shape != (_VECTOR_DIM,):
                msg = f"Embedding has shape {arr.shape}, expected ({_VECTOR_DIM},)"
                raise ValueError(msg)
            return arr.tolist()  # type: ignore[no-any-return]
        msg = f"embedding must be GraphEmbedding or np.ndarray, got {type(embedding)}"
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self, embedding: np.ndarray | GraphEmbedding, top_k: int = 100
    ) -> list[CoarseMatch]:
        """Return the *top_k* most similar candidates ranked by score."""
        vector = self._coerce_vector(embedding)
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points

        results: list[CoarseMatch] = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                CoarseMatch(
                    fingerprint_id=payload.get("fingerprint_id", str(hit.id)),
                    score=hit.score,
                    metadata={k: v for k, v in payload.items() if k != "fingerprint_id"},
                )
            )
        return results

    def collection_size(self) -> int:
        """Return the number of points in the collection."""
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, fingerprint_id: str) -> None:
        """Remove a single embedding point by *fingerprint_id*."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.PointIdsList(
                points=[_point_id(fingerprint_id)],
            ),
        )

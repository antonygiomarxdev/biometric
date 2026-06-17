"""
QdrantChunkRepository — adapter for the IChunkMatcher port.

Clean Architecture: infrastructure adapter (right side of the hexagon).
Services depend on :class:`~src.core.interfaces.IChunkMatcher`, never
on this class directly. Swap for Milvus/Pinecone without touching domain.

Phase 15: stores Delaunay triangle chunks (9-dim) in a Qdrant collection
with person_id + weight payload for forensic aggregation. Deterministic
point IDs (SHA-256) make upserts idempotent.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import PointId
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import config
from src.core.interfaces import IChunkMatcher
from src.core.types import ChunkHit, PersonHit, TripletVector

logger = logging.getLogger(__name__)

COLLECTION_NAME: str = "fingerprint_chunks"
_DEFAULT_HOST: str = "localhost"
_DEFAULT_PORT: int = 6333
DELAUNAY_DIM: int = 9


def _chunk_point_id(
    person_id: str, fingerprint_id: str, chunk_index: int, chunk_type: str = "delaunay",
) -> int:
    """Deterministic integer point ID from quadruple."""
    key = f"{person_id}:{fingerprint_id}:{chunk_type}:{chunk_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


class QdrantChunkRepository(IChunkMatcher):
    """Qdrant-backed chunk store for Delaunay/MCC invariant chunks.

    Replaces ``RagVectorRepository`` (PostgreSQL) for the search hot-path.
    One Qdrant point per chunk; payload holds forensic weight and identity.

    Args:
        client: Already-constructed :class:`QdrantClient`. Injected so
            tests can pass ``QdrantClient(location=":memory:")``.
        collection: Collection name (overridable for multi-tenant).
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
    ) -> "QdrantChunkRepository":
        """Construct from a host/port pair, falling back to in-memory on failure."""
        try:
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            return cls(client, collection=collection)
        except Exception as exc:
            logger.warning(
                "Qdrant at %s:%s unreachable (%s). Falling back to in-memory storage.",
                host, port, exc,
            )
            return cls(QdrantClient(location=":memory:"), collection=collection)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection with HNSW config if it does not exist."""
        if self._collection_exists():
            return
        idx_cfg = config.qdrant_index
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=DELAUNAY_DIM,
                distance=qdrant_models.Distance.COSINE,
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=idx_cfg.hnsw_m,
                    ef_construct=idx_cfg.hnsw_ef_construct,
                ),
            ),
        )
        # Payload indexes for fast filtering
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="person_id",
            field_type=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="chunk_type",
            field_type=qdrant_models.PayloadSchemaType.KEYWORD,
        )

    def _collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._collection)
        except (ValueError, UnexpectedResponse):
            return False
        return True

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def bulk_insert_chunks(
        self,
        person_id: str,
        fingerprint_id: str,
        chunks: list[TripletVector],
        chunk_type: str = "delaunay",
        capture_id: str = "",
        graph_id: str = "",
    ) -> int:
        """Insert N chunks for one fingerprint enrollment.

        Args:
            person_id: Person owning these chunks.
            fingerprint_id: Fingerprint identifier.
            chunks: TripletVector list from RagTripletVectorizer.
            chunk_type: "delaunay" or "mcc".

        Returns:
            Number of points inserted.
        """
        if not chunks:
            return 0

        points: list[qdrant_models.PointStruct] = []
        for i, chunk in enumerate(chunks):
            vec = np.array(chunk.features, dtype=np.float32).tolist()
            point_id = _chunk_point_id(person_id, fingerprint_id, i, chunk_type)
            payload: dict[str, Any] = {
                "person_id": person_id,
                "fingerprint_id": fingerprint_id,
                "capture_id": capture_id,
                "graph_id": graph_id,
                "chunk_type": chunk_type,
                "weight": chunk.weight,
                "chunk_index": i,
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.debug(
            "Inserted %d chunks for %s/%s (type=%s)",
            len(points), person_id, fingerprint_id, chunk_type,
        )
        return len(points)

    # ------------------------------------------------------------------
    # Read / Search
    # ------------------------------------------------------------------

    def weighted_knn_search(
        self,
        query_chunks: list[TripletVector],
        top_k_per_chunk: int = 5,
        chunk_type: str = "delaunay",
    ) -> list[ChunkHit]:
        """Per-chunk KNN, return unscored hits with weighted similarity.

        Each probe chunk queries Qdrant for the *top_k_per_chunk* nearest
        chunks of the given *chunk_type*. Results are merged into a flat
        list of :class:`ChunkHit`.
        """
        if not query_chunks:
            return []

        all_hits: list[ChunkHit] = []
        for chunk in query_chunks:
            vector = np.array(chunk.features, dtype=np.float32).tolist()
            try:
                results = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    limit=top_k_per_chunk,
                    with_payload=True,
                    query_filter=qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="chunk_type",
                                match=qdrant_models.MatchValue(value=chunk_type),
                            ),
                        ],
                    ),
                ).points
            except Exception as exc:
                logger.warning("Qdrant search error: %s", exc)
                continue

            for hit in results:
                payload = hit.payload or {}
                similarity = float(hit.score)
                weight = float(payload.get("weight", 1.0))
                all_hits.append(
                    ChunkHit(
                        person_id=str(payload.get("person_id", "")),
                        fingerprint_id=str(payload.get("fingerprint_id", "")),
                        capture_id=str(payload.get("capture_id", "")),
                        graph_id=str(payload.get("graph_id", "")),
                        chunk_type=str(payload.get("chunk_type", chunk_type)),
                        weight=weight,
                        similarity=similarity,
                        weighted_score=similarity * weight,
                    )
                )

        # Deduplicate by (person_id, fingerprint_id) keeping highest score
        deduped = self._deduplicate_chunk_hits(all_hits)

        return sorted(
            deduped,
            key=lambda h: h.weighted_score,
            reverse=True,
        )

    def aggregate_scores_by_person(
        self,
        hits: list[ChunkHit],
    ) -> list[PersonHit]:
        """Group chunk hits by *person_id*, sum weighted scores.

        Returns sorted list of :class:`PersonHit`.
        """
        groups: dict[str, dict[str, Any]] = {}
        for hit in hits:
            entry = groups.setdefault(
                hit.person_id,
                {"person_id": hit.person_id, "total_score": 0.0, "hits": 0, "fps": set()},
            )
            entry["total_score"] += hit.weighted_score
            entry["hits"] += 1
            entry["fps"].add(hit.fingerprint_id)

        result = [
            PersonHit(
                person_id=g["person_id"],
                total_score=g["total_score"],
                hits=g["hits"],
                contributing_fingerprints=sorted(g["fps"]),
            )
            for g in groups.values()
        ]
        return sorted(result, key=lambda r: r.total_score, reverse=True)

    @staticmethod
    def _deduplicate_chunk_hits(hits: list[ChunkHit]) -> list[ChunkHit]:
        """For each (person_id, fingerprint_id), keep the highest score."""
        best: dict[tuple[str, str], ChunkHit] = {}
        for hit in hits:
            key = (hit.person_id, hit.fingerprint_id)
            if key not in best or hit.weighted_score > best[key].weighted_score:
                best[key] = hit
        return list(best.values())

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_by_person(self, person_id: str) -> int:
        """Remove all chunks for a person. Returns count of deleted points."""
        filter_ = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="person_id",
                    match=qdrant_models.MatchValue(value=person_id),
                ),
            ],
        )
        try:
            count_before = self._client.count(
                collection_name=self._collection,
                count_filter=filter_,
            ).count or 0
            self._client.delete(
                collection_name=self._collection,
                points_selector=qdrant_models.FilterSelector(filter=filter_),
            )
            logger.debug("Deleted %d chunks for person_id=%s", count_before, person_id)
            return count_before
        except Exception as exc:
            logger.warning("Delete error for %s: %s", person_id, exc)
            return 0

    def delete_by_fingerprint(self, fingerprint_id: str) -> int:
        """Remove all chunks for a specific fingerprint."""
        filter_ = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="fingerprint_id",
                    match=qdrant_models.MatchValue(value=fingerprint_id),
                ),
            ],
        )
        try:
            count_before = self._client.count(
                collection_name=self._collection,
                count_filter=filter_,
            ).count or 0
            self._client.delete(
                collection_name=self._collection,
                points_selector=qdrant_models.FilterSelector(filter=filter_),
            )
            logger.debug("Deleted %d chunks for fingerprint_id=%s", count_before, fingerprint_id)
            return count_before
        except Exception as exc:
            logger.warning("Delete error for %s: %s", fingerprint_id, exc)
            return 0

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def collection_size(self, chunk_type: str | None = None) -> int:
        """Return point count. Optionally filtered by chunk_type."""
        if chunk_type is not None:
            info = self._client.count(
                collection_name=self._collection,
                count_filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="chunk_type",
                            match=qdrant_models.MatchValue(value=chunk_type),
                        ),
                    ],
                ),
            )
            return info.count or 0
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def scroll_all(
        self,
        chunk_type: str | None = None,
        limit: int = 100,
    ) -> list[ChunkHit]:
        """Scroll through ALL points, paginating internally.

        ``limit`` is the *page* size for each network round-trip.
        The function follows ``offset`` until the server reports no
        more results, returning the full collection.
        """
        if chunk_type is not None:
            scroll_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="chunk_type",
                        match=qdrant_models.MatchValue(value=chunk_type),
                    ),
                ],
            )
        else:
            scroll_filter = None

        hits: list[ChunkHit] = []
        offset: PointId | None = None
        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
                scroll_filter=scroll_filter,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                hits.append(
                    ChunkHit(
                        person_id=str(payload.get("person_id", "")),
                        fingerprint_id=str(payload.get("fingerprint_id", "")),
                        capture_id=str(payload.get("capture_id", "")),
                        graph_id=str(payload.get("graph_id", "")),
                        chunk_type=str(payload.get("chunk_type", "")),
                        weight=float(payload.get("weight", 1.0)),
                        similarity=1.0,
                        weighted_score=float(payload.get("weight", 1.0)),
                    )
                )
            if next_offset is None:
                break
            offset = next_offset
        return hits

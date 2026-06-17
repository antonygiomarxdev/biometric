"""
QdrantMccRepository — adapter for the IMccMatcher port (Phase 21).

Stores L2-normalized MCC cylinder vectors (144-D by default) in a
dedicated Qdrant collection. Unlike :class:`QdrantChunkRepository`
(Delaunay triplets), this adapter:

* Uses a separate collection with vector size 144.
* Aggregates cylinder hits per-person with per-fingerprint normalization
  (eliminates bias toward enrollees with more minutiae).
* Stores payload ``{person_id, fingerprint_id, capture_id}`` for join-back.

Clean Architecture: infrastructure adapter. Services depend on
:class:`~src.core.interfaces.IMccMatcher`, never on this class directly.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.core.config import config
from src.core.interfaces import IMccMatcher
from src.core.types import MccCylinderHit, MccPersonHit

logger = logging.getLogger(__name__)

COLLECTION_NAME: str = "mcc_cylinders"
_DEFAULT_HOST: str = "localhost"
_DEFAULT_PORT: int = 6333
DEFAULT_VECTOR_SIZE: int = 144


def _cylinder_point_id(
    person_id: str,
    fingerprint_id: str,
    capture_id: str,
    cylinder_index: int,
) -> int:
    """Deterministic integer point ID from enrollment quadruple."""
    key = f"{person_id}:{fingerprint_id}:{capture_id}:{cylinder_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


class QdrantMccRepository(IMccMatcher):
    """Qdrant-backed cylinder store for MCC descriptors.

    Args:
        client: Already-constructed :class:`QdrantClient`. Injected so
            tests can pass ``QdrantClient(location=":memory:")``.
        collection: Collection name (overridable for multi-tenant).
        vector_size: Descriptor dimension (144 by default).
    """

    def __init__(
        self,
        client: QdrantClient,
        collection: str = COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> None:
        self._client = client
        self._collection = collection
        self._vector_size = vector_size

    @classmethod
    def from_host(
        cls,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
        collection: str = COLLECTION_NAME,
        vector_size: int = DEFAULT_VECTOR_SIZE,
    ) -> "QdrantMccRepository":
        """Construct from a host/port pair, falling back to in-memory on failure."""
        try:
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            return cls(client, collection=collection, vector_size=vector_size)
        except Exception as exc:
            logger.warning(
                "Qdrant at %s:%s unreachable (%s). Falling back to in-memory storage.",
                host, port, exc,
            )
            return cls(
                QdrantClient(location=":memory:"),
                collection=collection,
                vector_size=vector_size,
            )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _collection_exists(self) -> bool:
        try:
            collections = self._client.get_collections().collections
            return any(c.name == self._collection for c in collections)
        except Exception:
            return False

    def ensure_collection(self) -> None:
        """Create the collection with HNSW config if it does not exist."""
        if self._collection_exists():
            return
        idx_cfg = config.qdrant_index
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=self._vector_size,
                distance=qdrant_models.Distance.COSINE,
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=idx_cfg.hnsw_m,
                    ef_construct=idx_cfg.hnsw_ef_construct,
                    full_scan_threshold=idx_cfg.hnsw_full_scan_threshold,
                ),
                quantization_config=qdrant_models.ScalarQuantization(
                    scalar=qdrant_models.ScalarQuantizationConfig(
                        type=qdrant_models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                default_segment_number=2,
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="person_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name="fingerprint_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection %s (size=%d)", self._collection, self._vector_size)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def bulk_insert_cylinders(
        self,
        person_id: str,
        fingerprint_id: str,
        capture_id: str,
        vectors: list[np.ndarray],
        cylinder_positions: list[tuple[int, int, float]] | None = None,
    ) -> int:
        """Insert N cylinder vectors. Returns the count inserted.

        When ``cylinder_positions`` is provided (same length as ``vectors``),
        each point's payload includes ``x``, ``y``, and ``angle`` so that
        :meth:`knn_search` can surface the candidate minutia's spatial
        location for match-trace rendering (Phase 23).
        """
        if not vectors:
            return 0
        points: list[qdrant_models.PointStruct] = []
        for i, v in enumerate(vectors):
            payload: dict[str, object] = {
                "person_id": person_id,
                "fingerprint_id": fingerprint_id,
                "capture_id": capture_id,
            }
            if cylinder_positions is not None:
                x, y, angle = cylinder_positions[i]
                payload["x"] = int(x)
                payload["y"] = int(y)
                payload["angle"] = float(angle)
            points.append(
                qdrant_models.PointStruct(
                    id=_cylinder_point_id(person_id, fingerprint_id, capture_id, i),
                    vector=v.astype(np.float32).tolist(),
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def knn_search(
        self,
        query_vectors: list[np.ndarray],
        top_k_per_vector: int = 5,
    ) -> list[MccCylinderHit]:
        """For each query cylinder, return top-K similar cylinders.

        The returned ``MccCylinderHit`` includes the loop index
        (``query_cylinder_index``) so callers can correlate hits back
        to the probe minutia that generated the query, and spatial
        position (``candidate_x``/``candidate_y``/``candidate_angle``)
        from the Qdrant payload for match-trace rendering.
        """
        if not query_vectors:
            return []
        all_hits: list[MccCylinderHit] = []
        for query_idx, qv in enumerate(query_vectors):
            response = self._client.query_points(
                collection_name=self._collection,
                query=qv.astype(np.float32).tolist(),
                limit=top_k_per_vector,
                with_payload=True,
            )
            for hit in response.points:
                payload = hit.payload or {}
                all_hits.append(
                    MccCylinderHit(
                        person_id=str(payload.get("person_id", "")),
                        fingerprint_id=str(payload.get("fingerprint_id", "")),
                        capture_id=str(payload.get("capture_id", "")),
                        similarity=float(hit.score),
                        query_cylinder_index=query_idx,
                        candidate_x=int(payload.get("x", 0)),
                        candidate_y=int(payload.get("y", 0)),
                        candidate_angle=float(payload.get("angle", 0.0)),
                    )
                )
        return all_hits

    def aggregate_scores_by_person(
        self,
        hits: list[MccCylinderHit],
        enrolled_counts: dict[str, int],
    ) -> list[MccPersonHit]:
        """Group cylinder hits by person, normalize by per-person count."""
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        fps: dict[str, set[str]] = defaultdict(set)
        for h in hits:
            sums[h.person_id] += h.similarity
            counts[h.person_id] += 1
            fps[h.person_id].add(h.fingerprint_id)

        norm_mode = config.matching.score_normalization
        persons: list[MccPersonHit] = []
        for person_id, total in sums.items():
            if norm_mode == "fingerprint":
                denom = enrolled_counts.get(person_id, 1) or 1
                score = total / denom
            else:
                score = total
            persons.append(
                MccPersonHit(
                    person_id=person_id,
                    total_score=score,
                    hits=counts[person_id],
                    contributing_fingerprints=sorted(fps[person_id]),
                )
            )
        persons.sort(key=lambda p: p.total_score, reverse=True)
        return persons

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_by_person(self, person_id: str) -> int:
        """Remove all cylinders for a person. Returns count of deleted points."""
        filter_ = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="person_id",
                    match=qdrant_models.MatchValue(value=person_id),
                )
            ]
        )
        before = self.count_by_person(person_id)
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.FilterSelector(filter=filter_),
        )
        return before

    def count_by_person(self, person_id: str) -> int:
        """Count cylinders stored for a given person."""
        result = self._client.count(
            collection_name=self._collection,
            count_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="person_id",
                        match=qdrant_models.MatchValue(value=person_id),
                    )
                ]
            ),
        )
        return int(result.count)

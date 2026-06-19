"""
QdrantPairRepository — adapter for the NIST Bozorth3 pair matcher (Phase 27).

Stores L2-normalized 5-D pair vectors in the ``pair_features`` Qdrant
collection. The cylinder matcher (Phase 21) and triplet matcher (Phase 25)
were removed; this is the only matcher. See
``docs/adr/009-remove-cylinders.md`` for the decision.

Clean Architecture: infrastructure adapter. Services depend on the
repository directly (it has a focused, single-collection API).
"""
from __future__ import annotations

import hashlib
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.core.config import config

logger = logging.getLogger(__name__)

PAIR_COLLECTION_NAME: str = "pair_features"
PAIR_VECTOR_SIZE: int = 5
_DEFAULT_HOST: str = "localhost"
_DEFAULT_PORT: int = 6333


def _pair_point_id(
    person_id: str,
    fingerprint_id: str,
    capture_id: str,
    pair_index: int,
) -> int:
    """Deterministic integer point ID from enrollment quadruple."""
    key = f"pair:{person_id}:{fingerprint_id}:{capture_id}:{pair_index}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


class QdrantPairRepository:
    """Qdrant-backed store for NIST Bozorth3 pair descriptors.

    Args:
        client: Already-constructed :class:`QdrantClient`. Injected so
            tests can pass ``QdrantClient(location=":memory:")``.
    """

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    @classmethod
    def from_host(
        cls,
        host: str = _DEFAULT_HOST,
        port: int = _DEFAULT_PORT,
    ) -> QdrantPairRepository:
        """Construct from a host/port pair, falling back to in-memory on failure."""
        try:
            client = QdrantClient(host=host, port=port)
            client.get_collections()
            return cls(client)
        except Exception as exc:
            logger.warning(
                "Qdrant at %s:%s unreachable (%s). Falling back to in-memory storage.",
                host, port, exc,
            )
            return cls(QdrantClient(location=":memory:"))

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _collection_exists(self) -> bool:
        try:
            collections = self._client.get_collections().collections
            return any(c.name == PAIR_COLLECTION_NAME for c in collections)
        except Exception:
            return False

    def ensure_collection(self) -> None:
        """Create the pair_features collection with 5-D vectors if it does not exist."""
        if self._collection_exists():
            return
        idx_cfg = config.qdrant_index
        self._client.create_collection(
            collection_name=PAIR_COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=PAIR_VECTOR_SIZE,
                distance=qdrant_models.Distance.COSINE,
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=idx_cfg.hnsw_m,
                    ef_construct=idx_cfg.hnsw_ef_construct,
                    full_scan_threshold=idx_cfg.hnsw_full_scan_threshold,
                ),
            ),
            optimizers_config=qdrant_models.OptimizersConfigDiff(
                default_segment_number=2,
            ),
        )
        self._client.create_payload_index(
            collection_name=PAIR_COLLECTION_NAME,
            field_name="person_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=PAIR_COLLECTION_NAME,
            field_name="capture_id",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        logger.info("Created Qdrant collection %s (size=%d)", PAIR_COLLECTION_NAME, PAIR_VECTOR_SIZE)

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def bulk_insert_pairs(
        self,
        person_id: str,
        fingerprint_id: str,
        capture_id: str,
        pair_dicts: list[dict],
    ) -> int:
        """Insert pair feature vectors into the pair_features collection."""
        if not pair_dicts:
            return 0
        from src.processing.pair_extractor import pair_to_vector

        points: list[qdrant_models.PointStruct] = []
        for i, p in enumerate(pair_dicts):
            vec = pair_to_vector(p)
            points.append(
                qdrant_models.PointStruct(
                    id=_pair_point_id(person_id, fingerprint_id, capture_id, i),
                    vector=vec,
                    payload={
                        "person_id": person_id,
                        "fingerprint_id": fingerprint_id,
                        "capture_id": capture_id,
                        "i_idx": p["i"],
                        "j_idx": p["j"],
                        "mi_x": p["mi_x"],
                        "mi_y": p["mi_y"],
                        "mi_angle": p["mi_angle"],
                        "mj_x": p["mj_x"],
                        "mj_y": p["mj_y"],
                        "mj_angle": p["mj_angle"],
                        "dx": p["dx"],
                        "dy": p["dy"],
                        "dtheta": p["dtheta"],
                        "distance": p["distance"],
                        "type_pair": p["type_pair"],
                    },
                )
            )
        self._client.upsert(collection_name=PAIR_COLLECTION_NAME, points=points, wait=False)
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def knn_search_pairs(
        self,
        query_vectors: list[list[float]],
        top_k_per_vector: int = 10,
    ) -> list[dict]:
        """For each query pair vector, return top-K similar pairs."""
        if not query_vectors:
            return []
        all_hits: list[dict] = []
        for query_idx, qv in enumerate(query_vectors):
            response = self._client.query_points(
                collection_name=PAIR_COLLECTION_NAME,
                query=qv,
                limit=top_k_per_vector,
                with_payload=True,
            )
            for hit in response.points:
                payload = hit.payload or {}
                all_hits.append({
                    "query_pair_index": query_idx,
                    "similarity": float(hit.score),
                    "person_id": str(payload.get("person_id", "")),
                    "fingerprint_id": str(payload.get("fingerprint_id", "")),
                    "capture_id": str(payload.get("capture_id", "")),
                    "i_idx": int(payload.get("i_idx", 0)),
                    "j_idx": int(payload.get("j_idx", 0)),
                    "mi_x": float(payload.get("mi_x", 0.0)),
                    "mi_y": float(payload.get("mi_y", 0.0)),
                    "mi_angle": float(payload.get("mi_angle", 0.0)),
                    "mj_x": float(payload.get("mj_x", 0.0)),
                    "mj_y": float(payload.get("mj_y", 0.0)),
                    "mj_angle": float(payload.get("mj_angle", 0.0)),
                    "dx": float(payload.get("dx", 0.0)),
                    "dy": float(payload.get("dy", 0.0)),
                    "dtheta": float(payload.get("dtheta", 0.0)),
                })
        return all_hits

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def delete_by_person(self, person_id: str) -> int:
        """Remove all pairs for a person. Returns count of deleted points."""
        filter_ = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="person_id",
                    match=qdrant_models.MatchValue(value=person_id),
                )
            ]
        )
        before = self._client.count(
            collection_name=PAIR_COLLECTION_NAME,
            count_filter=filter_,
        )
        self._client.delete(
            collection_name=PAIR_COLLECTION_NAME,
            points_selector=qdrant_models.FilterSelector(filter=filter_),
        )
        return int(before.count)

    def count_by_person(self, person_id: str) -> int:
        """Count pairs stored for a given person."""
        result = self._client.count(
            collection_name=PAIR_COLLECTION_NAME,
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

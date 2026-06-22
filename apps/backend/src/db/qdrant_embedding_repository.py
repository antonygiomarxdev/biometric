from __future__ import annotations

import hashlib
import logging
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import config

logger = logging.getLogger(__name__)


class QdrantPayload(TypedDict, total=False):
    """Subset of the Qdrant payload shape we actually write/read.

    Fields are all optional (``total=False``) because callers may
    omit ``finger_name``.  Qdrant itself accepts arbitrary JSON, but
    we keep a strict superset of what we use in production.
    """

    person_id: str
    capture_id: str
    finger_name: str
    fingerprint_id: str


class QdrantHit(TypedDict):
    """Internal result type returned by ``search`` and consumed by
    ``EmbeddingService.search``.
    """

    fingerprint_id: str
    score: float
    payload: QdrantPayload


class QdrantEmbeddingRepository:
    """Cosine-similarity vector store for 512-D fingerprint embeddings.

    The class is purely about *reading and writing* the embedding
    collection (upsert, search, delete single points, count).
    Lifecycle operations (creating, dropping, or migrating a
    collection) are intentionally NOT methods on this class.

    Why: A ``drop_old=True`` flag on a runtime method like
    ``ensure_collection`` makes it trivially possible for a normal
    request handler to silently nuke a production gallery on the
    next dependency-injection initialisation.  That is an
    antipattern — destructive operations must be explicit, batch
    scripts, and gated by human review.

    Where to do destructive operations instead:

    * ``scripts/cleanup_qdrant.py`` — one-shot deletion of legacy
      Qdrant collections (``ridge_graphs``, ``pair_features``,
      ``deepprint_poc``) and re-creation of the embedding
      collection from scratch during a controlled re-enrollment.
    * The ``qdrant_client`` HTTP API / CLI for ad-hoc inspection.
    * The standard Qdrant migration workflow (snapshot → restore)
      for schema changes.

    See ``docs/adr/011-repository-no-destructive-ops.md``.

    Memory: the underlying ``QdrantClient`` is owned by the
    repository.  Call ``close()`` (or use as a context manager) on
    application shutdown to release the HTTP/2 connection pool.
    """

    def __init__(self, client: QdrantClient,
                 collection: str | None = None) -> None:
        self._client = client
        self._collection = collection or config.qdrant_embedding_collection
        self._dim = config.embedding_dim

    @classmethod
    def from_host(cls, host: str = "localhost", port: int = 6333,
                  collection: str | None = None) -> QdrantEmbeddingRepository:
        return cls(QdrantClient(host=host, port=port), collection=collection)

    def __enter__(self) -> QdrantEmbeddingRepository:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying Qdrant client and its connection pool."""
        if hasattr(self._client, "close"):
            self._client.close()

    @staticmethod
    def _point_id(fingerprint_id: str) -> int:
        return int(hashlib.sha256(fingerprint_id.encode()).hexdigest()[:16], 16)

    def _collection_exists(self) -> bool:
        try:
            self._client.get_collection(self._collection)
            return True
        except (ValueError, UnexpectedResponse):
            return False

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist. No-op otherwise.

        Antipattern note: this method does NOT accept a ``drop_old``
        parameter.  Destructive collection operations are a separate
        concern, executed by ``scripts/cleanup_qdrant.py`` under
        human review, not by a runtime request handler.  See the
        class docstring and ``docs/adr/011-repository-no-destructive-ops.md``.

        Concurrency: when multiple uvicorn workers boot in parallel,
        each one races to create the collection.  The Qdrant server
        returns ``409 Conflict`` to all but one of them.  We catch
        that race and re-read the existing collection's metadata
        instead of failing the request.

        Raises ``ValueError`` if the collection exists with a
        different vector size than ``config.embedding_dim`` — this
        protects against silent dim mismatches that would break
        cosine search.
        """
        if self._collection_exists():
            self._check_dim()
            return
        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qdrant_models.VectorParams(
                    size=self._dim,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                self._collection, self._dim,
            )
        except (ValueError, UnexpectedResponse) as exc:
            # Another worker created the collection between our
            # existence check and the create call.  Fall through to
            # the dim-validation path.
            if not self._collection_exists():
                msg = (
                    f"create_collection failed and collection is "
                    f"still missing: {exc}"
                )
                raise RuntimeError(msg) from exc
        self._check_dim()

    def _check_dim(self) -> None:
        info = self._client.get_collection(self._collection)
        params = info.config.params
        vectors = params.vectors
        if hasattr(vectors, "size"):
            existing_size = int(vectors.size)  # type: ignore[union-attr]
        else:
            existing_size = int(vectors["size"])  # type: ignore[index]
        if existing_size != self._dim:
            msg = (
                f"Collection '{self._collection}' exists with dim={existing_size}, "
                f"but config.embedding_dim={self._dim}. Drop the collection "
                f"manually with scripts/cleanup_qdrant.py to fix the mismatch."
            )
            raise ValueError(msg)

    def upsert(self, fingerprint_id: str,
               vector: NDArray[np.float32],
               payload: QdrantPayload | None = None) -> None:
        p: QdrantPayload = QdrantPayload()
        if payload is not None:
            p["person_id"] = payload.get("person_id", "")
            p["capture_id"] = payload.get("capture_id", "")
            if "finger_name" in payload:
                p["finger_name"] = payload["finger_name"]
        p["fingerprint_id"] = fingerprint_id
        self._client.upsert(
            collection_name=self._collection,
            points=[
                qdrant_models.PointStruct(
                    id=self._point_id(fingerprint_id),
                    vector=vector.astype(np.float32).tolist(),
                    payload=dict(p),
                )
            ],
        )

    def search(self, vector: NDArray[np.float32],
               top_k: int = 10) -> list[QdrantHit]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector.astype(np.float32).tolist(),
            limit=top_k,
            with_payload=True,
        )
        hits: list[QdrantHit] = []
        for hit in result.points:
            payload_raw = hit.payload or {}
            payload: QdrantPayload = {
                "person_id": str(payload_raw.get("person_id", "")),
                "capture_id": str(payload_raw.get("capture_id", "")),
            }
            finger_name_raw = payload_raw.get("finger_name")
            if finger_name_raw is not None:
                payload["finger_name"] = str(finger_name_raw)
            hits.append(
                QdrantHit(
                    fingerprint_id=str(
                        payload_raw.get("fingerprint_id", hit.id)
                    ),
                    score=float(hit.score or 0.0),
                    payload=payload,
                )
            )
        return hits

    def delete(self, fingerprint_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qdrant_models.PointIdsList(
                points=[self._point_id(fingerprint_id)],
            ),
        )

    def count(self) -> int:
        info = self._client.get_collection(self._collection)
        return int(info.points_count or 0)

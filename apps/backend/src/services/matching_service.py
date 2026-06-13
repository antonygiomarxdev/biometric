"""
MatchingService — bridges CPU-bound biometric processing and pgvector HNSW queries.

Per D-11, D-12:
  - Delegates fingerprint processing to ``run_in_executor`` with the global
    ``ProcessPoolExecutor`` so the ASGI event loop never blocks.
  - Executes pgvector HNSW L2 distance queries (``<->``) for Top-K candidate
    search against the ``fingerprint_vectors`` table.
  - Remains domain-agnostic — it knows nothing about Cases or Evidences;
    it works with raw image bytes, ``NormalizedFingerprint``, and the vector
    index.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.core.config import config
from src.core.types import NormalizedFingerprint
from src.services.fingerprint_service import FingerprintService

logger = logging.getLogger(__name__)


@dataclass
class CandidateMatch:
    """
    A single candidate returned by a similarity search.

    Attributes:
        person_id:  External person identifier (from the known-prints table).
        name:       Display name for the candidate.
        document:   Document number of the candidate.
        evidence_id: UUID of the Evidence row (or ``None`` for direct inserts).
        l2_distance: Euclidean distance between the probe and this candidate.
        score:      Normalised similarity score (0-1, higher is more similar).
    """

    person_id: str
    name: str
    document: str
    evidence_id: str | None
    l2_distance: float
    score: float


class MatchingService:
    """
    Orchestrates CPU-bound fingerprint processing and vector similarity search.

    Usage:

        from concurrent.futures import ProcessPoolExecutor

        pool = ProcessPoolExecutor(max_workers=4)
        matching = MatchingService(pool=pool)

        # Process an image and search Top-K candidates
        candidates = await matching.search_latent(image_bytes, top_k=10)
        for c in candidates:
            print(c.person_id, c.l2_distance, c.score)

    The service **does not** manage its own executor — callers pass the
    application-scoped pool (created by the ``lifespan`` manager).
    """

    def __init__(
        self,
        pool: ProcessPoolExecutor | None = None,
        fingerprint_service: FingerprintService | None = None,
    ) -> None:
        """
        Args:
            pool: Application-scoped ``ProcessPoolExecutor`` for CPU offload.
                  When ``None``, falls back to ``asyncio.get_running_loop()``
                  default executor (``ThreadPoolExecutor``).
            fingerprint_service: Optional override for the pipeline service.
        """
        self._pool = pool
        self._fingerprint_service = fingerprint_service or FingerprintService()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_latent(
        self,
        image_bytes: bytes,
        top_k: int = config.top_k_matches,
        db: Session | None = None,
    ) -> list[CandidateMatch]:
        """
        Process a latent fingerprint image and search for Top-K candidates.

        This is the primary entry-point for the ``/matching`` endpoint.

        Steps:
          1. Decode and process the image via ``FingerprintService`` inside
             the process pool (CPU-bound).
          2. Build a fixed-dimension query vector from the result.
          3. Query ``fingerprint_vectors`` with the pgvector HNSW L2 distance
             operator (``<->``) and return the Top-K matches.

        Args:
            image_bytes: Raw image bytes (BMP, PNG, or JPEG).
            top_k:       Number of candidates to return (default: from config).
            db:          SQLAlchemy session.  When ``None`` the caller must
                         provide one via the ``with_session`` context manager
                         or the search will raise.

        Returns:
            A list of ``CandidateMatch`` ordered by ascending L2 distance
            (most similar first).
        """
        # --- CPU-bound processing (offloaded to process pool) ---------------
        fingerprint = await self._run_cpu_bound(image_bytes)

        if not fingerprint.minutiae:
            logger.warning("No minutiae extracted from latent — returning empty results")
            return []

        # --- Build query vector ---------------------------------------------
        query_vector = self._build_query_vector(fingerprint)

        # --- HNSW L2 search -------------------------------------------------
        return await self._vector_search(query_vector, top_k, db)

    async def register_known(
        self,
        image_bytes: bytes,
        person_id: str,
        name: str,
        document: str,
        db: Session,
    ) -> NormalizedFingerprint:
        """
        Process a known (ten-print) image and return the normalised result
        suitable for insertion into the ``fingerprint_vectors`` table.

        The caller is responsible for persisting the returned fingerprint
        and its vector embedding to the database.

        Args:
            image_bytes: Raw image bytes.
            person_id:   External person identifier.
            name:        Display name.
            document:    Document number.
            db:          SQLAlchemy session for any ancillary queries.

        Returns:
            The ``NormalizedFingerprint`` produced by the pipeline.
        """
        fingerprint = await self._run_cpu_bound(image_bytes)

        if not fingerprint.minutiae:
            logger.warning(
                "No minutiae extracted from known print %s (%s)",
                person_id,
                name,
            )

        return fingerprint

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_cpu_bound(
        self,
        image_bytes: bytes,
    ) -> NormalizedFingerprint:
        """
        Decode and process a fingerprint image in the process pool.

        The image decoding (``cv2.imdecode``) and the full pipeline
        (enhance → extract → normalise) run inside the executor to avoid
        blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        pool = self._pool

        # We wrap both decode + process in a single callable so every
        # CPU-heavy step stays off the main thread.
        def _work(payload: bytes) -> NormalizedFingerprint:
            import cv2
            import numpy as np

            nparr = np.frombuffer(payload, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Failed to decode image bytes — file may be corrupt")

            # The service instance is pickle-safe because FingerprintService
            # only holds lightweight strategy objects (enhancer, extractor, etc.)
            return self._fingerprint_service.process_image(
                image, fingerprint_id="latent", resize=True
            )

        return await loop.run_in_executor(pool, _work, image_bytes)

    def _build_query_vector(self, fp: NormalizedFingerprint) -> np.ndarray:
        """
        Build a fixed-dimension float32 vector from a normalised fingerprint.

        Padding / truncation to ``config.vector_dimension`` ensures
        compatibility with the pgvector index.
        """
        raw = fp.vector  # variable-length from NormalizedFingerprint.vector
        target = config.vector_dimension

        if len(raw) >= target:
            return raw[:target].astype(np.float32)

        padded = np.zeros(target, dtype=np.float32)
        padded[: len(raw)] = raw
        return padded

    async def _vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        db: Session | None,
    ) -> list[CandidateMatch]:
        """
        Execute a pgvector HNSW L2 distance query via the ``<->`` operator.

        The SQL uses the HNSW index created in migration 0001 to perform
        approximate nearest-neighbour search in O(log n) time.
        """
        if db is None:
            raise RuntimeError(
                "A SQLAlchemy Session is required for vector search. "
                "Pass ``db`` to ``search_latent`` or use ``with_session``."
            )

        vector_str = f"[{','.join(f'{v:.8f}' for v in query_vector.tolist())}]"

        sql = text(
            """
            SELECT
                fv.person_id,
                fv.name,
                fv.document,
                fv.evidence_id::text,
                fv.embedding <-> :query_vec  AS l2_distance
            FROM fingerprint_vectors fv
            ORDER BY fv.embedding <-> :query_vec
            LIMIT :top_k
            """
        )

        rows = db.execute(
            sql,
            {"query_vec": vector_str, "top_k": top_k},
        ).fetchall()

        candidates: list[CandidateMatch] = []
        # Normalise scores: scale distances to (0, 1] where closer -> higher.
        # Use the config match_threshold as the reference "far" distance.
        threshold = config.match_threshold
        for row in rows:
            l2 = float(row.l2_distance)
            score = max(0.0, 1.0 - (l2 / threshold)) if threshold > 0 else 0.0
            candidates.append(
                CandidateMatch(
                    person_id=row.person_id,
                    name=row.name,
                    document=row.document,
                    evidence_id=row.evidence_id if row.evidence_id else None,
                    l2_distance=l2,
                    score=score,
                )
            )

        logger.info(
            "Vector search returned %d candidates (top_k=%d)",
            len(candidates),
            top_k,
        )
        return candidates

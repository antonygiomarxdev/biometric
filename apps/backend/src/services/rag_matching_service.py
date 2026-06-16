"""
QdrantRagMatchingService — Phase 15+ (Qdrant Chunked Indexing).

Wires together:
  * ``FingerprintService`` with the appropriate forensic validation strategy
    (enrollment requires >=8 minutiae, search accepts >=2)
  * ``RagTripletVectorizer`` to chunk a normalized fingerprint into
    weighted Delaunay-triangle invariants
  * ``QdrantChunkRepository`` to persist and search chunks in Qdrant

Replaces the deprecated ``RagMatchingService`` (pgvector). No SQLAlchemy
dependency. Preferred path for production.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.core.types import NormalizedFingerprint
from src.db.qdrant_chunk_repository import QdrantChunkRepository
from src.domain.forensic_rules import (
    EnrollmentValidationStrategy,
    SearchValidationStrategy,
)
from src.processing.vectorizer import RagTripletVectorizer
from src.services.fingerprint_service import FingerprintService

if TYPE_CHECKING:
    from src.domain.forensic_rules import IForensicValidationStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EnrollmentResult:
    """Summary of a successful enrollment into the RAG store."""

    person_id: str
    chunks_inserted: int
    total_weight: float


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A single aggregated match candidate."""

    person_id: str
    total_score: float
    hits: int



class QdrantRagMatchingService:
    """Orchestrates enrollment and search against the Qdrant chunk store.

    Preferred path for production. Replaces the deprecated
    ``RagMatchingService`` (pgvector). No SQLAlchemy ``Session`` required.
    """

    def __init__(
        self,
        fingerprint_service: FingerprintService | None = None,
        chunk_repository: QdrantChunkRepository | None = None,
        vectorizer: RagTripletVectorizer | None = None,
        pool: Executor | None = None,
    ) -> None:
        self._fp_service = fingerprint_service
        self._chunk_repo = chunk_repository or QdrantChunkRepository.from_host()
        self._vectorizer = vectorizer or RagTripletVectorizer()
        self._pool = pool

    # ------------------------------------------------------------------
    # Internal pipeline wiring
    # ------------------------------------------------------------------

    def _ensure_service(self) -> FingerprintService:
        if self._fp_service is None:
            self._fp_service = FingerprintService()
        return self._fp_service

    def _run_pipeline(
        self,
        image: np.ndarray,
        strategy: "IForensicValidationStrategy",
        fingerprint_id: str,
    ) -> NormalizedFingerprint:
        service = self._ensure_service()
        normalized = service.process_image(image, fingerprint_id=fingerprint_id)
        if hasattr(normalized, "minutiae"):
            strategy.validate(normalized.minutiae)
        return normalized

    # ------------------------------------------------------------------
    # Public API — sync
    # ------------------------------------------------------------------

    def enroll(
        self,
        image: np.ndarray,
        person_id: str,
        fingerprint_id: str | None = None,
    ) -> EnrollmentResult:
        """Enroll a fingerprint into the Qdrant chunk store.

        Args:
            image: Grayscale fingerprint image.
            person_id: Person identifier.
            fingerprint_id: Optional fingerprint-level identifier.
                Defaults to ``person_id``.

        Returns:
            EnrollmentResult with inserted chunk count.
        """
        fid = fingerprint_id or person_id
        normalized = self._run_pipeline(
            image, EnrollmentValidationStrategy(), fid,
        )
        chunks = self._vectorizer._chunks_from_normalized(normalized)
        if not chunks:
            logger.warning("Enrollment for %s produced 0 chunks", person_id)
            return EnrollmentResult(
                person_id=person_id,
                chunks_inserted=0,
                total_weight=0.0,
            )
        inserted = self._chunk_repo.bulk_insert_chunks(
            person_id, fid, chunks,
        )
        total_weight = sum(c.weight for c in chunks)
        logger.info(
            "Enrolled %s: %d chunks, total_weight=%.3f",
            person_id, inserted, total_weight,
        )
        return EnrollmentResult(
            person_id=person_id,
            chunks_inserted=inserted,
            total_weight=total_weight,
        )

    def search(
        self,
        image: np.ndarray,
        top_k_per_chunk: int = 5,
        top_k_persons: int = 10,
    ) -> list[SearchHit]:
        """Search the Qdrant chunk store for matches to a latent image.

        Args:
            image: Grayscale fingerprint image (probe).
            top_k_per_chunk: Nearest neighbors per chunk.
            top_k_persons: Maximum persons to return.

        Returns:
            List of SearchHit sorted by total_score descending.
        """
        normalized = self._run_pipeline(
            image, SearchValidationStrategy(), "latent",
        )
        chunks = self._vectorizer._chunks_from_normalized(normalized)
        if not chunks:
            return []
        hits = self._chunk_repo.weighted_knn_search(
            chunks,
            top_k_per_chunk=top_k_per_chunk,
        )
        person_hits = self._chunk_repo.aggregate_scores_by_person(hits)
        return [
            SearchHit(
                person_id=h.person_id,
                total_score=h.total_score,
                hits=h.hits,
            )
            for h in person_hits[:top_k_persons]
        ]

    # ------------------------------------------------------------------
    # Public API — async (for FastAPI routers)
    # ------------------------------------------------------------------

    async def enroll_async(
        self,
        image_bytes: bytes,
        person_id: str,
        fingerprint_id: str | None = None,
    ) -> EnrollmentResult:
        image = await self._decode_async(image_bytes)
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, self.enroll, image, person_id, fingerprint_id,
        )

    async def search_async(
        self,
        image_bytes: bytes,
        top_k_per_chunk: int = 5,
        top_k_persons: int = 10,
    ) -> list[SearchHit]:
        image = await self._decode_async(image_bytes)
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, self.search, image, top_k_per_chunk, top_k_persons,
        )

    async def _decode_async(self, image_bytes: bytes) -> np.ndarray:
        loop = asyncio.get_running_loop()

        def _decode(payload: bytes) -> np.ndarray:
            import cv2
            nparr = np.frombuffer(payload, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to decode image bytes")
            return img

        return await loop.run_in_executor(self._pool, _decode, image_bytes)

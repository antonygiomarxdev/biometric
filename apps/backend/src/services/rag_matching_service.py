"""
RagMatchingService — Phase 10 (RAG Dactilar) end-to-end orchestrator.

Wires together:
  * ``FingerprintService`` with the appropriate forensic validation strategy
    (enrollment requires >=8 minutiae, search accepts >=2)
  * ``RagTripletVectorizer`` to chunk a normalized fingerprint into
    weighted Delaunay-triangle invariants
  * ``RagVectorRepository`` to persist and search chunks in pgvector

This service replaces the legacy single-vector 256-dim matching flow.
The old approach could not match partial latent prints because the
query vector was a global aggregation. The RAG approach works on
local invariant structures, so a 2-minutiae fragment produces a
query that may match a subset of the chunks enrolled from a full
ten-print.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.orm import Session

from src.core.types import NormalizedFingerprint
from src.db.repositories.rag_vector_repository import RagVectorRepository
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


class RagMatchingService:
    """Orchestrates enrollment and search against the RAG chunk store.

    The service is fully synchronous in its business logic; async
    wrappers around the CPU-bound decoding are exposed for the
    FastAPI router via :meth:`enroll_async` and :meth:`search_async`.
    """

    def __init__(
        self,
        fingerprint_service: FingerprintService | None = None,
        rag_repository: RagVectorRepository | None = None,
        vectorizer: RagTripletVectorizer | None = None,
        pool: Executor | None = None,
    ) -> None:
        self._fp_service = fingerprint_service
        self._rag_repo = rag_repository or RagVectorRepository()
        self._vectorizer = vectorizer or RagTripletVectorizer()
        self._pool = pool

    # ------------------------------------------------------------------
    # Internal pipeline wiring
    # ------------------------------------------------------------------

    def _ensure_service(self) -> FingerprintService:
        """Return the injected FingerprintService, or build a default one."""
        if self._fp_service is None:
            self._fp_service = FingerprintService()
        return self._fp_service

    def _run_pipeline(
        self,
        image: np.ndarray,
        strategy: "IForensicValidationStrategy",
        fingerprint_id: str,
    ) -> NormalizedFingerprint:
        """Run the full pipeline and apply the validation strategy.

        The validation strategy is applied here (not inside the
        pipeline) so that the service remains decoupled from any
        FingerprintService instance the caller might have injected.
        """
        service = self._ensure_service()
        normalized = service.process_image(image, fingerprint_id=fingerprint_id)
        # Apply forensic validation after the pipeline produces
        # candidates. The FingerprintService does this too, but doing
        # it here keeps RagMatchingService self-contained and lets
        # unit tests use a mock FingerprintService.
        if hasattr(normalized, "minutiae"):
            strategy.validate(normalized.minutiae)
        return normalized

    # ------------------------------------------------------------------
    # Public API — sync (for tests and CLI)
    # ------------------------------------------------------------------

    def enroll(
        self,
        image: np.ndarray,
        person_id: str,
        db: Session,
    ) -> EnrollmentResult:
        """Enroll a fingerprint into the RAG chunk store."""
        normalized = self._run_pipeline(
            image, EnrollmentValidationStrategy(), person_id
        )
        chunks = self._vectorizer._chunks_from_normalized(normalized)
        if not chunks:
            logger.warning("Enrollment for %s produced 0 chunks", person_id)
            return EnrollmentResult(
                person_id=person_id,
                chunks_inserted=0,
                total_weight=0.0,
            )
        self._rag_repo.bulk_insert_chunks(db, person_id, chunks)
        total_weight = sum(c.weight for c in chunks)
        logger.info(
            "Enrolled %s: %d chunks, total_weight=%.3f",
            person_id,
            len(chunks),
            total_weight,
        )
        return EnrollmentResult(
            person_id=person_id,
            chunks_inserted=len(chunks),
            total_weight=total_weight,
        )

    def search(
        self,
        image: np.ndarray,
        db: Session,
        top_k_per_chunk: int = 5,
    ) -> list[SearchHit]:
        """Search the RAG store for matches to a latent image."""
        normalized = self._run_pipeline(
            image, SearchValidationStrategy(), "latent"
        )
        chunks = self._vectorizer._chunks_from_normalized(normalized)
        if not chunks:
            return []
        all_hits: list[dict] = []
        for chunk in chunks:
            all_hits.extend(
                self._rag_repo.weighted_knn_search(
                    db,
                    query_embedding=chunk.features,
                    top_k=top_k_per_chunk,
                )
            )
        aggregated = self._rag_repo.aggregate_scores_by_person(all_hits)
        return [
            SearchHit(
                person_id=row["person_id"],
                total_score=row["total_score"],
                hits=row["hits"],
            )
            for row in aggregated
        ]

    # ------------------------------------------------------------------
    # Public API — async (for FastAPI routers)
    # ------------------------------------------------------------------

    async def enroll_async(
        self,
        image_bytes: bytes,
        person_id: str,
        db: Session,
    ) -> EnrollmentResult:
        """Async enrollment from raw image bytes (offloads CPU work)."""
        image = await self._decode_async(image_bytes)
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, self.enroll, image, person_id, db
        )

    async def search_async(
        self,
        image_bytes: bytes,
        db: Session,
        top_k_per_chunk: int = 5,
    ) -> list[SearchHit]:
        """Async search from raw image bytes (offloads CPU work)."""
        image = await self._decode_async(image_bytes)
        return await asyncio.get_running_loop().run_in_executor(
            self._pool, self.search, image, db, top_k_per_chunk
        )

    async def _decode_async(self, image_bytes: bytes) -> np.ndarray:
        """Decode image bytes in a worker thread."""
        loop = asyncio.get_running_loop()

        def _decode(payload: bytes) -> np.ndarray:
            import cv2
            nparr = np.frombuffer(payload, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("Failed to decode image bytes")
            return img

        return await loop.run_in_executor(self._pool, _decode, image_bytes)

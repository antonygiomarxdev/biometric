"""
Qdrant-backed known fingerprint enrollment router.

Phase 15+ (Qdrant Chunked Indexing): replaces the legacy pgvector
flow. The endpoint accepts a fingerprint image, runs the full RAG
pipeline via :class:`QdrantRagMatchingService.enroll_async`, and
persists weighted Delaunay-chunk vectors in the Qdrant ``fingerprint_chunks``
collection.

Per Clean Architecture (CA-03):
  - The router is an anemic HTTP controller — it extracts the file
    payload and delegates to the service layer.
  - No SQLAlchemy ``Session`` dependency needed (Qdrant is the store).
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import resources
from src.services.rag_matching_service import QdrantRagMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/known-fingerprints",
    tags=["known-fingerprints"],
)


def _get_qdrant_matching_service() -> QdrantRagMatchingService:
    """Dependency provider for :class:`QdrantRagMatchingService`."""
    return QdrantRagMatchingService(pool=resources.process_pool)


@router.post("/")
async def enroll_known(
    person_id: str = Form(..., description="External person identifier"),
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    matching: QdrantRagMatchingService = Depends(_get_qdrant_matching_service),
) -> dict[str, Any]:
    """Enroll a known fingerprint into the Qdrant chunk store.

    The forensic validation rule (``EnrollmentValidationStrategy``)
    is enforced inside the service layer: prints with fewer than
    8 minutiae are rejected before any database write.
    """
    logger.info("Qdrant enrollment — person_id=%s", person_id)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    result = await matching.enroll_async(
        image_bytes=image_bytes,
        person_id=person_id,
    )
    return {
        "success": True,
        "person_id": result.person_id,
        "chunks_inserted": result.chunks_inserted,
        "total_weight": result.total_weight,
        "message": (
            f"Enrolled {result.person_id} into Qdrant chunk store with "
            f"{result.chunks_inserted} chunks"
        ),
    }

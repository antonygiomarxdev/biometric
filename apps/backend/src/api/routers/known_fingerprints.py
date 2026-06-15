"""
RAG-based known fingerprint enrollment router.

Phase 10 (RAG Dactilar): replaces the legacy single-vector flow.
The endpoint accepts a fingerprint image, runs the full RAG pipeline
via :class:`RagMatchingService.enroll_async`, and persists weighted
Delaunay-chunk vectors in the ``rag_vector_chunks`` table.

Per Clean Architecture (CA-03):
  - The router is an anemic HTTP controller — it extracts the file
    payload and delegates to the service layer.
  - The router MUST NOT call ``db.add()`` or import any SQLAlchemy
    model directly.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, resources
from src.services.rag_matching_service import RagMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/known-fingerprints",
    tags=["known-fingerprints"],
)


def _get_rag_matching_service() -> RagMatchingService:
    """Dependency provider for :class:`RagMatchingService`."""
    return RagMatchingService(pool=resources.process_pool)


@router.post("/")
async def enroll_known(
    person_id: str = Form(..., description="External person identifier"),
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    db: Session = Depends(get_db),
    matching: RagMatchingService = Depends(_get_rag_matching_service),
) -> dict[str, Any]:
    """Enroll a known fingerprint into the RAG chunk store.

    The forensic validation rule (``EnrollmentValidationStrategy``)
    is enforced inside the service layer: prints with fewer than
    8 minutiae are rejected before any database write.
    """
    logger.info("RAG enrollment — person_id=%s", person_id)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    result = await matching.enroll_async(
        image_bytes=image_bytes,
        person_id=person_id,
        db=db,
    )
    return {
        "success": True,
        "person_id": result.person_id,
        "chunks_inserted": result.chunks_inserted,
        "total_weight": result.total_weight,
        "message": (
            f"Enrolled {result.person_id} into RAG store with "
            f"{result.chunks_inserted} chunks"
        ),
    }

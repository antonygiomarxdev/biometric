"""
Qdrant-backed known fingerprint enrollment router.

[DEPRECATED] Use POST /api/v1/persons/{id}/fingerprints/{fp_id}/captures
instead. This endpoint is kept for backward compatibility but will be
removed in the next release.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Response

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
    response: Response,
    person_id: str = Form(..., description="External person identifier"),
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    matching: QdrantRagMatchingService = Depends(_get_qdrant_matching_service),
) -> dict[str, Any]:
    """[DEPRECATED] Enroll a known fingerprint into the Qdrant chunk store.

    Use POST /api/v1/persons/{id}/fingerprints/{fp_id}/captures instead.
    """
    response.headers["Deprecation"] = "true"
    response.headers["Sunset"] = "2026-09-16"
    logger.warning("Deprecated endpoint called: POST /api/v1/known-fingerprints/ — person_id=%s", person_id)
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
            f"[DEPRECATED] Enrolled {result.person_id} into Qdrant chunk store with "
            f"{result.chunks_inserted} chunks. Use POST /api/v1/persons/... instead."
        ),
    }

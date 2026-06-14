"""
Router for known fingerprint records (``/api/v1/known-fingerprints``).

Per D-02, D-03:
  - One router per resource, mounted under ``/api/v1/known-fingerprints``.
  - Injects ``get_db`` and the lifespan-managed ``ProcessPoolExecutor``
    via ``Depends(get_db)`` and the ``MatchingService`` dependency.

Endpoints:
  - ``POST /`` — Upload and register a known (ten-print) fingerprint record.

Per CA-03:
  - The router is an anemic HTTP controller — all business logic and
    persistence is delegated to ``MatchingService.register_known()``.
  - The router MUST NOT import ``FingerprintVector`` or call ``db.add()``,
    ``db.commit()``, or ``db.refresh()``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, resources
from src.services.matching_service import MatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/known-fingerprints",
    tags=["known-fingerprints"],
)


def _get_matching_service() -> MatchingService:
    """
    Dependency provider for ``MatchingService``.

    Uses the application-scoped ``ProcessPoolExecutor`` from the lifespan
    manager so CPU-heavy processing never blocks the event loop.
    """
    return MatchingService(pool=resources.process_pool)


@router.post("/")
async def upload_known(
    person_id: str = Form(..., description="External person identifier"),
    name: str = Form(..., description="Full name of the person"),
    document: str = Form(..., description="Document number (e.g. DNI, passport)"),
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    db: Session = Depends(get_db),
    matching: MatchingService = Depends(_get_matching_service),
) -> dict[str, Any]:
    """
    Upload and register a known (ten-print) fingerprint record.

    Delegates all business logic and persistence to
    ``MatchingService.register_known()``.  The router only:
      1. Extracts file bytes from the request.
      2. Delegates to the service layer.
      3. Returns the registration summary as a JSON response.

    Returns the allocated database IDs together with extraction metadata.
    """
    logger.info("Registering known print — person_id=%s, name=%s", person_id, name)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Full pipeline: extract, vectorise, persist via the service layer
    result = await matching.register_known(
        image_bytes=image_bytes,
        person_id=person_id,
        name=name,
        document=document,
        db=db,
    )

    return {
        "success": True,
        "vector_id": str(result.vector_id),
        "person_id": result.person_id,
        "name": result.name,
        "document": result.document,
        "minutiae_count": result.minutiae_count,
        "message": f"Known fingerprint registered for {result.name}",
    }

"""
Router for known fingerprint records (``/api/v1/known-fingerprints``).

Per D-02, D-03:
  - One router per resource, mounted under ``/api/v1/known-fingerprints``.
  - Injects ``get_db`` and the lifespan-managed ``ProcessPoolExecutor``
    via ``Depends(get_db)`` and the ``MatchingService`` dependency.

Endpoints:
  - ``POST /`` — Upload and register a known (ten-print) fingerprint record.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, resources
from src.core.config import config
from src.core.types import NormalizedFingerprint
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

    The image is:
      1. Processed by ``FingerprintService`` in the process pool.
      2. Its vector embedding stored in the ``fingerprint_vectors`` table
         for future similarity searches.

    Returns the allocated database IDs together with extraction metadata.
    """
    logger.info("Registering known print — person_id=%s, name=%s", person_id, name)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Process image via the MatchingService (CPU-bound → ProcessPoolExecutor)
    fingerprint: NormalizedFingerprint = await matching.register_known(
        image_bytes=image_bytes,
        person_id=person_id,
        name=name,
        document=document,
        db=db,
    )

    if not fingerprint.minutiae:
        logger.warning(
            "No minutiae extracted for %s (%s) — registering anyway",
            person_id,
            name,
        )

    # Build the fixed-dimension vector
    vector = matching._build_query_vector(fingerprint)  # noqa: SLF001

    # Persist to fingerprint_vectors
    from src.db.models import FingerprintVector as FpVectorModel

    fv = FpVectorModel(
        person_id=person_id,
        name=name,
        document=document,
        embedding=vector.tolist(),
        num_minutiae=len(fingerprint.minutiae),
        minutiae_data=[
            {
                "x": m.x,
                "y": m.y,
                "type": m.type.name,
                "angle": m.angle,
                "confidence": m.confidence,
            }
            for m in fingerprint.minutiae
        ]
        if fingerprint.minutiae
        else None,
    )
    db.add(fv)
    db.commit()
    db.refresh(fv)

    logger.info(
        "Known print registered — vector_id=%s, person_id=%s, minutiae=%d",
        fv.id,
        person_id,
        len(fingerprint.minutiae),
    )

    return {
        "success": True,
        "vector_id": str(fv.id),
        "person_id": person_id,
        "name": name,
        "document": document,
        "minutiae_count": len(fingerprint.minutiae),
        "message": f"Known fingerprint registered for {name}",
    }

"""Fingerprints API — Phase 17 + Phase 23 (/preview)."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db, get_fingerprint_service
from src.api.prefix import API_PREFIX
from src.db.models import Person
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.schemas.fingerprint_schema import (
    FingerprintCreate,
    FingerprintListResponse,
    FingerprintPreviewResponse,
    FingerprintResponse,
    MinutiaPoint,
)
from src.services.fingerprint_service import FingerprintService

log = logging.getLogger(__name__)

router = APIRouter(tags=["fingerprints"])


@router.post(
    API_PREFIX + "/persons/{person_id}/fingerprints",
    response_model=FingerprintResponse,
    status_code=201,
    summary="Create a fingerprint slot",
    responses={
        404: {"description": "Person not found"},
        409: {"description": "Fingerprint slot already exists"},
    }
)
async def create_fingerprint(
    person_id: uuid.UUID,
    data: FingerprintCreate,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    """Create a new fingerprint slot for a person."""
    person = await session.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    existing = await FingerprintRepository.find_slot(
        session, person_id, data.finger_position, data.capture_type,
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Slot for finger {data.finger_position}/{data.capture_type} already exists",
        )
    return await FingerprintRepository.create(
        session,
        person_id=person_id,
        finger_position=data.finger_position,
        capture_type=data.capture_type,
        notes=data.notes,
    )


@router.get(
    API_PREFIX + "/persons/{person_id}/fingerprints",
    response_model=FingerprintListResponse,
    summary="List fingerprints for a person",
)
async def list_fingerprints(
    person_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    """List fingerprint slots for a person."""
    items = await FingerprintRepository.list_by_person(session, person_id)
    response_items = [FingerprintResponse.model_validate(item) for item in items]
    return FingerprintListResponse(items=response_items, total=len(response_items))


@router.post(
    API_PREFIX + "/fingerprints/preview",
    response_model=FingerprintPreviewResponse,
    summary="Preview minutiae extraction without persisting (Phase 23)",
    description=(
        "Accepts a fingerprint image (BMP, PNG, JPEG), runs the full "
        "extraction pipeline (Gabor enhancement + skeletonization + minutiae), "
        "and returns the processed image (base64 PNG) plus the detected "
        "minutiae. Does NOT persist a capture; the perito reviews the "
        "extraction in the UI and explicitly enrolls via the captures endpoint."
    ),
    responses={
        400: {"description": "Invalid image or extraction failure"},
    },
)
async def preview_fingerprint(
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    fp_service: FingerprintService = Depends(get_fingerprint_service),
) -> Any:
    """Preview minutiae for a fingerprint image without persisting.

    The perito uploads a fingerprint image from the enrollment wizard.
    The backend runs the full extraction pipeline and returns the
    enhanced image (base64 PNG), detected minutiae, and summary stats.
    The perito reviews the result before committing via the captures
    endpoint.

    This endpoint is the backend for ``getMinutiaeForImage`` in
    ``apps/frontend/src/lib/api.ts`` (Phase 23, D-29).
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    loop = asyncio.get_running_loop()
    try:
        normalized = await loop.run_in_executor(
            None,
            fp_service.process_image_from_bytes,
            image_bytes,
            "preview",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Serialize minutiae
    minutiae_points: list[dict[str, Any]] = []
    terminations = 0
    bifurcations = 0
    for m in normalized.minutiae:
        type_value = int(m.type.value)
        minutiae_points.append(
            {"x": int(m.x), "y": int(m.y), "angle": float(m.angle), "type": type_value},
        )
        if type_value == 0:
            terminations += 1
        elif type_value == 1:
            bifurcations += 1

    # Render the enhanced image to a base64 PNG
    img = getattr(normalized, "image", None)
    if img is None:
        # Fall back to the raw bytes via cv2 if the pipeline didn't carry the image forward
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to encode processed image")
    b64_png = base64.b64encode(buf.tobytes()).decode("ascii")

    h, w = img.shape[:2]
    return FingerprintPreviewResponse(
        processed_image=b64_png,
        minutiae=[MinutiaPoint(**p) for p in minutiae_points],
        terminations=terminations,
        bifurcations=bifurcations,
        image_shape=[int(h), int(w)],
        image_dtype=str(img.dtype),
    )

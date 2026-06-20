"""Fingerprints API — Phase 17 + Phase 23 (/preview)."""

from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from typing import TYPE_CHECKING, Any

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile

from src.api.dependencies import get_async_db, get_mcc_matching_service
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

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.mcc_matching_service import MccMatchingService

log = logging.getLogger(__name__)

router = APIRouter(tags=["fingerprints"])


@router.post(
    API_PREFIX + "/persons/{person_id}/fingerprints",
    response_model=FingerprintResponse,
    status_code=201,
    summary="Create or fetch a fingerprint slot (idempotent)",
    responses={
        200: {"description": "Slot already existed, returned as-is"},
        201: {"description": "Slot created"},
        404: {"description": "Person not found"},
    }
)
async def create_fingerprint(
    person_id: uuid.UUID,
    data: FingerprintCreate,
    session: AsyncSession = Depends(get_async_db),
    response: Response | None = None,
) -> Any:
    """Idempotent fingerprint slot creation.

    If a slot already exists for the (person, finger_position, capture_type)
    triple, return it with HTTP 200 instead of raising 409. This makes
    the enrollment wizard resilient to re-selection, repeated clicks,
    and re-enrollments of the same finger.
    """
    person = await session.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    existing = await FingerprintRepository.find_slot(
        session, person_id, data.finger_position, data.capture_type,
    )
    if existing is not None:
        if response is not None:
            response.status_code = 200
        return existing
    if response is not None:
        response.status_code = 201
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
    mcc_svc: MccMatchingService = Depends(get_mcc_matching_service),
) -> Any:
    """Preview minutiae for a fingerprint image without persisting.

    Runs the MCC mini-pipeline (Gabor + RidgeGraphExtractor) and
    returns the enhanced image (base64 PNG) plus detected minutiae.
    Does not touch the DB or Qdrant. Backend for ``getMinutiaeForImage``
    in ``apps/frontend/src/lib/api.ts`` (Phase 23, D-29).
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None, mcc_svc.preview, image_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    minutiae_dicts: list[dict[str, Any]] = result["minutiae"]
    skeleton = result["skeleton"]
    if skeleton is None or skeleton.size == 0:
        raise HTTPException(status_code=400, detail="Pipeline produced no skeleton")

    ok, buf = cv2.imencode(".png", skeleton)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to encode skeleton")
    b64_png = base64.b64encode(buf.tobytes()).decode("ascii")

    h, w = skeleton.shape[:2]
    return FingerprintPreviewResponse(
        processed_image=b64_png,
        minutiae=[MinutiaPoint(**p) for p in minutiae_dicts],
        terminations=0,
        bifurcations=0,
        image_shape=[int(h), int(w)],
        image_dtype=str(skeleton.dtype),
    )

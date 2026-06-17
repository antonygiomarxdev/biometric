"""Captures API — Phase 17. Multipart upload, list, get, update."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db, get_fingerprint_service, get_mcc_matching_service
from src.api.prefix import API_PREFIX
from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.ridge_graph_repository import RidgeGraphRepository
from src.schemas.capture_schema import (
    CaptureResponse,
    CaptureUpdate,
    CaptureUploadResponse,
    RidgeGraphResponse,
)
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)
from src.services.fingerprint_service import FingerprintService
from src.services.mcc_matching_service import MccMatchingService

log = logging.getLogger(__name__)

router = APIRouter(tags=["captures"])


@router.post(
    API_PREFIX + "/fingerprints/{fingerprint_id}/captures",
    response_model=CaptureUploadResponse,
    status_code=201,
    summary="Upload fingerprint capture",
    responses={
        400: {"description": "Invalid image or extraction failure"},
        404: {"description": "Fingerprint slot not found"},
    }
)
async def upload_capture(
    fingerprint_id: uuid.UUID,
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    image_dpi: int | None = Form(None),
    is_reference: bool = Form(False),
    is_exemplar: bool = Form(True),
    notes: str | None = Form(None),
    session: AsyncSession = Depends(get_async_db),
    fp_service: FingerprintService = Depends(get_fingerprint_service),
    mcc_service: MccMatchingService = Depends(get_mcc_matching_service),
) -> Any:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        svc = FingerprintEnrollmentService(
            session, fp_service, mcc_matching_service=mcc_service,
        )
        capture, graphs = await svc.create_capture(
            fingerprint_id=fingerprint_id,
            image_bytes=image_bytes,
            image_dpi=image_dpi,
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    capture_dto = CaptureResponse(
        id=capture.id,
        fingerprint_id=capture.fingerprint_id,
        capture_index=capture.capture_index,
        image_uri=capture.image_uri,
        image_dpi=capture.image_dpi,
        image_quality_score=capture.image_quality_score,
        algorithm_version=capture.algorithm_version,
        processed_at=capture.processed_at,
        num_minutiae=capture.num_minutiae,
        num_graphs=capture.num_graphs,
        is_reference=capture.is_reference,
        is_exemplar=capture.is_exemplar,
        notes=capture.notes,
        graphs=[],
    )
    return CaptureUploadResponse(
        capture=capture_dto,
        graphs_created=0,
    )


@router.get(
    API_PREFIX + "/captures/{capture_id}", 
    response_model=CaptureResponse,
    summary="Get capture details",
    responses={404: {"description": "Capture not found"}}
)
async def get_capture(
    capture_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    """Retrieve capture details by ID."""
    c = await FingerprintCaptureRepository.get_by_id(session, capture_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return CaptureResponse(
        id=c.id, fingerprint_id=c.fingerprint_id,
        capture_index=c.capture_index, image_uri=c.image_uri,
        image_dpi=c.image_dpi, image_quality_score=c.image_quality_score,
        algorithm_version=c.algorithm_version, processed_at=c.processed_at,
        num_minutiae=c.num_minutiae, num_graphs=c.num_graphs,
        is_reference=c.is_reference, is_exemplar=c.is_exemplar,
        notes=c.notes, graphs=[],
    )


@router.get(
    API_PREFIX + "/captures/{capture_id}/graphs",
    response_model=list[RidgeGraphResponse],
    summary="List graphs for capture",
)
async def get_capture_graphs(
    capture_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    """Retrieve all RidgeGraphs associated with a capture."""
    return await RidgeGraphRepository.list_by_capture(session, capture_id)


@router.patch(
    API_PREFIX + "/captures/{capture_id}", 
    response_model=CaptureResponse,
    summary="Update capture",
    responses={404: {"description": "Capture not found"}}
)
async def update_capture(
    capture_id: uuid.UUID,
    data: CaptureUpdate,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    """Update a capture's metadata."""
    updates = data.model_dump(exclude_unset=True)
    c = await FingerprintCaptureRepository.update(session, capture_id, **updates)
    if c is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return CaptureResponse(
        id=c.id, fingerprint_id=c.fingerprint_id,
        capture_index=c.capture_index, image_uri=c.image_uri,
        image_dpi=c.image_dpi, image_quality_score=c.image_quality_score,
        algorithm_version=c.algorithm_version, processed_at=c.processed_at,
        num_minutiae=c.num_minutiae, num_graphs=c.num_graphs,
        is_reference=c.is_reference, is_exemplar=c.is_exemplar,
        notes=c.notes, graphs=[],
    )

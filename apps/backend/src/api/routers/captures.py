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
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_fingerprint_service
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

log = logging.getLogger(__name__)

router = APIRouter(tags=["captures"])


@router.post(
    "/api/v1/fingerprints/{fingerprint_id}/captures",
    response_model=CaptureUploadResponse,
    status_code=201,
)
async def upload_capture(
    fingerprint_id: uuid.UUID,
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    image_dpi: int | None = Form(None),
    is_reference: bool = Form(False),
    is_exemplar: bool = Form(True),
    notes: str | None = Form(None),
    session: Session = Depends(get_db),
    fp_service: FingerprintService = Depends(get_fingerprint_service),
) -> Any:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    svc = FingerprintEnrollmentService(
        session, fp_service, qdrant_repo=None, nebula_repo=None,
    )
    try:
        capture, graphs = svc.create_capture(
            fingerprint_id=fingerprint_id,
            image_bytes=image_bytes,
            image_dpi=image_dpi,
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    capture_dto = CaptureResponse.model_validate(capture)
    capture_dto.graphs = [RidgeGraphResponse.model_validate(g) for g in graphs]
    return CaptureUploadResponse(
        capture=capture_dto,
        graphs_created=len(graphs),
    )


@router.get("/api/v1/captures/{capture_id}", response_model=CaptureResponse)
def get_capture(
    capture_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> Any:
    c = FingerprintCaptureRepository.get_by_id(session, capture_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return c


@router.get(
    "/api/v1/captures/{capture_id}/graphs",
    response_model=list[RidgeGraphResponse],
)
def get_capture_graphs(
    capture_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> Any:
    return RidgeGraphRepository.list_by_capture(session, capture_id)


@router.patch("/api/v1/captures/{capture_id}", response_model=CaptureResponse)
def update_capture(
    capture_id: uuid.UUID,
    data: CaptureUpdate,
    session: Session = Depends(get_db),
) -> Any:
    updates = data.model_dump(exclude_unset=True)
    c = FingerprintCaptureRepository.update(session, capture_id, **updates)
    if c is None:
        raise HTTPException(status_code=404, detail="Capture not found")
    return c

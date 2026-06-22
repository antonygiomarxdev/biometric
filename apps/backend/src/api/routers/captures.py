"""Captures API — Phase 17. Multipart upload, list, get, update."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)

from src.api.dependencies import (
    get_async_db,
    get_embedding_service,
    get_model_loader,
)
from src.api.prefix import API_PREFIX
from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.schemas.capture_schema import (
    CaptureResponse,
    CaptureUpdate,
    CaptureUploadResponse,
)
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.ai.loader import ModelLoader
    from src.services.embedding_service import EmbeddingService

log = logging.getLogger(__name__)

router = APIRouter(tags=["captures"])


@router.post(
    API_PREFIX + "/fingerprints/{fingerprint_id}/captures",
    response_model=CaptureUploadResponse,
    status_code=201,
    summary="Upload fingerprint capture (AFR-Net embedding)",
    responses={
        400: {"description": "Invalid image or extraction failure"},
        404: {"description": "Fingerprint slot not found"},
    }
)
async def upload_capture(
    fingerprint_id: uuid.UUID,
    file: UploadFile = File(..., description="Fingerprint image (BMP, PNG, JPEG)"),
    is_reference: bool = Form(default=False),  # noqa: FBT001
    is_exemplar: bool = Form(default=True),  # noqa: FBT001
    notes: str | None = Form(None),
    session: AsyncSession = Depends(get_async_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    loader: ModelLoader = Depends(get_model_loader),
) -> Any:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        svc = FingerprintEnrollmentService(
            session, embedding_service=embedding_service, loader=loader,
        )
        capture, embedding_id = await svc.create_capture(
            fingerprint_id=fingerprint_id,
            image_bytes=image_bytes,
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    capture_dto = CaptureResponse(
        id=capture.id,
        fingerprint_id=capture.fingerprint_id,
        capture_index=capture.capture_index,
        image_uri=capture.image_uri,
        algorithm_version=capture.algorithm_version,
        processed_at=capture.processed_at,
        is_reference=capture.is_reference,
        is_exemplar=capture.is_exemplar,
        notes=capture.notes,
    )
    return CaptureUploadResponse(
        capture=capture_dto,
        embedding_id=embedding_id,
    )


@router.get(
    API_PREFIX + "/captures/{capture_id}/image",
    summary="Get the original fingerprint image from MinIO",
    description=(
        "Returns the original captured image as image/png. "
        "Stored in MinIO at enrollment — single source of truth."
    ),
    responses={
        200: {"content": {"image/png": {}}, "description": "PNG bytes from MinIO"},
        404: {"description": "Capture not found or image missing in MinIO"},
    },
)
async def get_capture_image(
    capture_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Response:
    """Serve the original fingerprint image from MinIO."""
    from src.services.fingerprint_storage import FingerprintStorage

    c = await FingerprintCaptureRepository.get_by_id(session, capture_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Capture not found")

    png_bytes = FingerprintStorage.get_bytes(str(c.id))
    if png_bytes is None:
        raise HTTPException(
            status_code=404,
            detail=f"Image not in MinIO for capture {capture_id}; re-enroll required",
        )
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
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
        algorithm_version=c.algorithm_version, processed_at=c.processed_at,
        is_reference=c.is_reference, is_exemplar=c.is_exemplar,
        notes=c.notes,
    )


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
        algorithm_version=c.algorithm_version, processed_at=c.processed_at,
        is_reference=c.is_reference, is_exemplar=c.is_exemplar,
        notes=c.notes,
    )

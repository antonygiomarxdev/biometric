"""
CRUD router for fingerprint evidence (``/api/v1/evidencias``).

Each evidence item is a latent fingerprint image uploaded by the perito
and linked to a forensic case.  Image uploads are validated for MIME
type (T-01-05) and stored in MinIO object storage.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.errors import NotFoundError, ValidationError
from src.db.models import Case as CaseModel
from src.db.models import Evidence as EvidenceModel
from src.storage.object_storage import storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evidencias", tags=["evidencias"])

# ---------------------------------------------------------------------------
# Allowed image MIME types (T-01-05)
# ---------------------------------------------------------------------------
ALLOWED_MIME_TYPES: frozenset[str] = frozenset({
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
})

MIME_TO_EXT: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class EvidenceCreate(BaseModel):
    """Request body for registering new evidence metadata."""

    case_id: uuid.UUID
    fingerprint_id: str = Field(
        ..., min_length=1, max_length=100, description="Fingerprint identifier"
    )


class EvidenceUpdate(BaseModel):
    """Request body for updating evidence metadata."""

    fingerprint_id: str | None = Field(None, max_length=100)
    num_minutiae: int | None = Field(None, ge=0)
    minutiae_data: dict[str, Any] | None = None


class EvidenceResponse(BaseModel):
    """Response model for a single evidence item."""

    id: uuid.UUID
    case_id: uuid.UUID
    fingerprint_id: str
    image_path: str | None
    num_minutiae: int | None
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class EvidenceListResponse(BaseModel):
    """Response model for paginated evidence listing."""

    items: list[EvidenceResponse]
    total: int
    skip: int
    limit: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_image(file: UploadFile) -> None:
    """
    Validate that the uploaded file has an allowed image MIME type.

    Raises ``ValidationError`` if the MIME type is not in the allow-list
    (per T-01-05).
    """
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise ValidationError(
            message="Unsupported image format",
            detail={
                "received": file.content_type,
                "allowed": sorted(ALLOWED_MIME_TYPES),
            },
        )

    # Extension-based check as a secondary guard
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        expected_ext = MIME_TO_EXT.get(file.content_type, "")
        if expected_ext and ext != expected_ext.lstrip("."):
            logger.warning(
                "MIME/extension mismatch: content_type=%s filename=%s",
                file.content_type,
                file.filename,
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=EvidenceListResponse)
async def list_evidencias(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    case_id: uuid.UUID | None = Query(None, description="Filter by case"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    List evidence items with optional case filter and pagination.
    """
    query = select(EvidenceModel)
    if case_id is not None:
        query = query.where(EvidenceModel.case_id == case_id)
    query = query.order_by(EvidenceModel.created_at.desc()).offset(skip).limit(limit)

    items = db.scalars(query).all()
    count_query = (
        select(func.count(EvidenceModel.id))
        if case_id is None
        else select(func.count(EvidenceModel.id)).where(
            EvidenceModel.case_id == case_id
        )
    )
    total = db.scalar(count_query) or 0

    return {
        "items": [EvidenceResponse.model_validate(e) for e in items],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> EvidenceModel:
    """
    Retrieve a single evidence item by its UUID.
    """
    ev = db.get(EvidenceModel, evidence_id)
    if ev is None:
        raise NotFoundError(
            message=f"Evidence not found: {evidence_id}",
            detail={"evidence_id": str(evidence_id)},
        )
    return ev


@router.post(
    "/",
    response_model=EvidenceResponse,
    status_code=201,
)
async def create_evidence(
    case_id: uuid.UUID = Query(..., description="Case UUID to attach evidence to"),
    fingerprint_id: str = Query(
        ..., min_length=1, max_length=100, description="Fingerprint identifier"
    ),
    file: UploadFile | None = None,
    db: Session = Depends(get_db),
) -> EvidenceModel:
    """
    Register new evidence, optionally uploading a fingerprint image.

    If ``file`` is provided:
    - MIME type is validated against the allow-list (T-01-05).
    - The image is stored in MinIO object storage.
    - The resulting ``image_path`` is saved on the evidence record.

    If no file is provided the evidence entry is created with
    ``image_path = NULL`` (metadata-only registration).
    """
    # Verify the parent case exists
    case = db.get(CaseModel, case_id)
    if case is None:
        raise NotFoundError(
            message=f"Case not found: {case_id}",
            detail={"case_id": str(case_id)},
        )

    # Upload image if provided
    image_path: str | None = None
    if file is not None:
        _validate_image(file)
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise ValidationError(
                message="Uploaded file is empty",
                detail={"filename": file.filename},
            )

        object_name = (
            f"evidences/{case_id}/{fingerprint_id}"
            f"{MIME_TO_EXT.get(file.content_type or '', '')}"
        )
        image_path = storage.upload_file(
            image_bytes,
            object_name,
            content_type=file.content_type or "application/octet-stream",
        )
        if image_path is None:
            logger.warning(
                "MinIO upload returned None for %s — proceeding without storage",
                object_name,
            )

    ev = EvidenceModel(
        case_id=case_id,
        fingerprint_id=fingerprint_id,
        image_path=image_path,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)

    logger.info(
        "Evidence created: id=%s case_id=%s fingerprint_id=%s image_path=%s",
        ev.id,
        ev.case_id,
        ev.fingerprint_id,
        ev.image_path,
    )
    return ev


@router.delete("/{evidence_id}", status_code=204)
async def delete_evidence(
    evidence_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """
    Delete an evidence item.
    """
    ev = db.get(EvidenceModel, evidence_id)
    if ev is None:
        raise NotFoundError(
            message=f"Evidence not found: {evidence_id}",
            detail={"evidence_id": str(evidence_id)},
        )

    db.delete(ev)
    db.commit()
    logger.info("Evidence deleted: id=%s", evidence_id)

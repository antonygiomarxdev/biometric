"""
CRUD router for fingerprint evidence (``/api/v1/evidence``).

Clean Architecture: this module acts as a pure HTTP controller.
All business logic, MinIO storage operations, and database interactions
are delegated to :class:`~src.services.evidence_service.EvidenceService`.
"""

import logging
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db
from src.services.evidence_service import evidence_service
from src.api.prefix import API_PREFIX

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/evidence", tags=["evidence"])

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
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class EvidenceListResponse(BaseModel):
    """Response model for paginated evidence listing."""

    items: list[EvidenceResponse]
    total: int
    skip: int
    limit: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=EvidenceListResponse)
async def list_evidence(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    case_id: uuid.UUID | None = Query(None, description="Filter by case"),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, object]:
    """
    List evidence items with optional case filter and pagination.
    """
    result = await evidence_service.list_evidence(
        db, skip=skip, limit=limit, case_id=case_id
    )
    return {
        "items": [EvidenceResponse.model_validate(e) for e in result["items"]],
        "total": result["total"],
        "skip": result["skip"],
        "limit": result["limit"],
    }


@router.get("/{evidence_id}", response_model=EvidenceResponse)
async def get_evidence(
    evidence_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> object:
    """
    Retrieve a single evidence item by its UUID.
    """
    return await evidence_service.get_evidence(db, evidence_id)


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
    session: AsyncSession = Depends(get_async_db),
) -> object:
    """
    Register new evidence, optionally uploading a fingerprint image.
    """
    return await evidence_service.create_evidence(
        db,
        case_id=case_id,
        fingerprint_id=fingerprint_id,
        file=file,
    )


@router.get("/{evidence_id}/image")
async def get_evidence_image(
    evidence_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Response:
    """Serve the evidence image from MinIO object storage."""
    image_data = await evidence_service.get_evidence_image(db, evidence_id)
    return Response(content=image_data, media_type="image/png")


@router.delete("/{evidence_id}", status_code=204)
async def delete_evidence(
    evidence_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> None:
    """
    Delete an evidence item.
    """
    await evidence_service.delete_evidence(db, evidence_id)

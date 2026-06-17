"""
CRUD router for forensic cases (``/api/v1/cases``).

Clean Architecture: this module acts as a pure HTTP controller.
All business logic and database operations are delegated to
:class:`~src.services.case_service.CaseService`.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_async_db
from src.api.prefix import API_PREFIX
from src.services.case_service import case_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/cases", tags=["cases"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CaseCreate(BaseModel):
    """Request body for creating a new forensic case."""

    case_number: str = Field(..., min_length=1, max_length=50, description="Unique case number")
    title: str = Field(..., min_length=1, max_length=300, description="Case title")
    description: str | None = Field(None, description="Optional description")
    status: str = Field(
        "open",
        pattern=r"^(open|closed|archived)$",
        description="Case status",
    )


class CaseUpdate(BaseModel):
    """Request body for updating an existing case."""

    title: str | None = Field(None, max_length=300)
    description: str | None = Field(None)
    status: str | None = Field(None, pattern=r"^(open|closed|archived)$")


class CaseResponse(BaseModel):
    """Response model for a single case."""

    id: uuid.UUID
    case_number: str
    title: str
    description: str | None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CaseListResponse(BaseModel):
    """Response model for paginated case listing."""

    items: list[CaseResponse]
    total: int
    skip: int
    limit: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=CaseListResponse)
async def list_cases(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: str | None = Query(None, description="Filter by status"),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, object]:
    """
    List all forensic cases with optional status filter and pagination.
    """
    result = await case_service.list_cases(session, skip=skip, limit=limit, status=status)
    return {
        "items": [CaseResponse.model_validate(c) for c in result["items"]],
        "total": result["total"],
        "skip": result["skip"],
        "limit": result["limit"],
    }


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> object:
    """
    Retrieve a single case by its UUID.
    """
    return await case_service.get_case(session, case_id)


@router.post(
    "/",
    response_model=CaseResponse,
    status_code=201,
)
async def create_case(
    body: CaseCreate,
    session: AsyncSession = Depends(get_async_db),
) -> object:
    """
    Create a new forensic case.
    """
    return await case_service.create_case(
        session,
        case_number=body.case_number,
        title=body.title,
        description=body.description,
        status=body.status,
    )


@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: uuid.UUID,
    body: CaseUpdate,
    session: AsyncSession = Depends(get_async_db),
) -> object:
    """
    Update an existing forensic case.
    """
    return await case_service.update_case(
        session,
        case_id=case_id,
        title=body.title,
        description=body.description,
        status=body.status,
    )


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> None:
    """
    Delete a forensic case and all its associated evidence (CASCADE).
    """
    await case_service.delete_case(session, case_id)

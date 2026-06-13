"""
CRUD router for forensic cases (``/api/v1/cases``).

Manages the top-level entity a perito works with.  Each case represents
an investigation containing one or more pieces of fingerprint evidence.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.errors import NotFoundError, IntegrityError
from src.db.models import Case as CaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class CaseCreate(BaseModel):
    """Request body for creating a new forensic case."""

    case_number: str = Field(
        ..., min_length=1, max_length=50, description="Unique case number"
    )
    title: str = Field(
        ..., min_length=1, max_length=300, description="Case title"
    )
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
    status: str | None = Field(
        None, pattern=r"^(open|closed|archived)$"
    )


class CaseResponse(BaseModel):
    """Response model for a single case."""

    id: uuid.UUID
    case_number: str
    title: str
    description: str | None
    status: str
    created_at: str
    updated_at: str

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
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    List all forensic cases with optional status filter and pagination.
    """
    query = select(CaseModel)
    if status:
        query = query.where(CaseModel.status == status)
    query = query.order_by(CaseModel.created_at.desc()).offset(skip).limit(limit)

    cases = db.scalars(query).all()
    total_query = (
        select(func.count(CaseModel.id))
        if not status
        else select(func.count(CaseModel.id)).where(CaseModel.status == status)
    )
    total = db.scalar(total_query) or 0

    return {
        "items": [CaseResponse.model_validate(c) for c in cases],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> CaseModel:
    """
    Retrieve a single case by its UUID.
    """
    case = db.get(CaseModel, case_id)
    if case is None:
        raise NotFoundError(
            message=f"Case not found: {case_id}",
            detail={"case_id": str(case_id)},
        )
    return case


@router.post(
    "/",
    response_model=CaseResponse,
    status_code=201,
)
async def create_case(
    body: CaseCreate,
    db: Session = Depends(get_db),
) -> CaseModel:
    """
    Create a new forensic case.
    """
    # Check for duplicate case_number
    existing = db.scalar(
        select(CaseModel).where(CaseModel.case_number == body.case_number)
    )
    if existing is not None:
        raise IntegrityError(
            message=f"Case number '{body.case_number}' already exists",
            detail={"case_number": body.case_number},
        )

    case = CaseModel(
        case_number=body.case_number,
        title=body.title,
        description=body.description or "",
        status=body.status,
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    logger.info("Case created: id=%s case_number=%s", case.id, case.case_number)
    return case


@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: uuid.UUID,
    body: CaseUpdate,
    db: Session = Depends(get_db),
) -> CaseModel:
    """
    Update an existing forensic case.
    """
    case = db.get(CaseModel, case_id)
    if case is None:
        raise NotFoundError(
            message=f"Case not found: {case_id}",
            detail={"case_id": str(case_id)},
        )

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(case, field, value)

    db.commit()
    db.refresh(case)

    logger.info("Case updated: id=%s", case.id)
    return case


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> None:
    """
    Delete a forensic case and all its associated evidence (CASCADE).
    """
    case = db.get(CaseModel, case_id)
    if case is None:
        raise NotFoundError(
            message=f"Case not found: {case_id}",
            detail={"case_id": str(case_id)},
        )

    db.delete(case)
    db.commit()
    logger.info("Case deleted: id=%s", case_id)

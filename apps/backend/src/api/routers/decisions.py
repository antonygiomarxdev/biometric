"""
Router for examiner matching decisions (``/api/v1/decisions``).

The ``Perito`` role explicitly confirms or rejects a candidate match after
visual comparison (Identificación, Exclusión, Inconcluso).  Per D-01 the
system **never** auto-approves — every verdict is recorded and logged to
the immutable audit hash chain via ``DecisionService`` (D-09).
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.services.decision_service import decision_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/decisions", tags=["decisions"])

# ---------------------------------------------------------------------------
# Examiner verdict vocabulary (Spanish — forensic domain language)
# ---------------------------------------------------------------------------
VEREDICTOS_VALIDOS: frozenset[str] = frozenset({
    "Identificación",
    "Exclusión",
    "Inconcluso",
})

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class DecisionCreate(BaseModel):
    """Request body for submitting an examiner decision."""

    case_id: uuid.UUID
    evidence_id: uuid.UUID | None = Field(
        None, description="Optional evidence UUID"
    )
    verdict: str = Field(
        ...,
        description="Examiner verdict: Identificación, Exclusión, or Inconcluso",
    )
    comments: str | None = Field(
        None, max_length=2000, description="Optional examiner notes"
    )


class DecisionResponse(BaseModel):
    """Response model for a single decision."""

    id: uuid.UUID
    case_id: uuid.UUID
    evidence_id: uuid.UUID | None
    verdict: str
    comments: str | None
    created_at: str

    model_config = {"from_attributes": True}


class DecisionListResponse(BaseModel):
    """Response model for paginated decision listing."""

    items: list[DecisionResponse]
    total: int
    skip: int
    limit: int


# ---------------------------------------------------------------------------
# Role dependency (placeholder — Phase 2 will wire real auth)
# ---------------------------------------------------------------------------


async def _require_examiner_role() -> None:
    """
    Ensure the caller has the ``Perito`` role (T-01-04).

    **Placeholder:** In Phase 2 this dependency will extract the user
    from the JWT token and verify they hold the ``Perito`` role.  For
    now it unconditionally passes so that the router is operational for
    development and testing.

    Once auth is implemented, replace the body with::

        user = await get_current_user()
        if user.role != "Perito":
            raise HTTPException(status_code=403, detail="Perito role required")
    """
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=DecisionListResponse)
async def list_decisions(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    case_id: uuid.UUID | None = Query(None, description="Filter by case"),
    verdict: str | None = Query(None, description="Filter by verdict"),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    List examiner decisions with optional filters and pagination.
    """
    return decision_service.list_decisions(
        db,
        skip=skip,
        limit=limit,
        case_id=case_id,
        verdict=verdict,
    )


@router.get("/{decision_id}", response_model=DecisionResponse)
async def get_decision(
    decision_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> Any:
    """
    Retrieve a single decision by its UUID.
    """
    return decision_service.get_decision(db, decision_id)


@router.post(
    "/",
    response_model=DecisionResponse,
    status_code=201,
    dependencies=[Depends(_require_examiner_role)],
)
async def create_decision(
    body: DecisionCreate,
    db: Session = Depends(get_db),
) -> Any:
    """
    Submit an examiner matching decision.

    The caller **must** have the ``Perito`` role (T-01-04).  The decision
    is persisted in the ``decisions`` table **and** logged to the
    immutable ``AuditLog`` hash chain via ``DecisionService`` (D-09).

    Verdict must be one of: ``Identificación``, ``Exclusión``,
    ``Inconcluso``.
    """
    return decision_service.record_verdict(
        db,
        case_id=body.case_id,
        evidence_id=body.evidence_id,
        verdict=body.verdict,
        comments=body.comments,
    )

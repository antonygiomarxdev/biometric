"""
Router for examiner matching decisions (``/api/v1/decisions``).

The ``Perito`` role explicitly confirms or rejects a candidate match after
visual comparison (Identificación, Exclusión, Inconcluso).  Per D-01 the
system **never** auto-approves — every verdict is recorded and logged to
the immutable audit hash chain via ``AuditService`` (D-09).
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.errors import NotFoundError, ValidationError, IntegrityError
from src.db.models import Case as CaseModel
from src.db.models import Decision as DecisionModel
from src.db.models import Evidence as EvidenceModel
from src.services.audit_service import audit_service

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
    query = select(DecisionModel)
    if case_id is not None:
        query = query.where(DecisionModel.case_id == case_id)
    if verdict is not None:
        query = query.where(DecisionModel.verdict == verdict)
    query = query.order_by(DecisionModel.created_at.desc()).offset(skip).limit(limit)

    items = db.scalars(query).all()

    count_query = select(func.count(DecisionModel.id))
    if case_id is not None:
        count_query = count_query.where(DecisionModel.case_id == case_id)
    if verdict is not None:
        count_query = count_query.where(DecisionModel.verdict == verdict)
    total = db.scalar(count_query) or 0

    return {
        "items": [DecisionResponse.model_validate(d) for d in items],
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/{decision_id}", response_model=DecisionResponse)
async def get_decision(
    decision_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> DecisionModel:
    """
    Retrieve a single decision by its UUID.
    """
    decision = db.get(DecisionModel, decision_id)
    if decision is None:
        raise NotFoundError(
            message=f"Decision not found: {decision_id}",
            detail={"decision_id": str(decision_id)},
        )
    return decision


@router.post(
    "/",
    response_model=DecisionResponse,
    status_code=201,
    dependencies=[Depends(_require_examiner_role)],
)
async def create_decision(
    body: DecisionCreate,
    db: Session = Depends(get_db),
) -> DecisionModel:
    """
    Submit an examiner matching decision.

    The caller **must** have the ``Perito`` role (T-01-04).  The decision
    is persisted in the ``decisions`` table **and** logged to the
    immutable ``AuditLog`` hash chain via ``AuditService`` (D-09).

    Verdict must be one of: ``Identificación``, ``Exclusión``,
    ``Inconcluso``.
    """
    if body.verdict not in VEREDICTOS_VALIDOS:
        raise ValidationError(
            message=f"Invalid verdict: '{body.verdict}'",
            detail={
                "received": body.verdict,
                "allowed": sorted(VEREDICTOS_VALIDOS),
            },
        )

    # Verify referenced entities exist
    case = db.get(CaseModel, body.case_id)
    if case is None:
        raise NotFoundError(
            message=f"Case not found: {body.case_id}",
            detail={"case_id": str(body.case_id)},
        )

    if body.evidence_id is not None:
        ev = db.get(EvidenceModel, body.evidence_id)
        if ev is None:
            raise NotFoundError(
                message=f"Evidence not found: {body.evidence_id}",
                detail={"evidence_id": str(body.evidence_id)},
            )

    # Persist the decision
    decision = DecisionModel(
        case_id=body.case_id,
        evidence_id=body.evidence_id,
        verdict=body.verdict,
        comments=body.comments,
    )
    db.add(decision)
    db.flush()  # get decision.id before we commit

    # Log to the immutable audit hash chain (D-09)
    audit_service.log_event(
        session=db,
        table_name="decisions",
        record_id=decision.id,
        action="INSERT",
        payload={
            "case_id": str(body.case_id),
            "evidence_id": str(body.evidence_id) if body.evidence_id else None,
            "verdict": body.verdict,
            "comments": body.comments,
        },
    )

    db.commit()
    db.refresh(decision)

    logger.info(
        "Decision created: id=%s case_id=%s verdict=%s",
        decision.id,
        decision.case_id,
        decision.verdict,
    )
    return decision

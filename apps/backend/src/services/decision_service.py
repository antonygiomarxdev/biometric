"""
Decision service — encapsulates all Decision database operations.

Follows Clean Architecture: the router layer never accesses the database
directly.  All ``db.add``, ``db.commit``, ``db.flush``, and audit logging
live here, behind well-typed service methods that receive ``db: Session``
via method injection.
"""

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.errors import NotFoundError, ValidationError
from src.db.models import Case as CaseModel
from src.db.models import Decision as DecisionModel
from src.db.models import Evidence as EvidenceModel
from src.services.audit_service import audit_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Examiner verdict vocabulary (Spanish — forensic domain language)
# ---------------------------------------------------------------------------
VEREDICTOS_VALIDOS: frozenset[str] = frozenset({
    "Identificación",
    "Exclusión",
    "Inconcluso",
})


class DecisionService:
    """Service-layer operations for examiner matching decisions."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def list_decisions(
        db: Session,
        *,
        skip: int = 0,
        limit: int = 20,
        case_id: uuid.UUID | None = None,
        verdict: str | None = None,
    ) -> dict[str, object]:
        """Return a paginated list of decisions, optionally filtered.

        Args:
            db: SQLAlchemy ORM session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            case_id: Optional filter by case UUID.
            verdict: Optional filter by verdict text.

        Returns:
            A dict with ``items`` (list of ORM objects), ``total``,
            ``skip``, and ``limit``.
        """
        query = select(DecisionModel)
        if case_id is not None:
            query = query.where(DecisionModel.case_id == case_id)
        if verdict is not None:
            query = query.where(DecisionModel.verdict == verdict)
        query = query.order_by(DecisionModel.created_at.desc()).offset(skip).limit(limit)

        items = list(db.scalars(query).all())

        count_query = select(func.count(DecisionModel.id))
        if case_id is not None:
            count_query = count_query.where(DecisionModel.case_id == case_id)
        if verdict is not None:
            count_query = count_query.where(DecisionModel.verdict == verdict)
        total = db.scalar(count_query) or 0

        return {
            "items": items,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    @staticmethod
    def get_decision(
        db: Session,
        decision_id: uuid.UUID,
    ) -> DecisionModel:
        """Retrieve a single decision by UUID.

        Args:
            db: SQLAlchemy ORM session.
            decision_id: UUID of the decision to retrieve.

        Raises:
            NotFoundError: If no decision exists with *decision_id*.

        Returns:
            The ``Decision`` ORM instance.
        """
        decision = db.get(DecisionModel, decision_id)
        if decision is None:
            raise NotFoundError(
                message=f"Decision not found: {decision_id}",
                detail={"decision_id": str(decision_id)},
            )
        return decision

    @staticmethod
    def record_verdict(
        db: Session,
        *,
        case_id: uuid.UUID,
        evidence_id: uuid.UUID | None = None,
        verdict: str,
        comments: str | None = None,
    ) -> DecisionModel:
        """Record an examiner matching decision with audit trail.

        Validates the verdict, verifies that the referenced case and
        optional evidence exist, persists the decision, logs the event
        to the immutable audit hash chain (D-09), and commits the
        transaction.

        Args:
            db: SQLAlchemy ORM session.
            case_id: UUID of the parent case.
            evidence_id: Optional UUID of the evidence item.
            verdict: Examiner verdict (Identificación, Exclusión,
                or Inconcluso).
            comments: Optional examiner notes (max 2000 chars).

        Raises:
            ValidationError: If *verdict* is not in the allowed set.
            NotFoundError: If the referenced case or evidence does
                not exist.

        Returns:
            The newly created ``Decision`` ORM instance (committed
            and refreshed).
        """
        if verdict not in VEREDICTOS_VALIDOS:
            raise ValidationError(
                message=f"Invalid verdict: '{verdict}'",
                detail={
                    "received": verdict,
                    "allowed": sorted(VEREDICTOS_VALIDOS),
                },
            )

        # Verify referenced entities exist
        case = db.get(CaseModel, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        if evidence_id is not None:
            ev = db.get(EvidenceModel, evidence_id)
            if ev is None:
                raise NotFoundError(
                    message=f"Evidence not found: {evidence_id}",
                    detail={"evidence_id": str(evidence_id)},
                )

        # Persist the decision
        decision = DecisionModel(
            case_id=case_id,
            evidence_id=evidence_id,
            verdict=verdict,
            comments=comments,
        )
        db.add(decision)
        db.flush()  # get decision.id before audit logging

        # Log to the immutable audit hash chain (D-09)
        audit_service.log_event(
            session=db,
            table_name="decisions",
            record_id=decision.id,
            action="INSERT",
            payload={
                "case_id": str(case_id),
                "evidence_id": str(evidence_id) if evidence_id else None,
                "verdict": verdict,
                "comments": comments,
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


# Global instance
decision_service = DecisionService()

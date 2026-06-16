"""
Decision service — encapsulates decision business logic and audit trail.

Follows Clean Architecture: the service layer contains business logic
(verdict validation, entity existence checks, audit orchestration)
while all SQLAlchemy queries are delegated to repositories.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from src.api.errors import NotFoundError, ValidationError
from src.db.repositories.case_repository import CaseRepository
from src.db.repositories.decision_repository import DecisionRepository
from src.db.repositories.evidence_repository import EvidenceRepository
from src.services.audit_service import AuditService, audit_service as default_audit_service

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
    """Service-layer operations for examiner matching decisions.

    Receives its repository and service dependencies via constructor
    injection to keep the data-access layer swappable and testable.
    """

    def __init__(
        self,
        decision_repository: DecisionRepository | None = None,
        case_repository: CaseRepository | None = None,
        evidence_repository: EvidenceRepository | None = None,
        audit_service: AuditService | None = None,
    ) -> None:
        """Initialise the service.

        Args:
            decision_repository: Repository for ``Decision`` persistence.
                Defaults to a fresh :class:`DecisionRepository` if none
                is provided.
            case_repository: Repository for ``Case`` lookups.
                Defaults to a fresh :class:`CaseRepository` if none
                is provided.
            evidence_repository: Repository for ``Evidence`` lookups.
                Defaults to a fresh :class:`EvidenceRepository` if none
                is provided.
            audit_service: Service for immutable audit trail logging.
                Defaults to the global ``audit_service`` singleton.
        """
        self._decision_repo = decision_repository or DecisionRepository()
        self._case_repo = case_repository or CaseRepository()
        self._evidence_repo = evidence_repository or EvidenceRepository()
        self._audit_service = audit_service or default_audit_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_decisions(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 20,
        case_id: uuid.UUID | None = None,
        verdict: str | None = None,
    ) -> dict[str, object]:
        """Return a paginated list of decisions, optionally filtered.

        Args:
            db: Async SQLAlchemy session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            case_id: Optional filter by case UUID.
            verdict: Optional filter by verdict text.

        Returns:
            A dict with ``items`` (list of ORM objects), ``total``,
            ``skip``, and ``limit``.
        """
        items = await self._decision_repo.list(
            db, skip=skip, limit=limit, case_id=case_id, verdict=verdict
        )
        total = await self._decision_repo.count(
            db, case_id=case_id, verdict=verdict
        )

        return {
            "items": items,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    async def get_decision(
        self,
        db: AsyncSession,
        decision_id: uuid.UUID,
    ) -> object:
        """Retrieve a single decision by UUID.

        Args:
            db: Async SQLAlchemy session.
            decision_id: UUID of the decision to retrieve.

        Raises:
            NotFoundError: If no decision exists with *decision_id*.

        Returns:
            The ``Decision`` ORM instance.
        """
        decision = await self._decision_repo.get_by_id(db, decision_id)
        if decision is None:
            raise NotFoundError(
                message=f"Decision not found: {decision_id}",
                detail={"decision_id": str(decision_id)},
            )
        return decision

    async def record_verdict(
        self,
        db: AsyncSession,
        *,
        case_id: uuid.UUID,
        evidence_id: uuid.UUID | None = None,
        verdict: str,
        comments: str | None = None,
    ) -> object:
        """Record an examiner matching decision with audit trail.

        Validates the verdict, verifies that the referenced case and
        optional evidence exist, persists the decision, logs the event
        to the immutable audit hash chain (D-09), and commits the
        transaction.

        Args:
            db: Async SQLAlchemy session.
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
        case = await self._case_repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        if evidence_id is not None:
            ev = await self._evidence_repo.get_by_id(db, evidence_id)
            if ev is None:
                raise NotFoundError(
                    message=f"Evidence not found: {evidence_id}",
                    detail={"evidence_id": str(evidence_id)},
                )

        # Persist the decision (flush-only — get decision.id before audit)
        decision = await self._decision_repo.create(
            db,
            case_id=case_id,
            evidence_id=evidence_id,
            verdict=verdict,
            comments=comments,
        )

        # Log to the immutable audit hash chain (D-09)
        self._audit_service.log_event(
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

        await db.commit()
        await db.refresh(decision)

        logger.info(
            "Decision created: id=%s case_id=%s verdict=%s",
            decision.id,
            decision.case_id,
            decision.verdict,
        )
        return decision


# Global instance
decision_service = DecisionService()

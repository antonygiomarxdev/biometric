"""
Repository for :class:`~src.db.models.Decision` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``Decision``,
``select()``, ``func``, or ``desc()``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Decision


class DecisionRepository:
    """Persistence gateway for the ``decisions`` table.

    All methods accept an open :class:`~sqlalchemy.orm.Session` that
    the caller manages (commit / rollback).

    Usage::

        repo = DecisionRepository()
        decision = repo.get_by_id(session, decision_id)
        items = repo.list(session, skip=0, limit=20)
    """

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def list(
        session: Session,
        *,
        skip: int = 0,
        limit: int = 20,
        case_id: uuid.UUID | None = None,
        verdict: str | None = None,
    ) -> list[Decision]:
        """Return a paginated list of decisions, optionally filtered.

        Args:
            session: Active SQLAlchemy session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            case_id: Optional filter by case UUID.
            verdict: Optional filter by verdict text.

        Returns:
            A list of ``Decision`` ORM instances.
        """
        query = select(Decision)
        if case_id is not None:
            query = query.where(Decision.case_id == case_id)
        if verdict is not None:
            query = query.where(Decision.verdict == verdict)
        query = (
            query.order_by(Decision.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(session.scalars(query).all())

    @staticmethod
    def count(
        session: Session,
        *,
        case_id: uuid.UUID | None = None,
        verdict: str | None = None,
    ) -> int:
        """Return the total number of decisions, optionally filtered.

        Args:
            session: Active SQLAlchemy session.
            case_id: Optional filter by case UUID.
            verdict: Optional filter by verdict text.

        Returns:
            The count of matching decisions.
        """
        query = select(func.count(Decision.id))
        if case_id is not None:
            query = query.where(Decision.case_id == case_id)
        if verdict is not None:
            query = query.where(Decision.verdict == verdict)
        return session.scalar(query) or 0

    @staticmethod
    def get_by_id(
        session: Session, decision_id: uuid.UUID
    ) -> Decision | None:
        """Retrieve a single decision by UUID.

        Args:
            session: Active SQLAlchemy session.
            decision_id: UUID of the decision.

        Returns:
            The ``Decision`` instance, or ``None`` if not found.
        """
        return session.get(Decision, decision_id)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        session: Session,
        *,
        case_id: uuid.UUID,
        evidence_id: uuid.UUID | None = None,
        verdict: str,
        comments: str | None = None,
    ) -> Decision:
        """Create a new decision record and flush (no commit).

        The caller is expected to commit the transaction after performing
        additional operations (e.g. audit logging) to maintain atomicity.

        Args:
            session: Active SQLAlchemy session.
            case_id: UUID of the parent case.
            evidence_id: Optional UUID of the evidence item.
            verdict: Examiner verdict.
            comments: Optional examiner notes.

        Returns:
            The newly created ``Decision`` instance (present in the
            session but **not yet committed**).
        """
        decision = Decision(
            case_id=case_id,
            evidence_id=evidence_id,
            verdict=verdict,
            comments=comments,
        )
        session.add(decision)
        session.flush()
        return decision

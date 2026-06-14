"""
Repository for :class:`~src.db.models.Case` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``Case``,
``select()``, ``func``, or ``desc()``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Case


class CaseRepository:
    """Persistence gateway for the ``cases`` table.

    All methods accept an open :class:`~sqlalchemy.orm.Session` that
    the caller manages (commit / rollback).

    Usage::

        repo = CaseRepository()
        case = repo.get_by_id(session, case_id)
        cases = repo.list(session, skip=0, limit=20)
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
        status: str | None = None,
    ) -> list[Case]:
        """Return a paginated list of cases, optionally filtered by status.

        Args:
            session: Active SQLAlchemy session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            status: Optional status filter (``open``, ``closed``, ``archived``).

        Returns:
            A list of ``Case`` ORM instances.
        """
        query = select(Case)
        if status is not None:
            query = query.where(Case.status == status)
        query = query.order_by(Case.created_at.desc()).offset(skip).limit(limit)
        return list(session.scalars(query).all())

    @staticmethod
    def count(
        session: Session,
        *,
        status: str | None = None,
    ) -> int:
        """Return the total number of cases, optionally filtered by status.

        Args:
            session: Active SQLAlchemy session.
            status: Optional status filter.

        Returns:
            The count of matching cases.
        """
        query = select(func.count(Case.id))
        if status is not None:
            query = query.where(Case.status == status)
        return session.scalar(query) or 0

    @staticmethod
    def get_by_id(session: Session, case_id: uuid.UUID) -> Case | None:
        """Retrieve a single case by UUID.

        Args:
            session: Active SQLAlchemy session.
            case_id: UUID of the case.

        Returns:
            The ``Case`` instance, or ``None`` if not found.
        """
        return session.get(Case, case_id)

    @staticmethod
    def get_by_case_number(
        session: Session, case_number: str
    ) -> Case | None:
        """Retrieve a single case by its unique case number.

        Args:
            session: Active SQLAlchemy session.
            case_number: Unique case number string.

        Returns:
            The ``Case`` instance, or ``None`` if not found.
        """
        return session.scalar(
            select(Case).where(Case.case_number == case_number)
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        session: Session,
        *,
        case_number: str,
        title: str,
        description: str | None = None,
        status: str = "open",
    ) -> Case:
        """Create a new forensic case.

        Args:
            session: Active SQLAlchemy session.
            case_number: Unique case number.
            title: Case title.
            description: Optional description.
            status: Case status (default ``"open"``).

        Returns:
            The newly created ``Case`` ORM instance (committed and
            refreshed).
        """
        case = Case(
            case_number=case_number,
            title=title,
            description=description or "",
            status=status,
        )
        session.add(case)
        session.commit()
        session.refresh(case)
        return case

    @staticmethod
    def update(
        session: Session,
        case: Case,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> Case:
        """Update an existing case instance in place.

        Only the fields that are not ``None`` will be applied.

        Args:
            session: Active SQLAlchemy session.
            case: The ``Case`` instance to update.
            title: New title (or ``None`` to skip).
            description: New description (or ``None`` to skip).
            status: New status (or ``None`` to skip).

        Returns:
            The updated ``Case`` instance (committed and refreshed).
        """
        if title is not None:
            case.title = title
        if description is not None:
            case.description = description
        if status is not None:
            case.status = status
        session.commit()
        session.refresh(case)
        return case

    @staticmethod
    def delete(
        session: Session,
        case: Case | uuid.UUID,
    ) -> None:
        """Delete a case.

        Args:
            session: Active SQLAlchemy session.
            case: The ``Case`` instance or UUID to delete.
        """
        if isinstance(case, uuid.UUID):
            case_obj = session.get(Case, case)
            if case_obj is not None:
                session.delete(case_obj)
                session.commit()
        else:
            session.delete(case)
            session.commit()

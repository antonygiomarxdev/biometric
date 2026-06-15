"""
Case service — encapsulates case business logic.

Follows Clean Architecture: the service layer contains business logic
(validations, transformations, orchestration) while all SQLAlchemy
queries are delegated to :class:`~src.db.repositories.case_repository.CaseRepository`.
"""

from __future__ import annotations

import logging
import uuid
from typing import TypedDict

from sqlalchemy.orm import Session

from src.api.errors import IntegrityError, NotFoundError
from src.db.models import Case
from src.db.repositories.case_repository import CaseRepository

logger = logging.getLogger(__name__)


class PaginatedCases(TypedDict):
    """Typed return shape of :meth:`CaseService.list_cases`."""
    items: list[Case]
    total: int
    skip: int
    limit: int


class CaseService:
    """Service-layer operations for forensic cases.

    Receives its repository dependency via constructor injection
    to keep the data-access layer swappable and testable.
    """

    def __init__(self, case_repository: CaseRepository | None = None) -> None:
        """Initialise the service.

        Args:
            case_repository: Repository for ``Case`` persistence.
                Defaults to a fresh :class:`CaseRepository` if none
                is provided.
        """
        self._repo = case_repository or CaseRepository()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_cases(
        self,
        db: Session,
        *,
        skip: int = 0,
        limit: int = 20,
        status: str | None = None,
    ) -> "PaginatedCases":
        """Return a paginated list of cases, optionally filtered by status.

        Args:
            db: SQLAlchemy ORM session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            status: Optional status filter (``open``, ``closed``, ``archived``).

        Returns:
            A dict with ``items`` (list of ORM objects), ``total``,
            ``skip``, and ``limit``.
        """
        items = self._repo.list(db, skip=skip, limit=limit, status=status)
        total = self._repo.count(db, status=status)

        return {
            "items": items,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    def get_case(
        self,
        db: Session,
        case_id: uuid.UUID,
    ) -> object:
        """Retrieve a single case by UUID.

        Args:
            db: SQLAlchemy ORM session.
            case_id: UUID of the case.

        Raises:
            NotFoundError: If no case exists with *case_id*.

        Returns:
            The ``Case`` ORM instance.
        """
        case = self._repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )
        return case

    def create_case(
        self,
        db: Session,
        *,
        case_number: str,
        title: str,
        description: str | None = None,
        status: str = "open",
    ) -> object:
        """Create a new forensic case.

        Args:
            db: SQLAlchemy ORM session.
            case_number: Unique case number.
            title: Case title.
            description: Optional description.
            status: Case status (default ``"open"``).

        Raises:
            IntegrityError: If *case_number* already exists.

        Returns:
            The newly created ``Case`` ORM instance (already committed
            and refreshed).
        """
        # Check for duplicate case_number
        existing = self._repo.get_by_case_number(db, case_number)
        if existing is not None:
            raise IntegrityError(
                message=f"Case number '{case_number}' already exists",
                detail={"case_number": case_number},
            )

        case = self._repo.create(
            db,
            case_number=case_number,
            title=title,
            description=description,
            status=status,
        )

        logger.info("Case created: id=%s case_number=%s", case.id, case.case_number)
        return case

    def update_case(
        self,
        db: Session,
        case_id: uuid.UUID,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> object:
        """Update an existing case.

        Only the fields that are not ``None`` will be applied.

        Args:
            db: SQLAlchemy ORM session.
            case_id: UUID of the case to update.
            title: New title (or ``None`` to skip).
            description: New description (or ``None`` to skip).
            status: New status (or ``None`` to skip).

        Raises:
            NotFoundError: If no case exists with *case_id*.

        Returns:
            The updated ``Case`` ORM instance.
        """
        case = self._repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        case = self._repo.update(
            db,
            case,
            title=title,
            description=description,
            status=status,
        )

        logger.info("Case updated: id=%s", case.id)
        return case

    def delete_case(
        self,
        db: Session,
        case_id: uuid.UUID,
    ) -> None:
        """Delete a case and all its associated evidence (CASCADE).

        Args:
            db: SQLAlchemy ORM session.
            case_id: UUID of the case to delete.

        Raises:
            NotFoundError: If no case exists with *case_id*.
        """
        case = self._repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        self._repo.delete(db, case)
        logger.info("Case deleted: id=%s", case_id)


# Global instance
case_service = CaseService()

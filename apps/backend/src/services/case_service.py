"""
Case service — encapsulates case business logic.

Follows Clean Architecture: the service layer contains business logic
(validations, transformations, orchestration) while all SQLAlchemy
queries are delegated to :class:`~src.db.repositories.case_repository.CaseRepository`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

from src.api.errors import IntegrityError, NotFoundError
from src.db.repositories.case_repository import CaseRepository

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models import Case

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

    async def list_cases(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 20,
        status: str | None = None,
    ) -> PaginatedCases:
        """Return a paginated list of cases, optionally filtered by status."""
        items = await self._repo.list(db, skip=skip, limit=limit, status=status)
        total = await self._repo.count(db, status=status)

        return {
            "items": items,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    async def get_case(
        self,
        db: AsyncSession,
        case_id: uuid.UUID,
    ) -> object:
        """Retrieve a single case by UUID."""
        case = await self._repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )
        return case

    async def create_case(
        self,
        db: AsyncSession,
        *,
        case_number: str,
        title: str,
        description: str | None = None,
        status: str = "open",
    ) -> object:
        """Create a new forensic case."""
        existing = await self._repo.get_by_case_number(db, case_number)
        if existing is not None:
            raise IntegrityError(
                message=f"Case number '{case_number}' already exists",
                detail={"case_number": case_number},
            )

        case = await self._repo.create(
            db,
            case_number=case_number,
            title=title,
            description=description,
            status=status,
        )

        logger.info("Case created: id=%s case_number=%s", case.id, case.case_number)
        return case

    async def update_case(
        self,
        db: AsyncSession,
        case_id: uuid.UUID,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> object:
        """Update an existing case."""
        case = await self._repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        case = await self._repo.update(
            db,
            case,
            title=title,
            description=description,
            status=status,
        )

        logger.info("Case updated: id=%s", case.id)
        return case

    async def delete_case(
        self,
        db: AsyncSession,
        case_id: uuid.UUID,
    ) -> None:
        """Delete a case and all its associated evidence (CASCADE)."""
        case = await self._repo.get_by_id(db, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        await self._repo.delete(db, case)
        logger.info("Case deleted: id=%s", case_id)


# Global instance
case_service = CaseService()

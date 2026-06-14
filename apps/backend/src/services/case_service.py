"""
Case service — encapsulates all Case database operations.

Follows Clean Architecture: the router layer never accesses the database
directly.  All ``db.add``, ``db.commit``, and ``db.refresh`` calls live
here, behind well-typed service methods that receive ``db: Session`` via
method injection.
"""

import logging
import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.api.errors import IntegrityError, NotFoundError
from src.db.models import Case as CaseModel

logger = logging.getLogger(__name__)


class CaseService:
    """Service-layer operations for forensic cases."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def list_cases(
        db: Session,
        *,
        skip: int = 0,
        limit: int = 20,
        status: str | None = None,
    ) -> dict[str, object]:
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
        query = select(CaseModel)
        if status:
            query = query.where(CaseModel.status == status)
        query = query.order_by(CaseModel.created_at.desc()).offset(skip).limit(limit)

        cases = list(db.scalars(query).all())
        total_query = (
            select(func.count(CaseModel.id))
            if not status
            else select(func.count(CaseModel.id)).where(CaseModel.status == status)
        )
        total = db.scalar(total_query) or 0

        return {
            "items": cases,
            "total": total,
            "skip": skip,
            "limit": limit,
        }

    @staticmethod
    def get_case(
        db: Session,
        case_id: uuid.UUID,
    ) -> CaseModel:
        """Retrieve a single case by UUID.

        Raises:
            NotFoundError: If no case exists with *case_id*.
        """
        case = db.get(CaseModel, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )
        return case

    @staticmethod
    def create_case(
        db: Session,
        *,
        case_number: str,
        title: str,
        description: str | None = None,
        status: str = "open",
    ) -> CaseModel:
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
        existing = db.scalar(
            select(CaseModel).where(CaseModel.case_number == case_number)
        )
        if existing is not None:
            raise IntegrityError(
                message=f"Case number '{case_number}' already exists",
                detail={"case_number": case_number},
            )

        case = CaseModel(
            case_number=case_number,
            title=title,
            description=description or "",
            status=status,
        )
        db.add(case)
        db.commit()
        db.refresh(case)

        logger.info("Case created: id=%s case_number=%s", case.id, case.case_number)
        return case

    @staticmethod
    def update_case(
        db: Session,
        case_id: uuid.UUID,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> CaseModel:
        """Update an existing case.

        Only the fields that are not ``None`` will be applied.

        Raises:
            NotFoundError: If no case exists with *case_id*.
        """
        case = db.get(CaseModel, case_id)
        if case is None:
            raise NotFoundError(
                message=f"Case not found: {case_id}",
                detail={"case_id": str(case_id)},
            )

        if title is not None:
            case.title = title
        if description is not None:
            case.description = description
        if status is not None:
            case.status = status

        db.commit()
        db.refresh(case)

        logger.info("Case updated: id=%s", case.id)
        return case

    @staticmethod
    def delete_case(
        db: Session,
        case_id: uuid.UUID,
    ) -> None:
        """Delete a case and all its associated evidence (CASCADE).

        Raises:
            NotFoundError: If no case exists with *case_id*.
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


# Global instance
case_service = CaseService()

"""
Repository for :class:`~src.db.models.Evidence` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``Evidence``,
``select()``, ``func``, or ``desc()``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import Evidence


class EvidenceRepository:
    """Persistence gateway for the ``evidences`` table.

    All methods accept an open :class:`~sqlalchemy.orm.Session` that
    the caller manages (commit / rollback).

    Usage::

        repo = EvidenceRepository()
        ev = repo.get_by_id(session, evidence_id)
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
    ) -> list[Evidence]:
        """Return a paginated list of evidence, optionally filtered by case.

        Args:
            session: Active SQLAlchemy session.
            skip: Number of records to skip (offset).
            limit: Maximum number of records to return.
            case_id: Optional filter by parent case UUID.

        Returns:
            A list of ``Evidence`` ORM instances.
        """
        query = select(Evidence)
        if case_id is not None:
            query = query.where(Evidence.case_id == case_id)
        query = (
            query.order_by(Evidence.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(session.scalars(query).all())

    @staticmethod
    def count(
        session: Session,
        *,
        case_id: uuid.UUID | None = None,
    ) -> int:
        """Return the total number of evidence items, optionally filtered.

        Args:
            session: Active SQLAlchemy session.
            case_id: Optional filter by parent case UUID.

        Returns:
            The count of matching evidence items.
        """
        query = select(func.count(Evidence.id))
        if case_id is not None:
            query = query.where(Evidence.case_id == case_id)
        return session.scalar(query) or 0

    @staticmethod
    def get_by_id(session: Session, evidence_id: uuid.UUID) -> Evidence | None:
        """Retrieve a single evidence item by UUID.

        Args:
            session: Active SQLAlchemy session.
            evidence_id: UUID of the evidence.

        Returns:
            The ``Evidence`` instance, or ``None`` if not found.
        """
        return session.get(Evidence, evidence_id)

    @staticmethod
    def get_by_case_id(
        session: Session, case_id: uuid.UUID
    ) -> list[Evidence]:
        """Retrieve all evidence items for a given case.

        Args:
            session: Active SQLAlchemy session.
            case_id: UUID of the parent case.

        Returns:
            A list of ``Evidence`` ORM instances.
        """
        stmt = select(Evidence).where(Evidence.case_id == case_id)
        return list(session.scalars(stmt).all())

    @staticmethod
    def get_image_path(
        session: Session, evidence_id: uuid.UUID
    ) -> str | None:
        """Retrieve the ``image_path`` for a given evidence item.

        Args:
            session: Active SQLAlchemy session.
            evidence_id: UUID of the evidence.

        Returns:
            The ``image_path`` string, or ``None`` if the evidence does
            not exist or has no image.
        """
        ev = session.get(Evidence, evidence_id)
        if ev is None:
            return None
        return ev.image_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        session: Session,
        *,
        case_id: uuid.UUID,
        fingerprint_id: str,
        image_path: str | None = None,
    ) -> Evidence:
        """Create a new evidence item.

        Args:
            session: Active SQLAlchemy session.
            case_id: UUID of the parent case.
            fingerprint_id: Fingerprint identifier.
            image_path: Optional MinIO object path for the fingerprint image.

        Returns:
            The newly created ``Evidence`` ORM instance (committed and
            refreshed).
        """
        ev = Evidence(
            case_id=case_id,
            fingerprint_id=fingerprint_id,
            image_path=image_path,
        )
        session.add(ev)
        session.commit()
        session.refresh(ev)
        return ev

    @staticmethod
    def delete(
        session: Session,
        evidence: Evidence | uuid.UUID,
    ) -> None:
        """Delete an evidence item.

        Args:
            session: Active SQLAlchemy session.
            evidence: The ``Evidence`` instance or UUID to delete.
        """
        if isinstance(evidence, uuid.UUID):
            ev_obj = session.get(Evidence, evidence)
            if ev_obj is not None:
                session.delete(ev_obj)
                session.commit()
        else:
            session.delete(evidence)
            session.commit()

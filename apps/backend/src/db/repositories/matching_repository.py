"""
Repository for :class:`~src.db.models.FingerprintVector` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``FingerprintVector``,
``select()``, or ``desc()``.

Two operations:

* ``insert_fingerprint_vector`` — create, commit, and refresh a new row.
* ``get_latest_vector`` — return the most recent ``FingerprintVector``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.db.models import FingerprintVector


class MatchingRepository:
    """Persistence gateway for the ``fingerprint_vectors`` table.

    All methods accept an open :class:`~sqlalchemy.orm.Session` that
    the caller manages (commit / rollback).

    Usage::

        repo = MatchingRepository()
        fv = repo.insert_fingerprint_vector(session, {...})
        latest = repo.get_latest_vector(session)
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def insert_fingerprint_vector(
        session: Session,
        data: dict[str, Any],
    ) -> FingerprintVector:
        """Create a new ``FingerprintVector`` and persist it.

        Args:
            session: Active SQLAlchemy session.
            data: Dictionary with keys matching :class:`FingerprintVector`
                columns (``person_id``, ``name``, ``document``,
                ``embedding``, ``num_minutiae``, ``minutiae_data``).

        Returns:
            The newly created ``FingerprintVector`` instance (committed
            and refreshed).
        """
        fv = FingerprintVector(
            person_id=data["person_id"],
            name=data["name"],
            document=data["document"],
            embedding=data["embedding"],
            num_minutiae=data["num_minutiae"],
            minutiae_data=data.get("minutiae_data"),
        )
        session.add(fv)
        session.commit()
        session.refresh(fv)
        return fv

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def get_latest_vector(session: Session) -> FingerprintVector | None:
        """Return the most recent ``FingerprintVector``.

        Args:
            session: Active SQLAlchemy session.

        Returns:
            The latest ``FingerprintVector`` ordered by ``created_at``
            descending, or ``None`` when the table is empty.
        """
        stmt = (
            select(FingerprintVector)
            .order_by(desc(FingerprintVector.created_at))
            .limit(1)
        )
        return session.scalar(stmt)

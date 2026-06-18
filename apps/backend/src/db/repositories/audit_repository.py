"""
Repository for :class:`~src.db.models.AuditLog` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``AuditLog``,
``select()``, ``desc()``, or ``text()``.

Three operations:

* ``lock_table`` — table-level lock serialising concurrent appends.
* ``get_latest_entry`` — ``SELECT … FOR UPDATE`` ordered by ``created_at``.
* ``insert_entry`` — create and ``flush`` a new ``AuditLog`` row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import desc, select, text

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from src.db.models import AuditLog


class AuditRepository:
    """Persistence gateway for the ``audit_log`` table.

    All methods accept an open :class:`~sqlalchemy.orm.Session` that
    the caller manages (commit / rollback).

    Usage::

        repo = AuditRepository()
        repo.lock_table(session)
        latest = repo.get_latest_entry(session)
        entry = repo.insert_entry(session, {...})
    """

    # ------------------------------------------------------------------
    # Table lock
    # ------------------------------------------------------------------

    @staticmethod
    def lock_table(session: Session) -> None:
        """Acquire a table-level lock on ``audit_log``.

        ``SHARE ROW EXCLUSIVE MODE`` prevents concurrent writes and
        also prevents other transactions from acquiring ``SHARE`` or
        higher locks, serialising the hash-chain append sequence
        without blocking pure reads.
        """
        session.execute(
            text("LOCK TABLE audit_log IN SHARE ROW EXCLUSIVE MODE")
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    @staticmethod
    def get_latest_entry(session: Session) -> AuditLog | None:
        """Return the most recent ``AuditLog`` entry (with ``FOR UPDATE``).

        The row-level lock serialises concurrent appends inside a
        transaction so that only one caller reads the current tip of
        the hash chain at a time.

        Returns:
            The latest entry, or ``None`` when the table is empty.
        """
        stmt = (
            select(AuditLog)
            .order_by(desc(AuditLog.created_at))
            .limit(1)
            .with_for_update()
        )
        return session.execute(stmt).scalar_one_or_none()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    @staticmethod
    def insert_entry(session: Session, entry_data: dict[str, Any]) -> AuditLog:
        """Create a new ``AuditLog`` entry and flush it.

        Args:
            session: Active SQLAlchemy session.
            entry_data: Dictionary with keys matching :class:`AuditLog`
                columns (``table_name``, ``record_id``, ``action``,
                ``payload``, ``previous_hash``, ``current_hash``,
                ``created_at``).

        Returns:
            The newly created ``AuditLog`` instance (present in
            ``session`` but **not yet committed**).
        """
        entry = AuditLog(
            table_name=entry_data["table_name"],
            record_id=entry_data["record_id"],
            action=entry_data["action"],
            payload=entry_data["payload"],
            previous_hash=entry_data.get("previous_hash"),
            current_hash=entry_data["current_hash"],
            created_at=entry_data.get("created_at", datetime.now(UTC)),
        )
        session.add(entry)
        session.flush()
        return entry

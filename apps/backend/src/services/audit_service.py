"""
Audit trail service with SHA-256 hash chain integrity (per D-09).

Each call to ``log_event`` retrieves the most recent ``AuditLog`` entry
with a ``SELECT … FOR UPDATE`` lock, chains a new entry using
``SHA-256(previous_hash || canonical_payload)``, and inserts it.

The ``FOR UPDATE`` lock serialises concurrent appends inside a
transaction, preventing race conditions where two requests could read
the same ``previous_hash`` and produce parallel chain branches.

Hash format
-----------
``current_hash = SHA-256(previous_hash || json_payload)``

where ``json_payload`` is the deterministic (``sort_keys=True``) JSON
encoding of the full event context passed to ``log_event``.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from src.db.models import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    """
    Immutable audit trail — records data-mutation events in a
    cryptographic hash chain that enables tamper detection.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hash(
        previous_hash: str | None,
        chain_payload: dict[str, Any],
    ) -> str:
        """Return ``SHA-256(previous_hash || canonical_json)``.

        Args:
            previous_hash: Hex-encoded SHA-256 from the prior entry,
                or ``None`` when this is the first entry in the chain.
            chain_payload: The full event context dict to hash
                (sorted keys for determinism).

        Returns:
            Hex-encoded SHA-256 digest (64 chars).
        """
        hasher = hashlib.sha256()
        if previous_hash:
            hasher.update(previous_hash.encode("utf-8"))
        hasher.update(
            json.dumps(chain_payload, sort_keys=True, default=str).encode("utf-8")
        )
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def log_event(
        session: Session,
        action: str,
        payload: dict[str, Any],
        *,
        user_id: str | UUID | None = None,
        table_name: str = "",
        record_id: UUID | str | None = None,
    ) -> AuditLog:
        """Record an auditable event with hash-chain integrity.

        **Must** be called inside an active database transaction so that
        the ``SELECT … FOR UPDATE`` lock serialises concurrent appends.

        Args:
            session: SQLAlchemy ORM session (caller manages commit/rollback).
            action: Event type — e.g. ``'INSERT'``, ``'UPDATE'``,
                ``'DELETE'``, ``'LOGIN'``, ``'DECISION'``.
            payload: Arbitrary JSON-serialisable data describing the
                event.  The ``user_id`` is automatically injected here
                when provided.
            user_id: Optional identifier of the user who performed
                the action.  Injected into *payload* so it participates
                in the hash chain.
            table_name: Database table affected (empty for non-DML events).
            record_id: Primary key of the affected database row.

        Returns:
            The newly created ``AuditLog`` instance (present in
            ``session`` but **not yet committed**).
        """
        # ---- build the full chain material ----
        chain_payload: dict[str, Any] = {
            "table_name": table_name,
            "action": action,
            "payload": payload,
        }
        if user_id is not None:
            chain_payload["user_id"] = str(user_id)
        if record_id is not None:
            chain_payload["record_id"] = str(record_id)

        # ---- lock & read the latest entry ----
        stmt = (
            select(AuditLog)
            .order_by(desc(AuditLog.created_at))
            .limit(1)
            .with_for_update()
        )
        latest = session.execute(stmt).scalar_one_or_none()

        previous_hash: str | None = latest.current_hash if latest else None

        current_hash = AuditService._compute_hash(previous_hash, chain_payload)

        # ---- insert ----
        entry = AuditLog(
            table_name=table_name or "system",
            record_id=record_id if record_id else UUID(int=0),
            action=action,
            payload=chain_payload,
            previous_hash=previous_hash,
            current_hash=current_hash,
            created_at=datetime.now(timezone.utc),
        )
        session.add(entry)
        session.flush()

        logger.info(
            "Audit[%s]: table=%s action=%s hash=%s…",
            entry.id,
            table_name,
            action,
            current_hash[:12],
        )
        return entry


# Singleton — imported by routers and other services.
audit_service = AuditService()

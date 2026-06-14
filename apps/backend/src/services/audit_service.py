"""
Audit trail service with SHA-256 hash chain integrity (per D-09).

Each call to ``log_event`` delegates persistence to an
:class:`~src.db.repositories.audit_repository.AuditRepository`,
keeping the service layer free of ORM imports and SQLAlchemy
query builders (Clean Architecture).

Hash format
-----------
``current_hash = SHA-256(previous_hash || json_payload)``

where ``json_payload`` is the deterministic (``sort_keys=True``) JSON
encoding of the full event context passed to ``log_event``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.repositories.audit_repository import AuditRepository

logger = logging.getLogger(__name__)


class AuditService:
    """
    Immutable audit trail — records data-mutation events in a
    cryptographic hash chain that enables tamper detection.

    All SQLAlchemy operations are delegated to an injected
    :class:`~AuditRepository`.

    Usage::

        repo = AuditRepository()
        service = AuditService(repository=repo)
        entry = service.log_event(session, action="INSERT", payload={...})
    """

    def __init__(self, repository: AuditRepository) -> None:
        """Initialise the service with a repository instance.

        Args:
            repository: The persistence gateway for ``audit_log``.
        """
        self._repository = repository

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

    def log_event(
        self,
        session: Session,
        action: str,
        payload: dict[str, Any],
        *,
        user_id: str | UUID | None = None,
        table_name: str = "",
        record_id: UUID | str | None = None,
    ) -> Any:
        """Record an auditable event with hash-chain integrity.

        **Must** be called inside an active database transaction so that
        the table lock serialises concurrent appends.

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

        # ---- lock & read the latest entry via repository ----
        self._repository.lock_table(session)
        latest = self._repository.get_latest_entry(session)

        previous_hash: str | None = latest.current_hash if latest else None

        current_hash = AuditService._compute_hash(previous_hash, chain_payload)

        # ---- insert via repository ----
        entry = self._repository.insert_entry(session, {
            "table_name": table_name or "system",
            "record_id": record_id if record_id else UUID(int=0),
            "action": action,
            "payload": chain_payload,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
            "created_at": datetime.now(timezone.utc),
        })

        logger.info(
            "Audit[%s]: table=%s action=%s hash=%s…",
            entry.id,
            table_name,
            action,
            current_hash[:12],
        )
        return entry


# Singleton — imported by routers and other services.
# Created with a default AuditRepository for backward compatibility.
audit_service = AuditService(repository=AuditRepository())

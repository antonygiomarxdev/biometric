"""Async repository for :class:`~src.db.models.Fingerprint`."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Fingerprint


def _slot_lock_key(person_id: "uuid.UUID", finger_position: int,
                   capture_type: str) -> int:
    """Map a fingerprint slot to a 32-bit advisory lock key.

    Different slots can share the same key (collisions are
    harmless); what we need is stability for the same slot.
    """
    raw = hashlib.sha256(
        f"{person_id}:{finger_position}:{capture_type}".encode()
    ).digest()[:8]
    return int.from_bytes(raw, "big", signed=True)


class FingerprintRepository:
    """Async persistence gateway for the ``fingerprints`` table.

    Concurrency: ``create`` is idempotent on
    ``(person_id, finger_position, capture_type)`` and safe under
    concurrent requests.  A per-slot advisory lock serialises the
    existence check + insert; a UNIQUE constraint and
    ``ON CONFLICT DO NOTHING`` fallback guarantee correctness.
    """

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        person_id: uuid.UUID,
        finger_position: int,
        capture_type: str = "rolled",
        notes: str | None = None,
    ) -> tuple[Fingerprint, bool]:
        """Create a fingerprint slot, or return the existing one.

        Returns ``(fingerprint, created)``.

        Concurrency strategy is the same as
        :meth:`PersonRepository.create`: per-slot advisory lock on
        PostgreSQL, single-writer serialisation on SQLite, both
        backed by the UNIQUE constraint + ``ON CONFLICT DO NOTHING``
        for correctness.
        """
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            lock_key = _slot_lock_key(person_id, finger_position, capture_type)
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": lock_key},
            )

        existing = await FingerprintRepository.find_slot(
            session, person_id, finger_position, capture_type,
        )
        if existing is not None:
            return existing, False

        stmt = (
            pg_insert(Fingerprint)
            .values(
                person_id=person_id,
                finger_position=finger_position,
                capture_type=capture_type,
                notes=notes,
            )
            .on_conflict_do_nothing(
                index_elements=["person_id", "finger_position", "capture_type"],
            )
            .returning(Fingerprint)
        )
        result = await session.execute(stmt)
        f = result.scalar_one_or_none()
        await session.commit()
        if f is not None:
            return f, True
        existing = await FingerprintRepository.find_slot(
            session, person_id, finger_position, capture_type,
        )
        if existing is None:
            msg = "ON CONFLICT fired but slot is not queryable"
            raise RuntimeError(msg)
        return existing, False

    @staticmethod
    async def get_by_id(session: AsyncSession, fingerprint_id: uuid.UUID) -> Fingerprint | None:
        return await session.get(Fingerprint, fingerprint_id)

    @staticmethod
    async def list_by_person(
        session: AsyncSession,
        person_id: uuid.UUID,
    ) -> list[Fingerprint]:
        stmt = (
            select(Fingerprint)
            .where(Fingerprint.person_id == person_id)
            .order_by(Fingerprint.finger_position, Fingerprint.capture_type)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def find_slot(
        session: AsyncSession,
        person_id: uuid.UUID,
        finger_position: int,
        capture_type: str,
    ) -> Fingerprint | None:
        stmt = select(Fingerprint).where(
            Fingerprint.person_id == person_id,
            Fingerprint.finger_position == finger_position,
            Fingerprint.capture_type == capture_type,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def increment_capture_count(
        session: AsyncSession,
        fingerprint_id: uuid.UUID,
    ) -> Fingerprint | None:
        f = await session.get(Fingerprint, fingerprint_id)
        if f is None:
            return None
        f.capture_count = (f.capture_count or 0) + 1
        f.last_captured_at = datetime.now(UTC)
        if f.first_captured_at is None:
            f.first_captured_at = f.last_captured_at
        await session.commit()
        await session.refresh(f)
        return f

    @staticmethod
    async def delete(session: AsyncSession, fingerprint_id: uuid.UUID) -> bool:
        f = await session.get(Fingerprint, fingerprint_id)
        if f is None:
            return False
        await session.delete(f)
        await session.commit()
        return True

"""Async repository for :class:`~src.db.models.Fingerprint`."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Fingerprint


class FingerprintRepository:
    """Async persistence gateway for the ``fingerprints`` table."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        person_id: uuid.UUID,
        finger_position: int,
        capture_type: str = "rolled",
        notes: str | None = None,
    ) -> Fingerprint:
        f = Fingerprint(
            person_id=person_id,
            finger_position=finger_position,
            capture_type=capture_type,
            notes=notes,
        )
        session.add(f)
        await session.commit()
        await session.refresh(f)
        return f

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

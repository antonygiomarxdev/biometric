"""Async repository for :class:`~src.db.models.Person`."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    from datetime import datetime

    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Person


def _external_id_lock_key(external_id: str) -> int:
    """Map a person external_id to a 32-bit advisory lock key.

    Collisions across different external_ids are harmless — they
    just add a little contention.  The key needs to fit PostgreSQL's
    ``bigint`` advisory lock range.
    """
    raw = hashlib.sha256(external_id.encode()).digest()[:8]
    return int.from_bytes(raw, "big", signed=True)


class PersonRepository:
    """Async persistence gateway for the ``persons`` table.

    Concurrency: ``create`` is idempotent on ``external_id`` and
    safe under concurrent requests.  A per-``external_id`` advisory
    lock serialises the existence check + insert; a UNIQUE
    constraint on ``external_id`` and an ``ON CONFLICT DO NOTHING``
    fallback guarantee correctness even if the lock is somehow
    bypassed.
    """

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        external_id: str | None = None,
        full_name: str | None = None,
        doc_type: str | None = None,
        doc_number: str | None = None,
        sex: str | None = None,
        dob: datetime | None = None,
        notes: str | None = None,
    ) -> tuple[Person, bool]:
        """Create a person, or return the existing one.

        Returns ``(person, created)``.  When ``created`` is
        ``False`` the existing row was returned (idempotent replay).

        Concurrency strategy:
        * PostgreSQL: per-``external_id`` advisory lock serialises
          the existence check + insert; the UNIQUE constraint +
          ``ON CONFLICT DO NOTHING`` catches any race the lock
          misses (defence in depth).
        * SQLite (tests): a single-writer process serialises writes
          naturally.  We still rely on the UNIQUE constraint and
          ``ON CONFLICT DO NOTHING`` for correctness.
        """
        if external_id is not None and session.bind is not None and session.bind.dialect.name == "postgresql":
            lock_key = _external_id_lock_key(external_id)
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": lock_key},
            )

        if external_id is not None:
            existing = await PersonRepository.find_by_external_id(
                session, external_id,
            )
            if existing is not None:
                return existing, False

        stmt = (
            pg_insert(Person)
            .values(
                external_id=external_id, full_name=full_name,
                doc_type=doc_type, doc_number=doc_number,
                sex=sex, dob=dob, notes=notes,
            )
            .on_conflict_do_nothing(index_elements=["external_id"])
            .returning(Person)
        )
        result = await session.execute(stmt)
        p = result.scalar_one_or_none()
        await session.commit()
        if p is not None:
            return p, True
        if external_id is None:
            msg = "ON CONFLICT fired without an external_id to look up"
            raise RuntimeError(msg)
        existing = await PersonRepository.find_by_external_id(
            session, external_id,
        )
        if existing is None:
            msg = "ON CONFLICT fired but row is not queryable"
            raise RuntimeError(msg)
        return existing, False

    @staticmethod
    async def get_by_id(session: AsyncSession, person_id: uuid.UUID) -> Person | None:
        return await session.get(Person, person_id)

    @staticmethod
    async def find_by_external_id(session: AsyncSession, external_id: str) -> Person | None:
        stmt = select(Person).where(Person.external_id == external_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 20,
        search: str | None = None,
    ) -> list[Person]:
        stmt = select(Person)
        if search is not None:
            like = f"%{search}%"
            stmt = stmt.where(
                Person.full_name.ilike(like) | Person.external_id.ilike(like)
            )
        stmt = stmt.order_by(Person.created_at.desc()).offset(skip).limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def update(
        session: AsyncSession,
        person_id: uuid.UUID,
        **fields: Any,
    ) -> Person | None:
        p = await session.get(Person, person_id)
        if p is None:
            return None
        for key, value in fields.items():
            if value is not None and hasattr(p, key):
                setattr(p, key, value)
        await session.commit()
        await session.refresh(p)
        return p

    @staticmethod
    async def delete(session: AsyncSession, person_id: uuid.UUID) -> bool:
        p = await session.get(Person, person_id)
        if p is None:
            return False
        await session.delete(p)
        await session.commit()
        return True

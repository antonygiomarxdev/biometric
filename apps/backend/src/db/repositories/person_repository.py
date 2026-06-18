"""Async repository for :class:`~src.db.models.Person`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

from sqlalchemy import select

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Person


class PersonRepository:
    """Async persistence gateway for the ``persons`` table."""

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
    ) -> Person:
        p = Person(
            external_id=external_id, full_name=full_name,
            doc_type=doc_type, doc_number=doc_number,
            sex=sex, dob=dob, notes=notes,
        )
        session.add(p)
        await session.commit()
        await session.refresh(p)
        return p

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

"""
Async repository for :class:`~src.db.models.Case` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``Case``,
``select()``, ``func``, or ``desc()``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Case


class CaseRepository:
    """Async persistence gateway for the ``cases`` table."""

    @staticmethod
    async def list(
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 20,
        status: str | None = None,
    ) -> list[Case]:
        query = select(Case)
        if status is not None:
            query = query.where(Case.status == status)
        query = query.order_by(Case.created_at.desc()).offset(skip).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def count(
        session: AsyncSession,
        *,
        status: str | None = None,
    ) -> int:
        query = select(func.count(Case.id))
        if status is not None:
            query = query.where(Case.status == status)
        result = await session.execute(query)
        return result.scalar() or 0

    @staticmethod
    async def get_by_id(session: AsyncSession, case_id: uuid.UUID) -> Case | None:
        return await session.get(Case, case_id)

    @staticmethod
    async def get_by_case_number(
        session: AsyncSession, case_number: str
    ) -> Case | None:
        result = await session.execute(
            select(Case).where(Case.case_number == case_number)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        case_number: str,
        title: str,
        description: str | None = None,
        status: str = "open",
    ) -> Case:
        case = Case(
            case_number=case_number,
            title=title,
            description=description or "",
            status=status,
        )
        session.add(case)
        await session.commit()
        await session.refresh(case)
        return case

    @staticmethod
    async def update(
        session: AsyncSession,
        case: Case,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
    ) -> Case:
        if title is not None:
            case.title = title
        if description is not None:
            case.description = description
        if status is not None:
            case.status = status
        await session.commit()
        await session.refresh(case)
        return case

    @staticmethod
    async def delete(
        session: AsyncSession,
        case: Case | uuid.UUID,
    ) -> None:
        if isinstance(case, uuid.UUID):
            case_obj = await session.get(Case, case)
            if case_obj is not None:
                await session.delete(case_obj)
                await session.commit()
        else:
            await session.delete(case)
            await session.commit()

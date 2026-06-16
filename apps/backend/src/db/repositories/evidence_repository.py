"""
Async repository for :class:`~src.db.models.Evidence` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``Evidence``,
``select()``, ``func``, or ``desc()``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Evidence


class EvidenceRepository:
    """Async persistence gateway for the ``evidences`` table."""

    @staticmethod
    async def list(
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 20,
        case_id: uuid.UUID | None = None,
    ) -> list[Evidence]:
        query = select(Evidence)
        if case_id is not None:
            query = query.where(Evidence.case_id == case_id)
        query = query.order_by(Evidence.created_at.desc()).offset(skip).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def count(
        session: AsyncSession,
        *,
        case_id: uuid.UUID | None = None,
    ) -> int:
        query = select(func.count(Evidence.id))
        if case_id is not None:
            query = query.where(Evidence.case_id == case_id)
        result = await session.execute(query)
        return result.scalar() or 0

    @staticmethod
    async def get_by_id(session: AsyncSession, evidence_id: uuid.UUID) -> Evidence | None:
        return await session.get(Evidence, evidence_id)

    @staticmethod
    async def get_by_case_id(
        session: AsyncSession, case_id: uuid.UUID
    ) -> list[Evidence]:
        result = await session.execute(
            select(Evidence).where(Evidence.case_id == case_id)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_image_path(
        session: AsyncSession, evidence_id: uuid.UUID
    ) -> str | None:
        ev = await session.get(Evidence, evidence_id)
        if ev is None:
            return None
        return ev.image_path

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        case_id: uuid.UUID,
        fingerprint_id: str,
        image_path: str | None = None,
    ) -> Evidence:
        ev = Evidence(
            case_id=case_id,
            fingerprint_id=fingerprint_id,
            image_path=image_path,
        )
        session.add(ev)
        await session.commit()
        await session.refresh(ev)
        return ev

    @staticmethod
    async def delete(
        session: AsyncSession,
        evidence: Evidence | uuid.UUID,
    ) -> None:
        if isinstance(evidence, uuid.UUID):
            ev_obj = await session.get(Evidence, evidence)
            if ev_obj is not None:
                await session.delete(ev_obj)
                await session.commit()
        else:
            await session.delete(evidence)
            await session.commit()

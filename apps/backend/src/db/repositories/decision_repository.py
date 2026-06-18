"""
Async repository for :class:`~src.db.models.Decision` — encapsulates ALL
SQLAlchemy query logic so the service layer never imports ``Decision``,
``select()``, ``func``, or ``desc()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Decision


class DecisionRepository:
    """Async persistence gateway for the ``decisions`` table."""

    @staticmethod
    async def list(
        session: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 20,
        case_id: uuid.UUID | None = None,
        verdict: str | None = None,
    ) -> list[Decision]:
        query = select(Decision)
        if case_id is not None:
            query = query.where(Decision.case_id == case_id)
        if verdict is not None:
            query = query.where(Decision.verdict == verdict)
        query = query.order_by(Decision.created_at.desc()).offset(skip).limit(limit)
        result = await session.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def count(
        session: AsyncSession,
        *,
        case_id: uuid.UUID | None = None,
        verdict: str | None = None,
    ) -> int:
        query = select(func.count(Decision.id))
        if case_id is not None:
            query = query.where(Decision.case_id == case_id)
        if verdict is not None:
            query = query.where(Decision.verdict == verdict)
        result = await session.execute(query)
        return result.scalar() or 0

    @staticmethod
    async def get_by_id(
        session: AsyncSession, decision_id: uuid.UUID
    ) -> Decision | None:
        return await session.get(Decision, decision_id)

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        case_id: uuid.UUID,
        evidence_id: uuid.UUID | None = None,
        verdict: str,
        comments: str | None = None,
    ) -> Decision:
        decision = Decision(
            case_id=case_id,
            evidence_id=evidence_id,
            verdict=verdict,
            comments=comments,
        )
        session.add(decision)
        await session.flush()
        return decision

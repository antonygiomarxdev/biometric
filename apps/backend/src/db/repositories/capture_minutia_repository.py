"""Async repository for :class:`~src.db.models.CaptureMinutia`."""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import CaptureMinutia

if TYPE_CHECKING:
    import uuid
    pass


class CaptureMinutiaRepository:
    """Persistence gateway for the capture_minutiae table."""

    @staticmethod
    async def bulk_insert(
        session: AsyncSession,
        *,
        capture_id: "uuid.UUID",
        person_id: "uuid.UUID",
        minutiae: list[dict],
        algo_version: str = "pairs-v1",
    ) -> int:
        if not minutiae:
            return 0
        rows = [
            {
                "capture_id": capture_id,
                "person_id": person_id,
                "minutia_index": m["index"],
                "x": m["x"],
                "y": m["y"],
                "angle": m["angle"],
                "type": m["type"],
                "quality": m["quality"],
                "hash": m["hash"],
                "algo_version": algo_version,
            }
            for m in minutiae
        ]
        stmt = pg_insert(CaptureMinutia.__table__).values(rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["capture_id", "minutia_index"],
        )
        await session.execute(stmt)
        return len(rows)

    @staticmethod
    async def list_for_capture(
        session: AsyncSession,
        capture_id: "uuid.UUID",
    ) -> list[CaptureMinutia]:
        result = await session.execute(
            select(CaptureMinutia)
            .where(CaptureMinutia.capture_id == capture_id)
            .order_by(CaptureMinutia.minutia_index)
        )
        return list(result.scalars().all())

    @staticmethod
    async def list_for_person(
        session: AsyncSession,
        person_id: "uuid.UUID",
    ) -> list[CaptureMinutia]:
        result = await session.execute(
            select(CaptureMinutia)
            .where(CaptureMinutia.person_id == person_id)
            .order_by(CaptureMinutia.capture_id, CaptureMinutia.minutia_index)
        )
        return list(result.scalars().all())

    @staticmethod
    async def delete_for_capture(
        session: AsyncSession,
        capture_id: "uuid.UUID",
    ) -> int:
        result = await session.execute(
            delete(CaptureMinutia).where(CaptureMinutia.capture_id == capture_id)
        )
        return result.rowcount or 0

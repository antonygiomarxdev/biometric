"""Async repository for :class:`~src.db.models.RidgeGraph`."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import RidgeGraph


class RidgeGraphRepository:
    """Async persistence gateway for the ``ridge_graphs`` table."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        capture_id: uuid.UUID,
        graph_index: int = 1,
        region_x: int = 0,
        region_y: int = 0,
        region_w: int = 0,
        region_h: int = 0,
        num_nodes: int = 0,
        num_edges: int = 0,
        graph_data: dict | None = None,
        core_x: int | None = None,
        core_y: int | None = None,
        delta_x: int | None = None,
        delta_y: int | None = None,
        singularity_type: str | None = None,
    ) -> RidgeGraph:
        g = RidgeGraph(
            capture_id=capture_id,
            graph_index=graph_index,
            region_x=region_x, region_y=region_y,
            region_w=region_w, region_h=region_h,
            num_nodes=num_nodes, num_edges=num_edges,
            graph_data=graph_data or {},
            core_x=core_x, core_y=core_y,
            delta_x=delta_x, delta_y=delta_y,
            singularity_type=singularity_type,
        )
        session.add(g)
        await session.commit()
        await session.refresh(g)
        return g

    @staticmethod
    async def get_by_id(session: AsyncSession, graph_id: uuid.UUID) -> RidgeGraph | None:
        return await session.get(RidgeGraph, graph_id)

    @staticmethod
    async def list_by_capture(
        session: AsyncSession,
        capture_id: uuid.UUID,
    ) -> list[RidgeGraph]:
        stmt = (
            select(RidgeGraph)
            .where(RidgeGraph.capture_id == capture_id)
            .order_by(RidgeGraph.graph_index)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def count_by_capture(session: AsyncSession, capture_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(RidgeGraph).where(
            RidgeGraph.capture_id == capture_id
        )
        result = await session.execute(stmt)
        return int(result.scalar_one())

    @staticmethod
    async def delete(session: AsyncSession, graph_id: uuid.UUID) -> bool:
        g = await session.get(RidgeGraph, graph_id)
        if g is None:
            return False
        await session.delete(g)
        await session.commit()
        return True

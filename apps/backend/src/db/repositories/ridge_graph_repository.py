"""Repository for :class:`~src.db.models.RidgeGraph`."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import RidgeGraph


class RidgeGraphRepository:
    """Persistence gateway for the ``ridge_graphs`` table."""

    @staticmethod
    def create(
        session: Session,
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
        session.commit()
        session.refresh(g)
        return g

    @staticmethod
    def get_by_id(session: Session, graph_id: uuid.UUID) -> RidgeGraph | None:
        return session.get(RidgeGraph, graph_id)

    @staticmethod
    def list_by_capture(
        session: Session,
        capture_id: uuid.UUID,
    ) -> list[RidgeGraph]:
        stmt = (
            select(RidgeGraph)
            .where(RidgeGraph.capture_id == capture_id)
            .order_by(RidgeGraph.graph_index)
        )
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    def count_by_capture(session: Session, capture_id: uuid.UUID) -> int:
        stmt = select(func.count()).select_from(RidgeGraph).where(
            RidgeGraph.capture_id == capture_id
        )
        return int(session.execute(stmt).scalar_one())

    @staticmethod
    def delete(session: Session, graph_id: uuid.UUID) -> bool:
        g = session.get(RidgeGraph, graph_id)
        if g is None:
            return False
        session.delete(g)
        session.commit()
        return True

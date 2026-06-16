"""Tests for RidgeGraphRepository (Phase 17, async)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Fingerprint, Person, RidgeGraph
from src.db.repositories.fingerprint_capture_repository import FingerprintCaptureRepository
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.db.repositories.ridge_graph_repository import RidgeGraphRepository


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_pragma(dbapi_con, _):  # noqa: ARG001
        dbapi_con.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    s = Session()
    yield s
    await s.close()
    await engine.dispose()


@pytest.fixture
async def capture(session):
    p = await PersonRepository.create(session, external_id="X")
    f = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
    c = await FingerprintCaptureRepository.create(
        session, fingerprint_id=f.id,
        image_uri="minio://1.png", image_hash_sha256="h",
    )
    return c


@pytest.mark.asyncio
class TestRidgeGraphRepository:
    async def test_create_graph(self, session, capture) -> None:
        g = await RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_index=1,
            num_nodes=5, num_edges=4, graph_data={"k": "v"},
        )
        assert g.id is not None
        assert g.graph_data == {"k": "v"}

    async def test_create_with_singularity(self, session, capture) -> None:
        g = await RidgeGraphRepository.create(
            session, capture_id=capture.id,
            num_nodes=10, graph_data={},
            core_x=100, core_y=200, singularity_type="core",
        )
        assert g.core_x == 100
        assert g.singularity_type == "core"

    async def test_get_by_id(self, session, capture) -> None:
        g = await RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_data={},
        )
        found = await RidgeGraphRepository.get_by_id(session, g.id)
        assert found is not None

    async def test_get_by_id_missing(self, session) -> None:
        assert await RidgeGraphRepository.get_by_id(session, uuid.uuid4()) is None

    async def test_list_by_capture_ordered(self, session, capture) -> None:
        for i in range(3):
            await RidgeGraphRepository.create(
                session, capture_id=capture.id, graph_index=i + 1, graph_data={},
            )
        items = await RidgeGraphRepository.list_by_capture(session, capture.id)
        assert len(items) == 3
        assert items[0].graph_index == 1
        assert items[2].graph_index == 3

    async def test_count_by_capture(self, session, capture) -> None:
        assert await RidgeGraphRepository.count_by_capture(session, capture.id) == 0
        await RidgeGraphRepository.create(session, capture_id=capture.id, graph_data={})
        assert await RidgeGraphRepository.count_by_capture(session, capture.id) == 1

    async def test_delete(self, session, capture) -> None:
        g = await RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_data={},
        )
        assert await RidgeGraphRepository.delete(session, g.id) is True
        assert await RidgeGraphRepository.get_by_id(session, g.id) is None

    async def test_delete_missing(self, session) -> None:
        assert await RidgeGraphRepository.delete(session, uuid.uuid4()) is False

    async def test_graph_data_jsonb_round_trip(self, session, capture) -> None:
        data = {"nodes": [{"x": 1, "y": 2}], "edges": []}
        g = await RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_data=data,
        )
        assert g.graph_data == data
        fetched = await RidgeGraphRepository.get_by_id(session, g.id)
        assert fetched is not None
        assert fetched.graph_data == data

"""Tests for PersonRepository (Phase 17)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Person
from src.db.repositories.person_repository import PersonRepository


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


class TestPersonRepository:
    @pytest.mark.asyncio
    async def test_create_minimal(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X")
        assert p.id is not None
        assert p.external_id == "X"

    @pytest.mark.asyncio
    async def test_create_with_all_fields(self, session) -> None:
        p = await PersonRepository.create(
            session,
            external_id="X", full_name="Juan", doc_type="cedula",
            doc_number="001", sex="M",
        )
        assert p.full_name == "Juan"
        assert p.doc_type == "cedula"

    @pytest.mark.asyncio
    async def test_get_by_id_returns_none_for_missing(self, session) -> None:
        assert await PersonRepository.get_by_id(session, uuid.uuid4()) is None

    @pytest.mark.asyncio
    async def test_find_by_external_id(self, session) -> None:
        await PersonRepository.create(session, external_id="ABC")
        found = await PersonRepository.find_by_external_id(session, "ABC")
        assert found is not None
        assert found.external_id == "ABC"

    @pytest.mark.asyncio
    async def test_find_by_external_id_missing(self, session) -> None:
        assert await PersonRepository.find_by_external_id(session, "NOPE") is None

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, session) -> None:
        for i in range(5):
            await PersonRepository.create(session, external_id=f"P{i}")
        assert len(await PersonRepository.list(session, skip=0, limit=3)) == 3
        assert len(await PersonRepository.list(session, skip=3, limit=10)) == 2

    @pytest.mark.asyncio
    async def test_list_with_search_filter(self, session) -> None:
        await PersonRepository.create(session, external_id="A", full_name="Juan Pérez")
        await PersonRepository.create(session, external_id="B", full_name="Pedro Gómez")
        results = await PersonRepository.list(session, search="juan")
        assert len(results) == 1
        assert results[0].full_name == "Juan Pérez"

    @pytest.mark.asyncio
    async def test_update_fields(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X", full_name="Old")
        updated = await PersonRepository.update(session, p.id, full_name="New")
        assert updated is not None
        assert updated.full_name == "New"

    @pytest.mark.asyncio
    async def test_delete_returns_false_for_missing(self, session) -> None:
        assert await PersonRepository.delete(session, uuid.uuid4()) is False

    @pytest.mark.asyncio
    async def test_delete_success(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X")
        assert await PersonRepository.delete(session, p.id) is True
        assert await PersonRepository.get_by_id(session, p.id) is None

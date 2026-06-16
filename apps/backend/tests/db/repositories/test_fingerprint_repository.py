"""Tests for FingerprintRepository (Phase 17)."""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Fingerprint, Person
from src.db.repositories.fingerprint_repository import FingerprintRepository
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


@pytest.fixture
async def person(session) -> Person:
    return await PersonRepository.create(session, external_id="X")


class TestFingerprintRepository:
    @pytest.mark.asyncio
    async def test_create_slot(self, session, person) -> None:
        f = await FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert f.id is not None
        assert f.person_id == person.id

    @pytest.mark.asyncio
    async def test_list_by_person_returns_in_order(self, session, person) -> None:
        await FingerprintRepository.create(session, person_id=person.id, finger_position=10)
        await FingerprintRepository.create(session, person_id=person.id, finger_position=2)
        items = await FingerprintRepository.list_by_person(session, person.id)
        assert len(items) == 2
        assert items[0].finger_position == 2
        assert items[1].finger_position == 10

    @pytest.mark.asyncio
    async def test_find_slot_returns_existing(self, session, person) -> None:
        await FingerprintRepository.create(
            session, person_id=person.id, finger_position=3, capture_type="latent",
        )
        found = await FingerprintRepository.find_slot(
            session, person.id, 3, "latent",
        )
        assert found is not None

    @pytest.mark.asyncio
    async def test_find_slot_returns_none_for_missing(self, session, person) -> None:
        found = await FingerprintRepository.find_slot(
            session, person.id, 99, "rolled",
        )
        assert found is None

    @pytest.mark.asyncio
    async def test_increment_capture_count(self, session, person) -> None:
        f = await FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert f.capture_count == 0
        await FingerprintRepository.increment_capture_count(session, f.id)
        result = await FingerprintRepository.get_by_id(session, f.id)
        assert result is not None
        assert result.capture_count == 1

    @pytest.mark.asyncio
    async def test_increment_capture_count_sets_first_captured_at(self, session, person) -> None:
        f = await FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert f.first_captured_at is None
        await FingerprintRepository.increment_capture_count(session, f.id)
        await session.refresh(f)
        assert f.first_captured_at is not None
        assert f.last_captured_at is not None

    @pytest.mark.asyncio
    async def test_create_duplicate_slot_raises(self, session, person) -> None:
        await FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        with pytest.raises(Exception):
            await FingerprintRepository.create(
                session, person_id=person.id, finger_position=2,
            )

    @pytest.mark.asyncio
    async def test_delete(self, session, person) -> None:
        f = await FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert await FingerprintRepository.delete(session, f.id) is True
        assert await FingerprintRepository.get_by_id(session, f.id) is None

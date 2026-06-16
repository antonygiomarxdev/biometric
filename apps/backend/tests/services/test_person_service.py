"""Async tests for PersonService."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base
from src.schemas.person_schema import PersonCreate
from src.services.person_service import PersonService


@pytest.fixture
async def session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def service(session: AsyncSession) -> PersonService:
    return PersonService(session)


class TestCreatePerson:
    async def test_creates_with_minimal_data(self, service: PersonService) -> None:
        p = await service.create_person(PersonCreate(external_id="X"))
        assert p.id is not None
        assert p.external_id == "X"

    async def test_duplicate_external_id_raises(self, service: PersonService) -> None:
        await service.create_person(PersonCreate(external_id="X"))
        with pytest.raises(ValueError, match="already exists"):
            await service.create_person(PersonCreate(external_id="X"))

    async def test_normalizes_doc_type(self, service: PersonService) -> None:
        p = await service.create_person(PersonCreate(external_id="X", doc_type="CEDULA"))
        assert p.doc_type == "cedula"

    async def test_unknown_doc_type_kept_as_is(self, service: PersonService) -> None:
        p = await service.create_person(PersonCreate(external_id="X", doc_type="custom"))
        assert p.doc_type == "custom"

    async def test_sex_uppercased(self, service: PersonService) -> None:
        p = await service.create_person(PersonCreate(external_id="X", sex="m"))
        assert p.sex == "M"


class TestListAndGet:
    async def test_list_empty(self, service: PersonService) -> None:
        assert await service.list_persons() == []

    async def test_list_returns_persons(self, service: PersonService) -> None:
        await service.create_person(PersonCreate(external_id="A"))
        await service.create_person(PersonCreate(external_id="B"))
        assert len(await service.list_persons()) == 2

    async def test_search_filter(self, service: PersonService) -> None:
        await service.create_person(PersonCreate(external_id="A", full_name="Juan"))
        await service.create_person(PersonCreate(external_id="B", full_name="Pedro"))
        results = await service.list_persons(search="juan")
        assert len(results) == 1
        assert results[0].full_name == "Juan"

    async def test_get_person_by_id(self, service: PersonService) -> None:
        p = await service.create_person(PersonCreate(external_id="X"))
        found = await service.get_person(p.id)
        assert found is not None
        assert found.id == p.id

    async def test_get_person_not_found(self, service: PersonService) -> None:
        import uuid
        assert await service.get_person(uuid.uuid4()) is None


class TestFindOrCreate:
    async def test_returns_existing(self, service: PersonService) -> None:
        p1 = await service.find_or_create_person("X")
        p2 = await service.find_or_create_person("X")
        assert p1.id == p2.id

    async def test_creates_when_missing(self, service: PersonService) -> None:
        p = await service.find_or_create_person("NEW", full_name="Test")
        assert p.full_name == "Test"


class TestDeletePerson:
    async def test_delete(self, service: PersonService) -> None:
        p = await service.create_person(PersonCreate(external_id="X"))
        assert await service.delete_person(p.id) is True
        assert await service.get_person(p.id) is None

    async def test_delete_missing(self, service: PersonService) -> None:
        import uuid
        assert await service.delete_person(uuid.uuid4()) is False

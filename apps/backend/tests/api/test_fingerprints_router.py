"""Tests for fingerprints router (Phase 17, async)."""

from __future__ import annotations
from typing import Any, AsyncGenerator

import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_async_db
from src.api.routers.fingerprints import router as fingerprints_router
from src.db.models import Base, Person
from src.db.repositories.person_repository import PersonRepository


@pytest.fixture
async def engine_session() -> AsyncGenerator[tuple[Any, Any], None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    yield engine, Session_factory
    await engine.dispose()


@pytest.fixture
async def app() -> FastAPI:
    app = FastAPI()
    app.include_router(fingerprints_router)
    return app


@pytest.fixture
async def client(
    app: FastAPI,
    engine_session: tuple[Any, Any],
) -> AsyncGenerator[AsyncClient, None]:
    _, Session_factory = engine_session

    async def _get_async_db_override() -> AsyncGenerator[AsyncSession, None]:
        async with Session_factory() as session:
            yield session

    app.dependency_overrides[get_async_db] = _get_async_db_override
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _make_person(engine_session: tuple[Any, Any]) -> Person:
    _, Session_factory = engine_session
    async with Session_factory() as s:
        p = await PersonRepository.create(s, external_id="X")
    return p


@pytest.mark.asyncio
class TestFingerprintsRouter:
    async def test_create_fingerprint_for_person(
        self, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        person = await _make_person(engine_session)
        pid = str(person.id)
        resp = await client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        assert resp.status_code == 201

    async def test_create_duplicate_slot_returns_409(
        self, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        person = await _make_person(engine_session)
        pid = str(person.id)
        await client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        resp = await client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        assert resp.status_code == 409

    async def test_create_for_missing_person_returns_404(
        self, client: AsyncClient
    ) -> None:
        resp = await client.post(
            f"/api/v1/persons/{uuid.uuid4()}/fingerprints", json={
                "finger_position": 2,
            },
        )
        assert resp.status_code == 404

    async def test_list_fingerprints_empty(
        self, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        person = await _make_person(engine_session)
        resp = await client.get(f"/api/v1/persons/{person.id}/fingerprints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    async def test_list_fingerprints_returns_in_position_order(
        self, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        person = await _make_person(engine_session)
        pid = str(person.id)
        await client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 10, "capture_type": "rolled",
        })
        await client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "latent",
        })
        resp = await client.get(f"/api/v1/persons/{pid}/fingerprints")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2
        assert items[0]["finger_position"] == 2
        assert items[1]["finger_position"] == 10

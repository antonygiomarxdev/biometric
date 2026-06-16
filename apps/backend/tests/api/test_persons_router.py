"""Async tests for persons router (Phase 17)."""

from __future__ import annotations
from typing import AsyncGenerator

import uuid

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.api.dependencies import get_async_db
from src.api.routers.persons import router as persons_router
from src.db.models import Base


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(persons_router)
    return app


@pytest.fixture
async def async_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    async with async_session() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def client(app: FastAPI, async_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    async def _override() -> AsyncGenerator[AsyncSession, None]:
        yield async_session

    app.dependency_overrides[get_async_db] = _override
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
class TestPersonsRouter:
    async def test_create_person(self, client: AsyncClient) -> None:
        async with client as c:
            resp = await c.post("/api/v1/persons/", json={
                "external_id": "X-001", "full_name": "Juan Pérez",
            })
        assert resp.status_code == 201
        data = resp.json()
        assert data["external_id"] == "X-001"
        assert data["full_name"] == "Juan Pérez"

    async def test_create_person_duplicate_returns_409(self, client: AsyncClient) -> None:
        async with client as c:
            await c.post("/api/v1/persons/", json={"external_id": "X"})
            resp = await c.post("/api/v1/persons/", json={"external_id": "X"})
        assert resp.status_code == 409

    async def test_get_person_not_found_returns_404(self, client: AsyncClient) -> None:
        async with client as c:
            resp = await c.get(f"/api/v1/persons/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_list_persons_with_pagination(self, client: AsyncClient) -> None:
        async with client as c:
            for i in range(3):
                await c.post("/api/v1/persons/", json={"external_id": f"P{i}"})
            resp = await c.get("/api/v1/persons/?skip=0&limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    async def test_list_persons_with_search_filter(self, client: AsyncClient) -> None:
        async with client as c:
            await c.post("/api/v1/persons/", json={
                "external_id": "A", "full_name": "Juan Pérez",
            })
            await c.post("/api/v1/persons/", json={
                "external_id": "B", "full_name": "Pedro Gómez",
            })
            resp = await c.get("/api/v1/persons/?search=juan")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["full_name"] == "Juan Pérez"

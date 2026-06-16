"""Tests for persons router (Phase 17)."""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_db
from src.api.routers.persons import router as persons_router
from src.db.models import Base


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(persons_router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    def _get_db_override():
        session = Session()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _get_db_override
    with TestClient(app) as c:
        yield c


class TestPersonsRouter:
    def test_create_person(self, client: TestClient) -> None:
        resp = client.post("/api/v1/persons/", json={
            "external_id": "X-001", "full_name": "Juan Pérez",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["external_id"] == "X-001"
        assert data["full_name"] == "Juan Pérez"

    def test_create_person_duplicate_returns_409(self, client: TestClient) -> None:
        client.post("/api/v1/persons/", json={"external_id": "X"})
        resp = client.post("/api/v1/persons/", json={"external_id": "X"})
        assert resp.status_code == 409

    def test_get_person_not_found_returns_404(self, client: TestClient) -> None:
        resp = client.get(f"/api/v1/persons/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_list_persons_with_pagination(self, client: TestClient) -> None:
        for i in range(3):
            client.post("/api/v1/persons/", json={"external_id": f"P{i}"})
        resp = client.get("/api/v1/persons/?skip=0&limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_persons_with_search_filter(self, client: TestClient) -> None:
        client.post("/api/v1/persons/", json={
            "external_id": "A", "full_name": "Juan Pérez",
        })
        client.post("/api/v1/persons/", json={
            "external_id": "B", "full_name": "Pedro Gómez",
        })
        resp = client.get("/api/v1/persons/?search=juan")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["full_name"] == "Juan Pérez"

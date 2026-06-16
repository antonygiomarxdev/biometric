"""Tests for fingerprints router (Phase 17)."""

from __future__ import annotations
from typing import Generator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_db
from src.api.routers.persons import router as persons_router
from src.api.routers.fingerprints import router as fingerprints_router
from src.db.models import Base


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(persons_router)
    app.include_router(fingerprints_router)
    return app


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient, None, None]:
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


class TestFingerprintsRouter:
    def _create_person(self, client: TestClient, external_id: str = "X") -> str:
        resp = client.post("/api/v1/persons/", json={"external_id": external_id})
        return resp.json()["id"]

    def test_create_fingerprint_for_person(self, client: TestClient) -> None:
        pid = self._create_person(client)
        resp = client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["finger_position"] == 2
        assert data["person_id"] == pid

    def test_create_duplicate_slot_returns_409(self, client: TestClient) -> None:
        pid = self._create_person(client)
        client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        resp = client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        assert resp.status_code == 409

    def test_create_for_missing_person_returns_404(self, client: TestClient) -> None:
        import uuid
        resp = client.post(f"/api/v1/persons/{uuid.uuid4()}/fingerprints", json={
            "finger_position": 2,
        })
        assert resp.status_code == 404

    def test_list_fingerprints_empty(self, client: TestClient) -> None:
        pid = self._create_person(client)
        resp = client.get(f"/api/v1/persons/{pid}/fingerprints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_fingerprints_returns_in_position_order(self, client: TestClient) -> None:
        pid = self._create_person(client)
        client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 10, "capture_type": "rolled",
        })
        client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "latent",
        })
        resp = client.get(f"/api/v1/persons/{pid}/fingerprints")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 2
        assert items[0]["finger_position"] == 2
        assert items[1]["finger_position"] == 10

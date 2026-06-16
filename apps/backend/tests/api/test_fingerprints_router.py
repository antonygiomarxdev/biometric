"""Tests for fingerprints router (Phase 17)."""

from __future__ import annotations
from typing import Any, Generator

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_db
from src.api.routers.fingerprints import router as fingerprints_router
from src.db.models import Base, Person
from src.db.repositories.person_repository import PersonRepository


@pytest.fixture
def engine_session() -> Generator[tuple[Any, Any], None, None]:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    Session_factory = sessionmaker(bind=engine)
    yield engine, Session_factory
    engine.dispose()


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(fingerprints_router)
    return app


@pytest.fixture
def client(app: FastAPI, engine_session: tuple[Any, Any]) -> Generator[TestClient, None, None]:
    _, Session_factory = engine_session

    def _get_db_override():
        session = Session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _get_db_override
    with TestClient(app) as c:
        yield c


def _make_person(engine_session: tuple[Any, Any]) -> Person:
    _, Session_factory = engine_session
    s = Session_factory()
    p = PersonRepository.create(s, external_id="X")
    s.close()
    return p


class TestFingerprintsRouter:
    def test_create_fingerprint_for_person(self, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        person = _make_person(engine_session)
        pid = str(person.id)
        resp = client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        assert resp.status_code == 201

    def test_create_duplicate_slot_returns_409(self, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        person = _make_person(engine_session)
        pid = str(person.id)
        client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        resp = client.post(f"/api/v1/persons/{pid}/fingerprints", json={
            "finger_position": 2, "capture_type": "rolled",
        })
        assert resp.status_code == 409

    def test_create_for_missing_person_returns_404(self, client: TestClient) -> None:
        resp = client.post(f"/api/v1/persons/{uuid.uuid4()}/fingerprints", json={
            "finger_position": 2,
        })
        assert resp.status_code == 404

    def test_list_fingerprints_empty(self, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        person = _make_person(engine_session)
        resp = client.get(f"/api/v1/persons/{person.id}/fingerprints")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_fingerprints_returns_in_position_order(self, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        person = _make_person(engine_session)
        pid = str(person.id)
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

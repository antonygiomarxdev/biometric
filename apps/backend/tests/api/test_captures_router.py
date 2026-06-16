"""Tests for captures router (Phase 17)."""

from __future__ import annotations
from typing import Any, Generator

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_db, get_fingerprint_service
from src.api.routers.captures import router as captures_router
from src.api.routers.fingerprints import router as fingerprints_router
from src.db.models import Base, Person
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.services.fingerprint_service import FingerprintService


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
    app.include_router(captures_router)
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

    mock_fp = MagicMock(spec=FingerprintService)
    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[get_fingerprint_service] = lambda: mock_fp
    with TestClient(app) as c:
        yield c


def _create_person(engine_session: tuple[Any, Any]) -> Person:
    _, Session_factory = engine_session
    s = Session_factory()
    p = PersonRepository.create(s, external_id="X")
    s.close()
    return p


def _create_fingerprint(engine_session: tuple[Any, Any], person_id: uuid.UUID) -> str:
    _, Session_factory = engine_session
    s = Session_factory()
    fp = FingerprintRepository.create(s, person_id=person_id, finger_position=2, capture_type="rolled")
    fid = str(fp.id)
    s.close()
    return fid


class TestCapturesRouter:
    @patch("cv2.imdecode")
    def test_upload_capture(self, mock_decode, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        person = _create_person(engine_session)
        fid = _create_fingerprint(engine_session, person.id)
        resp = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        assert resp.status_code == 201

    def test_upload_capture_empty_file_returns_400(self, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        person = _create_person(engine_session)
        fid = _create_fingerprint(engine_session, person.id)
        resp = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("empty.bmp", b"", "image/bmp")},
        )
        assert resp.status_code == 400

    def test_get_capture_not_found(self, client: TestClient) -> None:
        resp = client.get(f"/api/v1/captures/{uuid.uuid4()}")
        assert resp.status_code == 404

    @patch("cv2.imdecode")
    def test_get_capture_graphs(self, mock_decode, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        person = _create_person(engine_session)
        fid = _create_fingerprint(engine_session, person.id)
        upload = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        cid = upload.json()["capture"]["id"]
        resp = client.get(f"/api/v1/captures/{cid}/graphs")
        assert resp.status_code == 200

    @patch("cv2.imdecode")
    def test_update_capture(self, mock_decode, client: TestClient, engine_session: tuple[Any, Any]) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        person = _create_person(engine_session)
        fid = _create_fingerprint(engine_session, person.id)
        upload = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        cid = upload.json()["capture"]["id"]
        resp = client.patch(
            f"/api/v1/captures/{cid}",
            json={"is_reference": True},
        )
        assert resp.status_code == 200
        assert resp.json()["is_reference"] is True

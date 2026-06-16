"""Tests for captures router (Phase 17)."""

from __future__ import annotations
from typing import Generator

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
from src.api.routers.persons import router as persons_router
from src.db.models import Base
from src.services.fingerprint_service import FingerprintService


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(persons_router)
    app.include_router(fingerprints_router)
    app.include_router(captures_router)
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

    mock_fp = MagicMock(spec=FingerprintService)
    app.dependency_overrides[get_db] = _get_db_override
    app.dependency_overrides[get_fingerprint_service] = lambda: mock_fp
    with TestClient(app) as c:
        yield c


class TestCapturesRouter:
    def _create_person(self, client: TestClient) -> str:
        resp = client.post("/api/v1/persons/", json={"external_id": "X"})
        return resp.json()["id"]

    def _create_fingerprint(self, client: TestClient, person_id: str) -> str:
        resp = client.post(
            f"/api/v1/persons/{person_id}/fingerprints",
            json={"finger_position": 2, "capture_type": "rolled"},
        )
        return resp.json()["id"]

    @patch("cv2.imdecode")
    def test_upload_capture(self, mock_decode, client: TestClient) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        pid = self._create_person(client)
        fid = self._create_fingerprint(client, pid)
        resp = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["graphs_created"] >= 0
        assert data["capture"]["fingerprint_id"] == fid

    def test_upload_capture_empty_file_returns_400(self, client: TestClient) -> None:
        pid = self._create_person(client)
        fid = self._create_fingerprint(client, pid)
        resp = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("empty.bmp", b"", "image/bmp")},
        )
        assert resp.status_code == 400
        assert "Empty" in resp.json()["detail"]

    def test_get_capture_not_found(self, client: TestClient) -> None:
        resp = client.get(f"/api/v1/captures/{uuid.uuid4()}")
        assert resp.status_code == 404

    @patch("cv2.imdecode")
    def test_get_capture_graphs(self, mock_decode, client: TestClient) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        pid = self._create_person(client)
        fid = self._create_fingerprint(client, pid)
        upload = client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        cid = upload.json()["capture"]["id"]
        resp = client.get(f"/api/v1/captures/{cid}/graphs")
        assert resp.status_code == 200

    @patch("cv2.imdecode")
    def test_update_capture(self, mock_decode, client: TestClient) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        pid = self._create_person(client)
        fid = self._create_fingerprint(client, pid)
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

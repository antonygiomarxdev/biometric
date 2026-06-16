"""Tests for captures router (Phase 17, async)."""

from __future__ import annotations
from typing import Any, AsyncGenerator

import io
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.api.dependencies import get_async_db, get_fingerprint_service
from src.api.routers.captures import router as captures_router
from src.api.routers.fingerprints import router as fingerprints_router
from src.db.models import Base, Person
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.services.fingerprint_service import FingerprintService


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
    app.include_router(captures_router)
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

    mock_fp = MagicMock(spec=FingerprintService)
    app.dependency_overrides[get_async_db] = _get_async_db_override
    app.dependency_overrides[get_fingerprint_service] = lambda: mock_fp
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _create_person(engine_session: tuple[Any, Any]) -> Person:
    _, Session_factory = engine_session
    async with Session_factory() as s:
        p = await PersonRepository.create(s, external_id="X")
    return p


async def _create_fingerprint(engine_session: tuple[Any, Any], person_id: uuid.UUID) -> str:
    _, Session_factory = engine_session
    async with Session_factory() as s:
        fp = await FingerprintRepository.create(
            s, person_id=person_id, finger_position=2, capture_type="rolled",
        )
        fid = str(fp.id)
    return fid


@pytest.mark.asyncio
class TestCapturesRouter:
    @patch("cv2.imdecode")
    async def test_upload_capture(
        self, mock_decode: MagicMock, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        person = await _create_person(engine_session)
        fid = await _create_fingerprint(engine_session, person.id)
        resp = await client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        assert resp.status_code == 201

    async def test_upload_capture_empty_file_returns_400(
        self, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        person = await _create_person(engine_session)
        fid = await _create_fingerprint(engine_session, person.id)
        resp = await client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("empty.bmp", b"", "image/bmp")},
        )
        assert resp.status_code == 400

    async def test_get_capture_not_found(self, client: AsyncClient) -> None:
        resp = await client.get(f"/api/v1/captures/{uuid.uuid4()}")
        assert resp.status_code == 404

    @patch("cv2.imdecode")
    async def test_get_capture_graphs(
        self, mock_decode: MagicMock, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        person = await _create_person(engine_session)
        fid = await _create_fingerprint(engine_session, person.id)
        upload = await client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        cid = upload.json()["capture"]["id"]
        resp = await client.get(f"/api/v1/captures/{cid}/graphs")
        assert resp.status_code == 200

    @patch("cv2.imdecode")
    async def test_update_capture(
        self, mock_decode: MagicMock, client: AsyncClient, engine_session: tuple[Any, Any]
    ) -> None:
        import numpy as np
        mock_decode.return_value = np.zeros((100, 100), dtype=np.uint8)
        person = await _create_person(engine_session)
        fid = await _create_fingerprint(engine_session, person.id)
        upload = await client.post(
            f"/api/v1/fingerprints/{fid}/captures",
            files={"file": ("test.bmp", io.BytesIO(b"fake"), "image/bmp")},
        )
        cid = upload.json()["capture"]["id"]
        resp = await client.patch(
            f"/api/v1/captures/{cid}",
            json={"is_reference": True},
        )
        assert resp.status_code == 200
        assert resp.json()["is_reference"] is True

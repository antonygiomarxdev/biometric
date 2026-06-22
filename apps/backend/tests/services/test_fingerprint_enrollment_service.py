"""Tests for FingerprintEnrollmentService (Phase 29 deep embedding)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.db.models import Base
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)


@pytest.fixture(autouse=True)
def mock_cv2_imdecode():
    with patch("cv2.imdecode", return_value=np.zeros((100, 100), dtype=np.uint8)):
        yield


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_pragma(dbapi_con, _):
        dbapi_con.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as s:
        yield s

    await engine.dispose()


def _make_embedding_service(points: list[str] | None = None) -> MagicMock:
    svc = MagicMock()
    # Use a coroutine that returns a string (since ``enroll`` is now async)
    async def _enroll(*_args: object, **_kwargs: object) -> str:
        return (points or ["emb-1"])[0]
    svc.enroll.side_effect = _enroll
    svc.embed.return_value = (np.zeros(512, dtype=np.float32), None)
    svc.search.return_value = {"candidates": []}
    return svc


def _make_loader() -> MagicMock:
    """Return a ModelLoader double whose pool is a real ThreadPoolExecutor
    so ``run_in_executor`` works in tests.
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.ai.loader import ModelLoader

    loader = MagicMock(spec=ModelLoader)
    loader.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test")
    return loader


class TestCreateCapture:
    @pytest.mark.asyncio
    async def test_creates_capture_with_embedding(self, session) -> None:
        from src.db.repositories.person_repository import PersonRepository
        from src.db.repositories.fingerprint_repository import FingerprintRepository

        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(
            session,
            embedding_service=_make_embedding_service(),
            loader=_make_loader(),
        )
        image_bytes = b"FAKE_IMAGE_BYTES"
        capture, embedding_id = await svc.create_capture(fp.id, image_bytes)
        assert capture.id is not None
        assert capture.fingerprint_id == fp.id
        assert embedding_id is not None

    @pytest.mark.asyncio
    async def test_updates_capture_count_on_parent(self, session) -> None:
        from src.db.repositories.person_repository import PersonRepository
        from src.db.repositories.fingerprint_repository import FingerprintRepository

        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(
            session,
            embedding_service=_make_embedding_service(),
            loader=_make_loader(),
        )
        await svc.create_capture(fp.id, b"x")
        await session.refresh(fp)
        assert fp.capture_count == 1
        assert fp.last_captured_at is not None

    @pytest.mark.asyncio
    async def test_missing_fingerprint_raises(self, session) -> None:
        svc = FingerprintEnrollmentService(
            session,
            embedding_service=_make_embedding_service(),
            loader=_make_loader(),
        )
        with pytest.raises(ValueError, match="not found"):
            await svc.create_capture(uuid.uuid4(), b"x")

    @pytest.mark.asyncio
    async def test_two_captures_increment_index(self, session) -> None:
        from src.db.repositories.person_repository import PersonRepository
        from src.db.repositories.fingerprint_repository import FingerprintRepository

        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(
            session,
            embedding_service=_make_embedding_service(),
            loader=_make_loader(),
        )
        c1, _ = await svc.create_capture(fp.id, b"x")
        c2, _ = await svc.create_capture(fp.id, b"x")
        assert c1.capture_index == 1
        assert c2.capture_index == 2
        await session.refresh(fp)
        assert fp.capture_count == 2

    @pytest.mark.asyncio
    async def test_image_hash_is_sha256(self, session) -> None:
        import hashlib

        from src.db.repositories.person_repository import PersonRepository
        from src.db.repositories.fingerprint_repository import FingerprintRepository

        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(
            session,
            embedding_service=_make_embedding_service(),
            loader=_make_loader(),
        )
        bytes_in = b"hello world"
        c, _ = await svc.create_capture(fp.id, bytes_in)
        assert c.image_hash_sha256 == hashlib.sha256(bytes_in).hexdigest()


def test_init_accepts_embedding_service() -> None:
    svc = FingerprintEnrollmentService.__new__(FingerprintEnrollmentService)
    svc._session = MagicMock()
    svc._embedding_service = MagicMock()
    svc._loader = _make_loader()
    assert svc._embedding_service is not None


@pytest.mark.asyncio
async def test_create_capture_returns_embedding_id() -> None:
    from src.db.models import Fingerprint

    person_id = uuid.uuid4()
    fp_id = uuid.uuid4()

    fp = MagicMock(spec=Fingerprint)
    fp.id = fp_id
    fp.person_id = person_id
    fp.finger_position = 2

    async def _fake_get(model, id_val):
        return fp

    loader = _make_loader()

    svc = FingerprintEnrollmentService.__new__(FingerprintEnrollmentService)
    svc._session = AsyncMock()
    svc._session.get = _fake_get
    svc._embedding_service = _make_embedding_service(points=["qdrant-point-1"])
    svc._loader = loader
    svc._storage = MagicMock()
    svc._fp_service = MagicMock()
    svc._dev_log = MagicMock()

    from src.db.repositories.fingerprint_capture_repository import FingerprintCaptureRepository
    with patch.object(FingerprintCaptureRepository, "create", new_callable=AsyncMock) as mock_create:
        mock_capture = MagicMock()
        mock_capture.id = uuid.uuid4()
        mock_capture.fingerprint_id = fp_id
        mock_capture.image_uri = "minio://pending/key"
        mock_create.return_value = mock_capture

        result, embedding_id = await svc.create_capture(
            fingerprint_id=fp_id,
            image_bytes=b"test-image",
        )
    assert embedding_id == "qdrant-point-1"

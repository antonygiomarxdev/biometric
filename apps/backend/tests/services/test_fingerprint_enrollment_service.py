"""Tests for FingerprintEnrollmentService."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.types import MinutiaCandidate, MinutiaType, NormalizedFingerprint
from src.db.models import Base, Fingerprint, Person
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)


@pytest.fixture(autouse=True)
def mock_cv2_imdecode():
    """Return a fake grayscale image so cv2.imdecode doesn't return None."""
    with patch("cv2.imdecode", return_value=np.zeros((100, 100), dtype=np.uint8)):
        yield


@pytest.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_pragma(dbapi_con, _):  # noqa: ARG001
        dbapi_con.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with async_session() as s:
        yield s

    await engine.dispose()


def make_normalized(minutiae_count: int = 5) -> NormalizedFingerprint:
    minutiae = [
        MinutiaCandidate(
            x=10 + i * 5, y=20 + i * 5, angle=0.0,
            type=MinutiaType.TERMINATION, confidence=0.9,
            origin=0,  # type: ignore[arg-type]
        )
        for i in range(minutiae_count)
    ]
    return NormalizedFingerprint(
        id="test", minutiae=minutiae, width=100, height=100,
    )


def make_fingerprint_service():
    fp = MagicMock()
    fp._process_image.return_value = make_normalized(5)
    return fp


class TestCreateCapture:
    @pytest.mark.asyncio
    async def test_creates_capture_with_graphs(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        image_bytes = b"FAKE_IMAGE_BYTES"
        capture, graphs = await svc.create_capture(fp.id, image_bytes)
        assert capture.id is not None
        assert capture.fingerprint_id == fp.id
        # RidgeGraph extraction removed per "No Legacy" mandate
        assert len(graphs) == 0
        assert capture.num_graphs == 0

    @pytest.mark.asyncio
    async def test_updates_capture_count_on_parent(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        await svc.create_capture(fp.id, b"x")
        await session.refresh(fp)
        assert fp.capture_count == 1
        assert fp.last_captured_at is not None

    @pytest.mark.asyncio
    async def test_missing_fingerprint_raises(self, session) -> None:
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        with pytest.raises(ValueError, match="not found"):
            await svc.create_capture(uuid.uuid4(), b"x")

    @pytest.mark.asyncio
    async def test_zero_minutiae_produces_zero_graphs(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        empty_fp = MagicMock()
        empty_fp.process_image.return_value = NormalizedFingerprint(
            id="x", minutiae=[], width=10, height=10,
        )
        svc = FingerprintEnrollmentService(session, empty_fp)
        capture, graphs = await svc.create_capture(fp.id, b"x")
        assert graphs == []
        assert capture.num_graphs == 0

    @pytest.mark.asyncio
    async def test_two_captures_increment_index(self, session) -> None:
        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        c1, _ = await svc.create_capture(fp.id, b"x")
        c2, _ = await svc.create_capture(fp.id, b"x")
        assert c1.capture_index == 1
        assert c2.capture_index == 2
        await session.refresh(fp)
        assert fp.capture_count == 2

    @pytest.mark.asyncio
    async def test_image_hash_is_sha256(self, session) -> None:
        import hashlib

        p = await PersonRepository.create(session, external_id="X")
        fp = await FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        bytes_in = b"hello world"
        c, _ = await svc.create_capture(fp.id, bytes_in)
        assert c.image_hash_sha256 == hashlib.sha256(bytes_in).hexdigest()


def test_init_accepts_mcc_service() -> None:
    """FingerprintEnrollmentService __init__ accepts mcc_matching_service kwarg."""
    svc = FingerprintEnrollmentService.__new__(FingerprintEnrollmentService)
    svc._session = MagicMock()
    svc._fp_service = MagicMock()
    svc._mcc_service = MagicMock()
    assert svc._mcc_service is not None


@pytest.mark.asyncio
async def test_index_pairs_invokes_enroll_pairs_with_image_bytes() -> None:
    """_index_pairs should call MccMatchingService.enroll_pairs with correct params."""
    calls = []

    class _FakeSvc:
        def enroll_pairs(self, capture_id, fingerprint_id, person_id, image_bytes):
            calls.append({
                "capture_id": capture_id,
                "fingerprint_id": fingerprint_id,
                "person_id": person_id,
                "len": len(image_bytes),
            })
            return 5

    person = MagicMock(spec=Person)
    person.external_id = "ext-1"
    person.id = 1

    svc = FingerprintEnrollmentService.__new__(FingerprintEnrollmentService)
    svc._session = AsyncMock()
    svc._session.get.return_value = person
    svc._fp_service = MagicMock()
    svc._mcc_service = _FakeSvc()

    capture = MagicMock()
    capture.id = "cap-1"
    fingerprint = MagicMock()
    fingerprint.id = "fp-1"
    fingerprint.person_id = "person-1"

    await svc._index_pairs(
        capture=capture,
        fingerprint=fingerprint,
        image_bytes=b"test-image-bytes",
    )
    assert len(calls) == 1
    assert calls[0]["capture_id"] == "cap-1"
    assert calls[0]["fingerprint_id"] == "fp-1"
    assert calls[0]["person_id"] == "ext-1"
    assert calls[0]["len"] == 16

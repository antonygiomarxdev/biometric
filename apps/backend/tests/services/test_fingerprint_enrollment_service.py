"""Tests for FingerprintEnrollmentService."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

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
def session():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _fk_pragma(dbapi_con, _):  # noqa: ARG001
        dbapi_con.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()
    engine.dispose()


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
    fp.process_image.return_value = make_normalized(5)
    return fp


class TestCreateCapture:
    def test_creates_capture_with_graphs(self, session) -> None:
        p = PersonRepository.create(session, external_id="X")
        fp = FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service(), qdrant_repo=None)
        image_bytes = b"FAKE_IMAGE_BYTES"
        capture, graphs = svc.create_capture(fp.id, image_bytes)
        assert capture.id is not None
        assert capture.fingerprint_id == fp.id
        assert len(graphs) >= 1
        assert capture.num_graphs == len(graphs)

    def test_updates_capture_count_on_parent(self, session) -> None:
        p = PersonRepository.create(session, external_id="X")
        fp = FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        svc.create_capture(fp.id, b"x")
        session.refresh(fp)
        assert fp.capture_count == 1
        assert fp.last_captured_at is not None

    def test_missing_fingerprint_raises(self, session) -> None:
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        with pytest.raises(ValueError, match="not found"):
            svc.create_capture(uuid.uuid4(), b"x")

    def test_zero_minutiae_produces_zero_graphs(self, session) -> None:
        p = PersonRepository.create(session, external_id="X")
        fp = FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        empty_fp = MagicMock()
        empty_fp.process_image.return_value = NormalizedFingerprint(
            id="x", minutiae=[], width=10, height=10,
        )
        svc = FingerprintEnrollmentService(session, empty_fp)
        capture, graphs = svc.create_capture(fp.id, b"x")
        assert graphs == []
        assert capture.num_graphs == 0

    def test_two_captures_increment_index(self, session) -> None:
        p = PersonRepository.create(session, external_id="X")
        fp = FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        c1, _ = svc.create_capture(fp.id, b"x")
        c2, _ = svc.create_capture(fp.id, b"x")
        assert c1.capture_index == 1
        assert c2.capture_index == 2
        session.refresh(fp)
        assert fp.capture_count == 2

    def test_image_hash_is_sha256(self, session) -> None:
        import hashlib

        p = PersonRepository.create(session, external_id="X")
        fp = FingerprintRepository.create(session, person_id=p.id, finger_position=2)
        svc = FingerprintEnrollmentService(session, make_fingerprint_service())
        bytes_in = b"hello world"
        c, _ = svc.create_capture(fp.id, bytes_in)
        assert c.image_hash_sha256 == hashlib.sha256(bytes_in).hexdigest()

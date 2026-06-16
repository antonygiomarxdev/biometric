"""Tests for FingerprintCaptureRepository (Phase 17)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Fingerprint, Person
from src.db.repositories.fingerprint_capture_repository import FingerprintCaptureRepository
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository


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


@pytest.fixture
def person(session) -> Person:
    return PersonRepository.create(session, external_id="X")


@pytest.fixture
def fingerprint(session, person) -> Fingerprint:
    return FingerprintRepository.create(
        session, person_id=person.id, finger_position=2,
    )


class TestFingerprintCaptureRepository:
    def test_create_assigns_capture_index_1(self, session, fingerprint) -> None:
        c = FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        assert c.capture_index == 1

    def test_create_second_capture_assigns_index_2(self, session, fingerprint) -> None:
        FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        c2 = FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://2.png", image_hash_sha256="h2",
        )
        assert c2.capture_index == 2

    def test_get_by_id(self, session, fingerprint) -> None:
        c = FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        found = FingerprintCaptureRepository.get_by_id(session, c.id)
        assert found is not None
        assert found.id == c.id

    def test_get_by_id_missing(self, session) -> None:
        import uuid
        assert FingerprintCaptureRepository.get_by_id(session, uuid.uuid4()) is None

    def test_list_by_fingerprint_ordered(self, session, fingerprint) -> None:
        for i in range(3):
            FingerprintCaptureRepository.create(
                session, fingerprint_id=fingerprint.id,
                image_uri=f"minio://{i}.png", image_hash_sha256=f"h{i}",
            )
        items = FingerprintCaptureRepository.list_by_fingerprint(session, fingerprint.id)
        assert len(items) == 3
        assert items[0].capture_index == 1
        assert items[2].capture_index == 3

    def test_find_by_image_hash(self, session, fingerprint) -> None:
        FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="unique_hash",
        )
        found = FingerprintCaptureRepository.find_by_image_hash(session, "unique_hash")
        assert found is not None

    def test_update_is_reference_flag(self, session, fingerprint) -> None:
        c = FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        updated = FingerprintCaptureRepository.update(session, c.id, is_reference=True)
        assert updated is not None
        assert updated.is_reference is True

    def test_count_by_fingerprint(self, session, fingerprint) -> None:
        assert FingerprintCaptureRepository.count_by_fingerprint(session, fingerprint.id) == 0
        FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        assert FingerprintCaptureRepository.count_by_fingerprint(session, fingerprint.id) == 1

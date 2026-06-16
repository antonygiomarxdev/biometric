"""Tests for FingerprintRepository (Phase 17)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Fingerprint, Person
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


class TestFingerprintRepository:
    def test_create_slot(self, session, person) -> None:
        f = FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert f.id is not None
        assert f.person_id == person.id

    def test_list_by_person_returns_in_order(self, session, person) -> None:
        FingerprintRepository.create(session, person_id=person.id, finger_position=10)
        FingerprintRepository.create(session, person_id=person.id, finger_position=2)
        items = FingerprintRepository.list_by_person(session, person.id)
        assert len(items) == 2
        assert items[0].finger_position == 2
        assert items[1].finger_position == 10

    def test_find_slot_returns_existing(self, session, person) -> None:
        FingerprintRepository.create(
            session, person_id=person.id, finger_position=3, capture_type="latent",
        )
        found = FingerprintRepository.find_slot(
            session, person.id, 3, "latent",
        )
        assert found is not None

    def test_find_slot_returns_none_for_missing(self, session, person) -> None:
        found = FingerprintRepository.find_slot(
            session, person.id, 99, "rolled",
        )
        assert found is None

    def test_increment_capture_count(self, session, person) -> None:
        f = FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert f.capture_count == 0
        FingerprintRepository.increment_capture_count(session, f.id)
        assert FingerprintRepository.get_by_id(session, f.id).capture_count == 1

    def test_increment_capture_count_sets_first_captured_at(self, session, person) -> None:
        f = FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert f.first_captured_at is None
        FingerprintRepository.increment_capture_count(session, f.id)
        session.refresh(f)
        assert f.first_captured_at is not None
        assert f.last_captured_at is not None

    def test_create_duplicate_slot_raises(self, session, person) -> None:
        FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        with pytest.raises(Exception):
            FingerprintRepository.create(
                session, person_id=person.id, finger_position=2,
            )

    def test_delete(self, session, person) -> None:
        f = FingerprintRepository.create(
            session, person_id=person.id, finger_position=2,
        )
        assert FingerprintRepository.delete(session, f.id) is True
        assert FingerprintRepository.get_by_id(session, f.id) is None

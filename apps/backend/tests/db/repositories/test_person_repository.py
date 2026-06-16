"""Tests for PersonRepository (Phase 17)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Person
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


class TestPersonRepository:
    def test_create_minimal(self, session) -> None:
        p = PersonRepository.create(session, external_id="X")
        assert p.id is not None
        assert p.external_id == "X"

    def test_create_with_all_fields(self, session) -> None:
        p = PersonRepository.create(
            session,
            external_id="X", full_name="Juan", doc_type="cedula",
            doc_number="001", sex="M",
        )
        assert p.full_name == "Juan"
        assert p.doc_type == "cedula"

    def test_get_by_id_returns_none_for_missing(self, session) -> None:
        import uuid
        assert PersonRepository.get_by_id(session, uuid.uuid4()) is None

    def test_find_by_external_id(self, session) -> None:
        PersonRepository.create(session, external_id="ABC")
        found = PersonRepository.find_by_external_id(session, "ABC")
        assert found is not None
        assert found.external_id == "ABC"

    def test_find_by_external_id_missing(self, session) -> None:
        assert PersonRepository.find_by_external_id(session, "NOPE") is None

    def test_list_with_pagination(self, session) -> None:
        for i in range(5):
            PersonRepository.create(session, external_id=f"P{i}")
        assert len(PersonRepository.list(session, skip=0, limit=3)) == 3
        assert len(PersonRepository.list(session, skip=3, limit=10)) == 2

    def test_list_with_search_filter(self, session) -> None:
        PersonRepository.create(session, external_id="A", full_name="Juan Pérez")
        PersonRepository.create(session, external_id="B", full_name="Pedro Gómez")
        results = PersonRepository.list(session, search="juan")
        assert len(results) == 1
        assert results[0].full_name == "Juan Pérez"

    def test_update_fields(self, session) -> None:
        p = PersonRepository.create(session, external_id="X", full_name="Old")
        updated = PersonRepository.update(session, p.id, full_name="New")
        assert updated is not None
        assert updated.full_name == "New"

    def test_delete_returns_false_for_missing(self, session) -> None:
        import uuid
        assert PersonRepository.delete(session, uuid.uuid4()) is False

    def test_delete_success(self, session) -> None:
        p = PersonRepository.create(session, external_id="X")
        assert PersonRepository.delete(session, p.id) is True
        assert PersonRepository.get_by_id(session, p.id) is None

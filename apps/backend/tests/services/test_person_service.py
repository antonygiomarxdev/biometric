"""Tests for PersonService."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base
from src.schemas.person_schema import PersonCreate
from src.services.person_service import PersonService


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()
    engine.dispose()


@pytest.fixture
def service(session: object) -> PersonService:
    return PersonService(session)


class TestCreatePerson:
    def test_creates_with_minimal_data(self, service: PersonService) -> None:
        p = service.create_person(PersonCreate(external_id="X"))
        assert p.id is not None
        assert p.external_id == "X"

    def test_duplicate_external_id_raises(self, service: PersonService) -> None:
        service.create_person(PersonCreate(external_id="X"))
        with pytest.raises(ValueError, match="already exists"):
            service.create_person(PersonCreate(external_id="X"))

    def test_normalizes_doc_type(self, service: PersonService) -> None:
        p = service.create_person(PersonCreate(external_id="X", doc_type="CEDULA"))
        assert p.doc_type == "cedula"

    def test_unknown_doc_type_kept_as_is(self, service: PersonService) -> None:
        p = service.create_person(PersonCreate(external_id="X", doc_type="custom"))
        assert p.doc_type == "custom"

    def test_sex_uppercased(self, service: PersonService) -> None:
        p = service.create_person(PersonCreate(external_id="X", sex="m"))
        assert p.sex == "M"


class TestListAndGet:
    def test_list_empty(self, service: PersonService) -> None:
        assert service.list_persons() == []

    def test_list_returns_persons(self, service: PersonService) -> None:
        service.create_person(PersonCreate(external_id="A"))
        service.create_person(PersonCreate(external_id="B"))
        assert len(service.list_persons()) == 2

    def test_search_filter(self, service: PersonService) -> None:
        service.create_person(PersonCreate(external_id="A", full_name="Juan"))
        service.create_person(PersonCreate(external_id="B", full_name="Pedro"))
        results = service.list_persons(search="juan")
        assert len(results) == 1
        assert results[0].full_name == "Juan"

    def test_get_person_by_id(self, service: PersonService) -> None:
        p = service.create_person(PersonCreate(external_id="X"))
        found = service.get_person(p.id)
        assert found is not None
        assert found.id == p.id

    def test_get_person_not_found(self, service: PersonService) -> None:
        import uuid
        assert service.get_person(uuid.uuid4()) is None


class TestFindOrCreate:
    def test_returns_existing(self, service: PersonService) -> None:
        p1 = service.find_or_create_person("X")
        p2 = service.find_or_create_person("X")
        assert p1.id == p2.id

    def test_creates_when_missing(self, service: PersonService) -> None:
        p = service.find_or_create_person("NEW", full_name="Test")
        assert p.full_name == "Test"


class TestDeletePerson:
    def test_delete(self, service: PersonService) -> None:
        p = service.create_person(PersonCreate(external_id="X"))
        assert service.delete_person(p.id) is True
        assert service.get_person(p.id) is None

    def test_delete_missing(self, service: PersonService) -> None:
        import uuid
        assert service.delete_person(uuid.uuid4()) is False

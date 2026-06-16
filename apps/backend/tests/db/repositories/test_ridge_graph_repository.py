"""Tests for RidgeGraphRepository (Phase 17)."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, Fingerprint, Person, RidgeGraph
from src.db.repositories.fingerprint_capture_repository import FingerprintCaptureRepository
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.db.repositories.ridge_graph_repository import RidgeGraphRepository


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
def capture(session):
    p = PersonRepository.create(session, external_id="X")
    f = FingerprintRepository.create(session, person_id=p.id, finger_position=2)
    c = FingerprintCaptureRepository.create(
        session, fingerprint_id=f.id,
        image_uri="minio://1.png", image_hash_sha256="h",
    )
    return c


class TestRidgeGraphRepository:
    def test_create_graph(self, session, capture) -> None:
        g = RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_index=1,
            num_nodes=5, num_edges=4, graph_data={"k": "v"},
        )
        assert g.id is not None
        assert g.graph_data == {"k": "v"}

    def test_create_with_singularity(self, session, capture) -> None:
        g = RidgeGraphRepository.create(
            session, capture_id=capture.id,
            num_nodes=10, graph_data={},
            core_x=100, core_y=200, singularity_type="core",
        )
        assert g.core_x == 100
        assert g.singularity_type == "core"

    def test_get_by_id(self, session, capture) -> None:
        g = RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_data={},
        )
        found = RidgeGraphRepository.get_by_id(session, g.id)
        assert found is not None

    def test_get_by_id_missing(self, session) -> None:
        import uuid
        assert RidgeGraphRepository.get_by_id(session, uuid.uuid4()) is None

    def test_list_by_capture_ordered(self, session, capture) -> None:
        for i in range(3):
            RidgeGraphRepository.create(
                session, capture_id=capture.id, graph_index=i + 1, graph_data={},
            )
        items = RidgeGraphRepository.list_by_capture(session, capture.id)
        assert len(items) == 3
        assert items[0].graph_index == 1
        assert items[2].graph_index == 3

    def test_count_by_capture(self, session, capture) -> None:
        assert RidgeGraphRepository.count_by_capture(session, capture.id) == 0
        RidgeGraphRepository.create(session, capture_id=capture.id, graph_data={})
        assert RidgeGraphRepository.count_by_capture(session, capture.id) == 1

    def test_delete(self, session, capture) -> None:
        g = RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_data={},
        )
        assert RidgeGraphRepository.delete(session, g.id) is True
        assert RidgeGraphRepository.get_by_id(session, g.id) is None

    def test_delete_missing(self, session) -> None:
        import uuid
        assert RidgeGraphRepository.delete(session, uuid.uuid4()) is False

    def test_graph_data_jsonb_round_trip(self, session, capture) -> None:
        data = {"nodes": [{"x": 1, "y": 2}], "edges": []}
        g = RidgeGraphRepository.create(
            session, capture_id=capture.id, graph_data=data,
        )
        assert g.graph_data == data
        fetched = RidgeGraphRepository.get_by_id(session, g.id)
        assert fetched is not None
        assert fetched.graph_data == data

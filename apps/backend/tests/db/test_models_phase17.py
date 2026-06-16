"""Tests for the 4-level forensic data model (Phase 17)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.db.models import (
    Base,
    Evidence,
    Fingerprint,
    FingerprintCapture,
    Person,
    RidgeGraph,
)


@pytest.fixture
def session():
    """In-memory SQLite session with FK enforcement."""
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


class TestPerson:
    def test_create_minimal(self, session) -> None:
        p = Person(external_id="001-ABC")
        session.add(p)
        session.commit()
        assert p.id is not None
        assert p.external_id == "001-ABC"
        assert p.created_at is not None

    def test_create_full(self, session) -> None:
        p = Person(
            external_id="X",
            full_name="Juan Pérez",
            doc_type="cedula",
            doc_number="001-ABC",
            sex="M",
        )
        session.add(p)
        session.commit()
        assert p.full_name == "Juan Pérez"

    def test_external_id_unique(self, session) -> None:
        p1 = Person(external_id="X")
        p2 = Person(external_id="X")
        session.add(p1)
        session.commit()
        session.add(p2)
        with pytest.raises(Exception):
            session.commit()


class TestFingerprint:
    def test_create(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f = Fingerprint(person_id=p.id, finger_position=2, capture_type="rolled")
        session.add(f)
        session.commit()
        assert f.id is not None
        assert f.capture_count == 0

    def test_unique_slot(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f1 = Fingerprint(person_id=p.id, finger_position=2, capture_type="rolled")
        f2 = Fingerprint(person_id=p.id, finger_position=2, capture_type="rolled")
        session.add(f1)
        session.commit()
        session.add(f2)
        with pytest.raises(Exception):
            session.commit()

    def test_different_capture_type_allowed(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f1 = Fingerprint(person_id=p.id, finger_position=2, capture_type="rolled")
        f2 = Fingerprint(person_id=p.id, finger_position=2, capture_type="latent")
        session.add(f1)
        session.add(f2)
        session.commit()
        assert f1.id != f2.id


class TestFingerprintCapture:
    def test_create(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f = Fingerprint(person_id=p.id, finger_position=2)
        session.add(f)
        session.commit()
        c = FingerprintCapture(
            fingerprint_id=f.id,
            image_uri="minio://test/1.png",
            image_hash_sha256="abc123",
        )
        session.add(c)
        session.commit()
        assert c.id is not None
        assert c.capture_index == 1
        assert c.algorithm_version == "phase-13-v1"
        assert c.is_exemplar is True
        assert c.is_reference is False


class TestRidgeGraph:
    def test_create(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f = Fingerprint(person_id=p.id, finger_position=2)
        session.add(f)
        session.commit()
        c = FingerprintCapture(
            fingerprint_id=f.id, image_uri="x", image_hash_sha256="h",
        )
        session.add(c)
        session.commit()
        g = RidgeGraph(
            capture_id=c.id,
            region_x=10, region_y=20, region_w=300, region_h=400,
            num_nodes=20, num_edges=25,
            graph_data={"nodes": [], "edges": []},
            core_x=150, core_y=200, singularity_type="core",
        )
        session.add(g)
        session.commit()
        assert g.id is not None
        assert g.graph_data == {"nodes": [], "edges": []}


class TestEvidenceMatchedColumns:
    def test_matched_columns_nullable(self, session) -> None:
        e = Evidence(case_id=uuid.uuid4(), fingerprint_id="X")
        assert hasattr(e, "matched_fingerprint_id")
        assert hasattr(e, "matched_person_id")
        assert e.matched_fingerprint_id is None
        assert e.matched_person_id is None


class TestCascadeDelete:
    def test_delete_person_cascades_to_fingerprints(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f = Fingerprint(person_id=p.id, finger_position=2)
        session.add(f)
        session.commit()
        session.delete(p)
        session.commit()
        assert session.query(Fingerprint).count() == 0

    def test_delete_fingerprint_cascades_to_captures(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f = Fingerprint(person_id=p.id, finger_position=2)
        session.add(f)
        session.commit()
        c = FingerprintCapture(
            fingerprint_id=f.id, image_uri="x", image_hash_sha256="h",
        )
        session.add(c)
        session.commit()
        session.delete(f)
        session.commit()
        assert session.query(FingerprintCapture).count() == 0

    def test_delete_capture_cascades_to_graphs(self, session) -> None:
        p = Person()
        session.add(p)
        session.commit()
        f = Fingerprint(person_id=p.id, finger_position=2)
        session.add(f)
        session.commit()
        c = FingerprintCapture(
            fingerprint_id=f.id, image_uri="x", image_hash_sha256="h",
        )
        session.add(c)
        session.commit()
        g = RidgeGraph(capture_id=c.id, graph_data={})
        session.add(g)
        session.commit()
        session.delete(c)
        session.commit()
        assert session.query(RidgeGraph).count() == 0

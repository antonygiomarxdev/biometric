"""Tests for FingerprintCaptureRepository (Phase 17, async)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Fingerprint, Person
from src.db.repositories.fingerprint_capture_repository import FingerprintCaptureRepository
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository


@pytest.fixture
async def session():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _fk_pragma(dbapi_con, _):  # noqa: ARG001
        dbapi_con.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    Session = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    s = Session()
    yield s
    await s.close()
    await engine.dispose()


@pytest.fixture
async def person(session) -> Person:
    return await PersonRepository.create(session, external_id="X")


@pytest.fixture
async def fingerprint(session, person) -> Fingerprint:
    return await FingerprintRepository.create(
        session, person_id=person.id, finger_position=2,
    )


@pytest.mark.asyncio
class TestFingerprintCaptureRepository:
    async def test_create_assigns_capture_index_1(self, session, fingerprint) -> None:
        c = await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        assert c.capture_index == 1

    async def test_create_second_capture_assigns_index_2(self, session, fingerprint) -> None:
        await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        c2 = await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://2.png", image_hash_sha256="h2",
        )
        assert c2.capture_index == 2

    async def test_get_by_id(self, session, fingerprint) -> None:
        c = await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        found = await FingerprintCaptureRepository.get_by_id(session, c.id)
        assert found is not None
        assert found.id == c.id

    async def test_get_by_id_missing(self, session) -> None:
        assert await FingerprintCaptureRepository.get_by_id(session, uuid.uuid4()) is None

    async def test_list_by_fingerprint_ordered(self, session, fingerprint) -> None:
        for i in range(3):
            await FingerprintCaptureRepository.create(
                session, fingerprint_id=fingerprint.id,
                image_uri=f"minio://{i}.png", image_hash_sha256=f"h{i}",
            )
        items = await FingerprintCaptureRepository.list_by_fingerprint(session, fingerprint.id)
        assert len(items) == 3
        assert items[0].capture_index == 1
        assert items[2].capture_index == 3

    async def test_find_by_image_hash(self, session, fingerprint) -> None:
        await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="unique_hash",
        )
        found = await FingerprintCaptureRepository.find_by_image_hash(session, "unique_hash")
        assert found is not None

    async def test_update_is_reference_flag(self, session, fingerprint) -> None:
        c = await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        updated = await FingerprintCaptureRepository.update(session, c.id, is_reference=True)
        assert updated is not None
        assert updated.is_reference is True

    async def test_count_by_fingerprint(self, session, fingerprint) -> None:
        assert await FingerprintCaptureRepository.count_by_fingerprint(session, fingerprint.id) == 0
        await FingerprintCaptureRepository.create(
            session, fingerprint_id=fingerprint.id,
            image_uri="minio://1.png", image_hash_sha256="h1",
        )
        assert await FingerprintCaptureRepository.count_by_fingerprint(session, fingerprint.id) == 1

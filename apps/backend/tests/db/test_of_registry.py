"""Integration tests for OFRegistry using SQLite in-memory DB.

No PostgreSQL required — SQLAlchemy handles dialect differences via
the ORM layer. Requires aiosqlite.
"""

from __future__ import annotations

import uuid
from typing import Any, AsyncGenerator


import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.db.models import Base
from src.db.of_registry import OFRegistry
from src.processing.of_similarity import OFSimilarity


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with Session_factory() as s:
        yield s
    await engine.dispose()


@pytest.fixture
def registry(session: AsyncSession) -> OFRegistry:
    return OFRegistry(session)


@pytest.fixture
def sample_of() -> OFSimilarity:
    ori = np.zeros((4, 4), dtype=np.float32)
    coh = np.ones((4, 4), dtype=np.float32) * 0.8
    return OFSimilarity(ori, coh)


@pytest.fixture
def fid() -> str:
    return str(uuid.uuid4())


@pytest.mark.asyncio
async def test_save_and_get_roundtrip(
    registry: OFRegistry,
    sample_of: OFSimilarity,
    fid: str,
) -> None:
    await registry.save(fid, sample_of)
    record = await registry.get(fid)
    assert record is not None
    assert record["fingerprint_id"] == fid
    assert len(record["of_ori"]) == 4
    assert record["block_size"] == 16
    assert record["pseudo_core"] is not None


@pytest.mark.asyncio
async def test_get_nonexistent_returns_none(
    registry: OFRegistry,
) -> None:
    record = await registry.get(str(uuid.uuid4()))
    assert record is None


@pytest.mark.asyncio
async def test_get_many_returns_dict(
    registry: OFRegistry,
    sample_of: OFSimilarity,
) -> None:
    fid_a = str(uuid.uuid4())
    fid_b = str(uuid.uuid4())
    await registry.save(fid_a, sample_of)
    await registry.save(fid_b, sample_of)

    records = await registry.get_many([fid_a, fid_b])
    assert len(records) == 2
    assert fid_a in records
    assert fid_b in records


@pytest.mark.asyncio
async def test_upsert_overwrites_existing(
    registry: OFRegistry,
    sample_of: OFSimilarity,
    fid: str,
) -> None:
    await registry.save(fid, sample_of)
    record1 = await registry.get(fid)
    assert record1 is not None

    # Save with a different OF
    ori2 = np.ones((4, 4), dtype=np.float32) * np.pi / 2
    coh2 = np.ones((4, 4), dtype=np.float32) * 0.5
    of2 = OFSimilarity(ori2, coh2)
    await registry.save(fid, of2)
    record2 = await registry.get(fid)
    assert record2 is not None
    assert record2["fingerprint_id"] == fid


@pytest.mark.asyncio
async def test_delete_removes_record(
    registry: OFRegistry,
    sample_of: OFSimilarity,
    fid: str,
) -> None:
    await registry.save(fid, sample_of)
    assert await registry.get(fid) is not None
    await registry.delete(fid)
    assert await registry.get(fid) is None

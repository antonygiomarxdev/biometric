"""Async repository for :class:`~src.db.models.FingerprintCapture`."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import FingerprintCapture


def _fingerprint_lock_key(fingerprint_id: "uuid.UUID") -> int:
    """Map a fingerprint UUID to a 32-bit advisory lock key.

    We only need a stable per-fingerprint key — collisions across
    different fingerprints do not cause incorrect behaviour, just
    slightly more contention.  SHA-256 truncated to 8 bytes fits
    PostgreSQL's ``bigint`` advisory lock range when interpreted as
    signed.
    """
    raw = hashlib.sha256(str(fingerprint_id).encode()).digest()[:8]
    return int.from_bytes(raw, "big", signed=True)


class FingerprintCaptureRepository:
    """Async persistence gateway for the ``fingerprint_captures`` table.

    Concurrency: ``create`` is the only write that needs cross-request
    coordination.  We acquire a per-fingerprint PostgreSQL advisory
    lock (``pg_advisory_xact_lock``) so concurrent enrollments against
    the same fingerprint slot serialise; enrollments against
    *different* fingerprints proceed in parallel.

    Idempotency: the UNIQUE constraint on
    ``(fingerprint_id, image_hash_sha256)`` (migration 0011) makes
    ``create`` safe to retry.  If a request is replayed (network
    retry, batch-script resume), the second attempt detects the
    existing row via ``ON CONFLICT DO NOTHING`` and returns the
    original capture — no duplicate row, no duplicate Qdrant point
    (Qdrant is also idempotent on point ID).
    """

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        fingerprint_id: uuid.UUID,
        image_uri: str,
        image_hash_sha256: str,
        algorithm_version: str = "afrnet-v1",
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> tuple[FingerprintCapture, bool]:
        """Create a new capture, or return the existing one.

        Returns ``(capture, created)`` where ``created`` is
        ``True`` if a new row was inserted and ``False`` if the
        capture already existed (idempotent replay).

        Concurrency: per-fingerprint advisory lock on PostgreSQL,
        single-writer on SQLite, both backed by the UNIQUE
        constraint + ``ON CONFLICT DO NOTHING``.
        """
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            lock_key = _fingerprint_lock_key(fingerprint_id)
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": lock_key},
            )

        existing = await FingerprintCaptureRepository.find_by_fingerprint_and_hash(
            session, fingerprint_id, image_hash_sha256,
        )
        if existing is not None:
            return existing, False

        existing_count = await FingerprintCaptureRepository.count_by_fingerprint(
            session, fingerprint_id,
        )
        stmt = (
            pg_insert(FingerprintCapture)
            .values(
                fingerprint_id=fingerprint_id,
                capture_index=existing_count + 1,
                image_uri=image_uri,
                image_hash_sha256=image_hash_sha256,
                algorithm_version=algorithm_version,
                is_reference=is_reference,
                is_exemplar=is_exemplar,
                notes=notes,
            )
            .on_conflict_do_nothing(
                index_elements=["fingerprint_id", "image_hash_sha256"],
            )
            .returning(FingerprintCapture)
        )
        result = await session.execute(stmt)
        c = result.scalar_one_or_none()
        await session.commit()
        if c is not None:
            return c, True

        existing = await FingerprintCaptureRepository.find_by_fingerprint_and_hash(
            session, fingerprint_id, image_hash_sha256,
        )
        if existing is None:
            msg = (
                "Insert returned no row but the capture is not "
                "queryable — inconsistent state"
            )
            raise RuntimeError(msg)
        return existing, False

    @staticmethod
    async def find_by_fingerprint_and_hash(
        session: AsyncSession,
        fingerprint_id: uuid.UUID,
        image_hash_sha256: str,
    ) -> FingerprintCapture | None:
        stmt = select(FingerprintCapture).where(
            FingerprintCapture.fingerprint_id == fingerprint_id,
            FingerprintCapture.image_hash_sha256 == image_hash_sha256,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def count_by_fingerprint(
        session: AsyncSession,
        fingerprint_id: uuid.UUID,
    ) -> int:
        stmt = select(func.count()).select_from(FingerprintCapture).where(
            FingerprintCapture.fingerprint_id == fingerprint_id
        )
        result = await session.execute(stmt)
        return int(result.scalar_one())

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        capture_id: uuid.UUID,
    ) -> FingerprintCapture | None:
        return await session.get(FingerprintCapture, capture_id)

    @staticmethod
    async def list_by_fingerprint(
        session: AsyncSession,
        fingerprint_id: uuid.UUID,
    ) -> list[FingerprintCapture]:
        stmt = (
            select(FingerprintCapture)
            .where(FingerprintCapture.fingerprint_id == fingerprint_id)
            .order_by(FingerprintCapture.capture_index)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def find_by_image_hash(
        session: AsyncSession,
        image_hash_sha256: str,
    ) -> FingerprintCapture | None:
        stmt = select(FingerprintCapture).where(
            FingerprintCapture.image_hash_sha256 == image_hash_sha256
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    @staticmethod
    async def update(
        session: AsyncSession,
        capture_id: uuid.UUID,
        **fields: object,
    ) -> FingerprintCapture | None:
        c = await session.get(FingerprintCapture, capture_id)
        if c is None:
            return None
        for key, value in fields.items():
            if hasattr(c, key):
                setattr(c, key, value)
        await session.commit()
        await session.refresh(c)
        return c

    @staticmethod
    async def delete(session: AsyncSession, capture_id: uuid.UUID) -> bool:
        c = await session.get(FingerprintCapture, capture_id)
        if c is None:
            return False
        await session.delete(c)
        await session.commit()
        return True

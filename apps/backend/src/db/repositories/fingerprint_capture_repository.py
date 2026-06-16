"""Async repository for :class:`~src.db.models.FingerprintCapture`."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import FingerprintCapture


class FingerprintCaptureRepository:
    """Async persistence gateway for the ``fingerprint_captures`` table."""

    @staticmethod
    async def create(
        session: AsyncSession,
        *,
        fingerprint_id: uuid.UUID,
        image_uri: str,
        image_hash_sha256: str,
        image_dpi: int | None = None,
        image_quality_score: float | None = None,
        algorithm_version: str = "phase-13-v1",
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> FingerprintCapture:
        existing_count = await FingerprintCaptureRepository.count_by_fingerprint(
            session, fingerprint_id,
        )
        c = FingerprintCapture(
            fingerprint_id=fingerprint_id,
            capture_index=existing_count + 1,
            image_uri=image_uri,
            image_hash_sha256=image_hash_sha256,
            image_dpi=image_dpi,
            image_quality_score=image_quality_score,
            algorithm_version=algorithm_version,
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )
        session.add(c)
        await session.commit()
        await session.refresh(c)
        return c

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

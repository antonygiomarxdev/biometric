"""Repository for :class:`~src.db.models.FingerprintCapture`."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.db.models import FingerprintCapture


class FingerprintCaptureRepository:
    """Persistence gateway for the ``fingerprint_captures`` table."""

    @staticmethod
    def create(
        session: Session,
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
        existing_count = FingerprintCaptureRepository.count_by_fingerprint(
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
        session.commit()
        session.refresh(c)
        return c

    @staticmethod
    def count_by_fingerprint(
        session: Session,
        fingerprint_id: uuid.UUID,
    ) -> int:
        stmt = select(func.count()).select_from(FingerprintCapture).where(
            FingerprintCapture.fingerprint_id == fingerprint_id
        )
        return int(session.execute(stmt).scalar_one())

    @staticmethod
    def get_by_id(
        session: Session,
        capture_id: uuid.UUID,
    ) -> FingerprintCapture | None:
        return session.get(FingerprintCapture, capture_id)

    @staticmethod
    def list_by_fingerprint(
        session: Session,
        fingerprint_id: uuid.UUID,
    ) -> list[FingerprintCapture]:
        stmt = (
            select(FingerprintCapture)
            .where(FingerprintCapture.fingerprint_id == fingerprint_id)
            .order_by(FingerprintCapture.capture_index)
        )
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    def find_by_image_hash(
        session: Session,
        image_hash_sha256: str,
    ) -> FingerprintCapture | None:
        stmt = select(FingerprintCapture).where(
            FingerprintCapture.image_hash_sha256 == image_hash_sha256
        )
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def update(
        session: Session,
        capture_id: uuid.UUID,
        **fields: object,
    ) -> FingerprintCapture | None:
        c = session.get(FingerprintCapture, capture_id)
        if c is None:
            return None
        for key, value in fields.items():
            if hasattr(c, key):
                setattr(c, key, value)
        session.commit()
        session.refresh(c)
        return c

    @staticmethod
    def delete(session: Session, capture_id: uuid.UUID) -> bool:
        c = session.get(FingerprintCapture, capture_id)
        if c is None:
            return False
        session.delete(c)
        session.commit()
        return True

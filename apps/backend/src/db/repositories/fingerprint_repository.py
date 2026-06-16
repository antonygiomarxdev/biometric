"""Repository for :class:`~src.db.models.Fingerprint`."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Fingerprint


class FingerprintRepository:
    """Persistence gateway for the ``fingerprints`` table."""

    @staticmethod
    def create(
        session: Session,
        *,
        person_id: uuid.UUID,
        finger_position: int,
        capture_type: str = "rolled",
        notes: str | None = None,
    ) -> Fingerprint:
        f = Fingerprint(
            person_id=person_id,
            finger_position=finger_position,
            capture_type=capture_type,
            notes=notes,
        )
        session.add(f)
        session.commit()
        session.refresh(f)
        return f

    @staticmethod
    def get_by_id(session: Session, fingerprint_id: uuid.UUID) -> Fingerprint | None:
        return session.get(Fingerprint, fingerprint_id)

    @staticmethod
    def list_by_person(
        session: Session,
        person_id: uuid.UUID,
    ) -> list[Fingerprint]:
        stmt = (
            select(Fingerprint)
            .where(Fingerprint.person_id == person_id)
            .order_by(Fingerprint.finger_position, Fingerprint.capture_type)
        )
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    def find_slot(
        session: Session,
        person_id: uuid.UUID,
        finger_position: int,
        capture_type: str,
    ) -> Fingerprint | None:
        stmt = select(Fingerprint).where(
            Fingerprint.person_id == person_id,
            Fingerprint.finger_position == finger_position,
            Fingerprint.capture_type == capture_type,
        )
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def increment_capture_count(
        session: Session,
        fingerprint_id: uuid.UUID,
    ) -> Fingerprint | None:
        f = session.get(Fingerprint, fingerprint_id)
        if f is None:
            return None
        f.capture_count = (f.capture_count or 0) + 1
        f.last_captured_at = datetime.now(timezone.utc)
        if f.first_captured_at is None:
            f.first_captured_at = f.last_captured_at
        session.commit()
        session.refresh(f)
        return f

    @staticmethod
    def delete(session: Session, fingerprint_id: uuid.UUID) -> bool:
        f = session.get(Fingerprint, fingerprint_id)
        if f is None:
            return False
        session.delete(f)
        session.commit()
        return True

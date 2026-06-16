"""Repository for :class:`~src.db.models.Person`."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db.models import Person


class PersonRepository:
    """Persistence gateway for the ``persons`` table."""

    @staticmethod
    def create(
        session: Session,
        *,
        external_id: str | None = None,
        full_name: str | None = None,
        doc_type: str | None = None,
        doc_number: str | None = None,
        sex: str | None = None,
        dob: datetime | None = None,
        notes: str | None = None,
    ) -> Person:
        p = Person(
            external_id=external_id, full_name=full_name,
            doc_type=doc_type, doc_number=doc_number,
            sex=sex, dob=dob, notes=notes,
        )
        session.add(p)
        session.commit()
        session.refresh(p)
        return p

    @staticmethod
    def get_by_id(session: Session, person_id: uuid.UUID) -> Person | None:
        return session.get(Person, person_id)

    @staticmethod
    def find_by_external_id(session: Session, external_id: str) -> Person | None:
        stmt = select(Person).where(Person.external_id == external_id)
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def list(
        session: Session,
        *,
        skip: int = 0,
        limit: int = 20,
        search: str | None = None,
    ) -> list[Person]:
        stmt = select(Person)
        if search is not None:
            like = f"%{search}%"
            stmt = stmt.where(
                Person.full_name.ilike(like) | Person.external_id.ilike(like)
            )
        stmt = stmt.order_by(Person.created_at.desc()).offset(skip).limit(limit)
        return list(session.execute(stmt).scalars().all())

    @staticmethod
    def update(
        session: Session,
        person_id: uuid.UUID,
        **fields: object,
    ) -> Person | None:
        p = session.get(Person, person_id)
        if p is None:
            return None
        for key, value in fields.items():
            if value is not None and hasattr(p, key):
                setattr(p, key, value)
        session.commit()
        session.refresh(p)
        return p

    @staticmethod
    def delete(session: Session, person_id: uuid.UUID) -> bool:
        p = session.get(Person, person_id)
        if p is None:
            return False
        session.delete(p)
        session.commit()
        return True

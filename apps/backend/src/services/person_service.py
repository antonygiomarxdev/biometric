"""Async PersonService — calls async PersonRepository directly."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.enums import DocumentType
from src.db.models import Person
from src.db.repositories.person_repository import PersonRepository
from src.schemas.person_schema import PersonCreate

log = logging.getLogger(__name__)


class PersonService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_person(self, data: PersonCreate) -> Person:
        if data.doc_type is not None:
            try:
                norm = DocumentType(data.doc_type.lower())
                data = data.model_copy(update={"doc_type": norm.value})
            except ValueError:
                pass
        ext_id = data.external_id
        if ext_id:
            existing = await PersonRepository.find_by_external_id(self._session, ext_id)
            if existing is not None:
                raise ValueError(
                    f"Person with external_id={data.external_id} already exists"
                )
        dob_dt = (
            datetime.combine(data.dob, datetime.min.time()).replace(tzinfo=timezone.utc)
            if data.dob
            else None
        )
        return await PersonRepository.create(
            self._session,
            external_id=data.external_id,
            full_name=data.full_name,
            doc_type=data.doc_type,
            doc_number=data.doc_number,
            sex=(data.sex or "").upper() or None,
            dob=dob_dt,
            notes=data.notes,
        )

    async def get_person(self, person_id: uuid.UUID) -> Person | None:
        return await PersonRepository.get_by_id(self._session, person_id)

    async def list_persons(
        self, *, skip: int = 0, limit: int = 20, search: str | None = None
    ) -> list[Person]:
        return await PersonRepository.list(
            self._session, skip=skip, limit=limit, search=search
        )

    async def update_person(
        self, person_id: uuid.UUID, **fields: Any
    ) -> Person | None:
        return await PersonRepository.update(self._session, person_id, **fields)

    async def find_or_create_person(
        self, external_id: str, **defaults: Any
    ) -> Person:
        existing = await PersonRepository.find_by_external_id(self._session, external_id)
        if existing is not None:
            return existing
        return await PersonRepository.create(self._session, external_id=external_id, **defaults)

    async def delete_person(self, person_id: uuid.UUID) -> bool:
        return await PersonRepository.delete(self._session, person_id)

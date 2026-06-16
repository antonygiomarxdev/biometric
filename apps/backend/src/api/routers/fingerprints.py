"""Fingerprints API — Phase 17."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.schemas.fingerprint_schema import (
    FingerprintCreate,
    FingerprintListResponse,
    FingerprintResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["fingerprints"])


@router.post(
    "/api/v1/persons/{person_id}/fingerprints",
    response_model=FingerprintResponse,
    status_code=201,
)
def create_fingerprint(
    person_id: uuid.UUID,
    data: FingerprintCreate,
    session: Session = Depends(get_db),
) -> Any:
    person = PersonRepository.get_by_id(session, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    existing = FingerprintRepository.find_slot(
        session, person_id, data.finger_position, data.capture_type,
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Slot for finger {data.finger_position}/{data.capture_type} already exists",
        )
    return FingerprintRepository.create(
        session,
        person_id=person_id,
        finger_position=data.finger_position,
        capture_type=data.capture_type,
        notes=data.notes,
    )


@router.get(
    "/api/v1/persons/{person_id}/fingerprints",
    response_model=FingerprintListResponse,
)
def list_fingerprints(
    person_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> Any:
    items = FingerprintRepository.list_by_person(session, person_id)
    return FingerprintListResponse(items=items, total=len(items))

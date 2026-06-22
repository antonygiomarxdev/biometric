"""Fingerprints API — Phase 17."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_async_db
from src.api.prefix import API_PREFIX
from src.db.models import Person
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.schemas.fingerprint_schema import FingerprintCreate, FingerprintListResponse, FingerprintResponse

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from starlette.responses import Response

log = logging.getLogger(__name__)

router = APIRouter(tags=["fingerprints"])


@router.post(
    API_PREFIX + "/persons/{person_id}/fingerprints",
    response_model=FingerprintResponse,
    status_code=201,
    summary="Create or fetch a fingerprint slot (idempotent)",
    responses={
        200: {"description": "Slot already existed, returned as-is"},
        201: {"description": "Slot created"},
        404: {"description": "Person not found"},
    },
)
async def create_fingerprint(
    person_id: uuid.UUID,
    data: FingerprintCreate,
    session: AsyncSession = Depends(get_async_db),
    response: Response | None = None,
) -> Any:
    """Idempotent fingerprint slot creation."""
    person = await session.get(Person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")
    fingerprint, created = await FingerprintRepository.create(
        session,
        person_id=person_id,
        finger_position=data.finger_position,
        capture_type=data.capture_type,
        notes=data.notes,
    )
    if response is not None:
        response.status_code = 201 if created else 200
    return fingerprint


@router.get(
    API_PREFIX + "/persons/{person_id}/fingerprints",
    response_model=FingerprintListResponse,
    summary="List fingerprints for a person",
)
async def list_fingerprints(
    person_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    """List fingerprint slots for a person."""
    items = await FingerprintRepository.list_by_person(session, person_id)
    response_items = [FingerprintResponse.model_validate(item) for item in items]
    return FingerprintListResponse(items=response_items, total=len(response_items))

"""Async Persons API — Phase 17."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query

if TYPE_CHECKING:
    import uuid

    from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db
from src.api.prefix import API_PREFIX
from src.schemas.person_schema import PersonCreate, PersonResponse
from src.services.person_service import PersonService

log = logging.getLogger(__name__)

router = APIRouter(prefix=f"{API_PREFIX}/persons", tags=["persons"])


@router.post(
    "/",
    response_model=PersonResponse,
    status_code=201,
    summary="Create a new person",
    responses={409: {"description": "Person with this external ID already exists"}},
)
async def create_person(
    data: PersonCreate,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    svc = PersonService(session)
    try:
        p = await svc.create_person(data)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return p


@router.get(
    "/{person_id}",
    response_model=PersonResponse,
    summary="Get person details",
    responses={404: {"description": "Person not found"}},
)
async def get_person(
    person_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    svc = PersonService(session)
    p = await svc.get_person(person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return p


@router.get(
    "/",
    response_model=list[PersonResponse],
    summary="List persons",
)
async def list_persons(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Max results per page"),
    search: str | None = Query(
        None, description="Search term for name or external ID"
    ),
    session: AsyncSession = Depends(get_async_db),
) -> Any:
    svc = PersonService(session)
    return await svc.list_persons(skip=skip, limit=limit, search=search)

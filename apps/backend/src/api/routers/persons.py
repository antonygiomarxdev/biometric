"""Persons API — Phase 17."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.schemas.person_schema import PersonCreate, PersonResponse
from src.services.person_service import PersonService

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/persons", tags=["persons"])


@router.post(
    "/", 
    response_model=PersonResponse, 
    status_code=201,
    summary="Create a new person",
    description="Register a new person (subject) in the system. A person can have multiple fingerprint slots.",
    responses={409: {"description": "Person with this external ID already exists"}}
)
def create_person(
    data: PersonCreate,
    session: Session = Depends(get_db),
) -> Any:
    """Create a person record."""
    svc = PersonService(session)
    try:
        p = svc.create_person(data)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return p


@router.get(
    "/{person_id}", 
    response_model=PersonResponse,
    summary="Get person details",
    description="Retrieve a person by their internal UUID.",
    responses={404: {"description": "Person not found"}}
)
def get_person(
    person_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> Any:
    """Retrieve person details."""
    svc = PersonService(session)
    p = svc.get_person(person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return p


@router.get(
    "/", 
    response_model=list[PersonResponse],
    summary="List persons",
    description="List and optionally search persons by name or ID with pagination.",
)
def list_persons(
    skip: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Max results per page"),
    search: str | None = Query(None, description="Search term for name or external ID"),
    session: Session = Depends(get_db),
) -> Any:
    """List persons with pagination and search."""
    svc = PersonService(session)
    return svc.list_persons(skip=skip, limit=limit, search=search)

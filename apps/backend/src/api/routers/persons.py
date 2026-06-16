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


@router.post("/", response_model=PersonResponse, status_code=201)
def create_person(
    data: PersonCreate,
    session: Session = Depends(get_db),
) -> Any:
    svc = PersonService(session)
    try:
        p = svc.create_person(data)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return p


@router.get("/{person_id}", response_model=PersonResponse)
def get_person(
    person_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> Any:
    svc = PersonService(session)
    p = svc.get_person(person_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Person not found")
    return p


@router.get("/", response_model=list[PersonResponse])
def list_persons(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    search: str | None = None,
    session: Session = Depends(get_db),
) -> Any:
    svc = PersonService(session)
    return svc.list_persons(skip=skip, limit=limit, search=search)

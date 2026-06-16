"""Pydantic DTOs for Person entities (Phase 17)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PersonCreate(BaseModel):
    """POST /api/v1/persons body."""

    external_id: str | None = Field(default=None, max_length=100)
    full_name: str | None = Field(default=None, max_length=300)
    doc_type: str | None = Field(default=None, max_length=20)
    doc_number: str | None = Field(default=None, max_length=100)
    sex: str | None = Field(default=None, max_length=1)
    dob: date | None = None
    notes: str | None = None


class PersonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    external_id: str | None
    full_name: str | None
    doc_type: str | None
    doc_number: str | None
    sex: str | None
    dob: date | None
    notes: str | None
    created_at: datetime
    updated_at: datetime

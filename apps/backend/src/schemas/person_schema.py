"""Pydantic DTOs for Person entities (Phase 17)."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PersonCreate(BaseModel):
    """POST /api/v1/persons body."""

    external_id: str | None = Field(default=None, max_length=100, description="External reference ID (e.g. from a national registry)")
    full_name: str | None = Field(default=None, max_length=300, description="Full name of the person")
    doc_type: str | None = Field(default=None, max_length=20, description="Type of document (e.g. cedula, passport)")
    doc_number: str | None = Field(default=None, max_length=100, description="Document number")
    sex: str | None = Field(default=None, max_length=1, description="Biological sex (M, F, X)")
    dob: date | None = Field(default=None, description="Date of birth")
    notes: str | None = Field(default=None, description="Forensic or internal notes")


class PersonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Internal UUIDv7")
    external_id: str | None = Field(description="External reference ID")
    full_name: str | None = Field(description="Full name")
    doc_type: str | None = Field(description="Document type")
    doc_number: str | None = Field(description="Document number")
    sex: str | None = Field(description="Biological sex")
    dob: date | None = Field(description="Date of birth")
    notes: str | None = Field(description="Internal notes")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

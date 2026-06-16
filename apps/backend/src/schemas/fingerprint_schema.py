"""Pydantic DTOs for Fingerprint slots (Phase 17)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FingerprintCreate(BaseModel):
    """POST /api/v1/persons/{id}/fingerprints body."""

    finger_position: int = Field(ge=0, le=14)
    capture_type: str = Field(default="rolled", max_length=20)
    notes: str | None = None


class FingerprintResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    person_id: uuid.UUID
    finger_position: int
    capture_type: str
    capture_count: int
    first_captured_at: datetime | None
    last_captured_at: datetime | None
    notes: str | None
    created_at: datetime
    updated_at: datetime


class FingerprintListResponse(BaseModel):
    items: list[FingerprintResponse]
    total: int

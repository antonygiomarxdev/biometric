"""Pydantic DTOs for Fingerprint slots (Phase 17)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FingerprintCreate(BaseModel):
    """POST /api/v1/persons/{id}/fingerprints body."""

    finger_position: int = Field(ge=0, le=14, description="NIST FGP (Finger Position Code) from 0 to 14")
    capture_type: str = Field(default="rolled", max_length=20, description="Capture type: rolled, plain, slap, latent, palm, segment")
    notes: str | None = Field(default=None, description="Forensic notes for this specific finger slot")


class FingerprintResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Fingerprint slot internal UUIDv7")
    person_id: uuid.UUID = Field(description="UUID of the person this fingerprint belongs to")
    finger_position: int = Field(description="NIST FGP code")
    capture_type: str = Field(description="Type of capture")
    capture_count: int = Field(description="Number of captures (images) enrolled in this slot")
    first_captured_at: datetime | None = Field(description="Timestamp of the first capture")
    last_captured_at: datetime | None = Field(description="Timestamp of the latest capture")
    notes: str | None = Field(description="Forensic notes")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class FingerprintListResponse(BaseModel):
    items: list[FingerprintResponse] = Field(description="List of fingerprint slots")
    total: int = Field(description="Total count of slots")

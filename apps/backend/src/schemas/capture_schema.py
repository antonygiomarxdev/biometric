"""Pydantic DTOs for FingerprintCapture (Phase 17, updated Phase 29)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class CaptureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Capture internal UUIDv7")
    fingerprint_id: uuid.UUID = Field(description="Parent fingerprint slot UUID")
    capture_index: int = Field(description="Index of this capture within the slot")
    image_uri: str = Field(description="URI of the original image (e.g., MinIO path)")
    algorithm_version: str = Field(description="Algorithm version used for extraction")
    processed_at: datetime = Field(description="When the image was processed")
    is_reference: bool = Field(description="Whether this is the designated reference capture")
    is_exemplar: bool = Field(description="Whether this is an exemplar (known) or latent capture")
    notes: str | None = Field(description="Forensic notes")


class CaptureUpdate(BaseModel):
    """PATCH /api/v1/captures/{id} body."""

    is_reference: bool | None = Field(default=None, description="Set as reference capture")
    notes: str | None = Field(default=None, description="Update forensic notes")


class CaptureUploadResponse(BaseModel):
    """Response of POST /fingerprints/{id}/captures (multipart)."""

    capture: CaptureResponse = Field(description="The created capture record")
    embedding_id: str | None = Field(default=None, description="Qdrant embedding point ID")

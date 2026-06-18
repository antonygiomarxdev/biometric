"""Pydantic DTOs for Fingerprint slots (Phase 17) and preview (Phase 23)."""

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


class MinutiaPoint(BaseModel):
    """A single detected minutia point (Phase 23 — preview and match overlay).

    Mirrors the ``MinutiaPoint`` interface in the frontend
    (``apps/frontend/src/lib/api.ts``) used by ``useCanvasDrawer`` and
    ``MatchOverlay``.
    """
    x: int
    y: int
    angle: float
    type: int  # 0=termination, 1=bifurcation, 2=unknown


class FingerprintPreviewResponse(BaseModel):
    """POST /api/v1/fingerprints/preview body (Phase 23).

    Mirrors the legacy ``ExtractResponse`` shape consumed by
    ``FingerprintViewer`` and the new ``getMinutiaeForImage`` in
    ``apps/frontend/src/lib/api.ts``. Does NOT persist a capture;
    the perito reviews the extracted minutiae in the UI before
    explicitly enrolling.
    """
    processed_image: str = Field(
        description="Base64 PNG of the enhanced/skeletonized image (no data: prefix)",
    )
    minutiae: list[MinutiaPoint] = Field(
        default_factory=list,
        description="Detected minutiae (x, y, angle, type)",
    )
    terminations: int = Field(default=0, description="Count of termination minutiae (type=0)")
    bifurcations: int = Field(default=0, description="Count of bifurcation minutiae (type=1)")
    image_shape: list[int] = Field(
        default_factory=lambda: [0, 0],
        description="[height, width] of the processed image",
    )
    image_dtype: str = Field(default="uint8", description="Numpy dtype of the processed image")


FingerprintPreviewResponse.model_rebuild()

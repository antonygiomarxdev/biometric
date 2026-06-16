"""Pydantic DTOs for FingerprintCapture and RidgeGraph (Phase 17)."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RidgeGraphResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    capture_id: uuid.UUID
    graph_index: int
    region_x: int
    region_y: int
    region_w: int
    region_h: int
    num_nodes: int
    num_edges: int
    core_x: int | None
    core_y: int | None
    delta_x: int | None
    delta_y: int | None
    singularity_type: str | None


class CaptureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    fingerprint_id: uuid.UUID
    capture_index: int
    image_uri: str
    image_dpi: int | None
    image_quality_score: float | None
    algorithm_version: str
    processed_at: datetime
    num_minutiae: int | None
    num_graphs: int | None
    is_reference: bool
    is_exemplar: bool
    notes: str | None
    graphs: list[RidgeGraphResponse] = []


class CaptureUpdate(BaseModel):
    """PATCH /api/v1/captures/{id} body."""

    is_reference: bool | None = None
    notes: str | None = None


class CaptureUploadResponse(BaseModel):
    """Response of POST /fingerprints/{id}/captures (multipart)."""

    capture: CaptureResponse
    graphs_created: int

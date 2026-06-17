"""Pydantic DTOs for FingerprintCapture and RidgeGraph (Phase 17)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import uuid
    from datetime import datetime


class RidgeGraphResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="RidgeGraph internal UUIDv7")
    capture_id: uuid.UUID = Field(description="Parent capture UUID")
    graph_index: int = Field(description="Index of this graph component within the capture")
    region_x: int = Field(description="Bounding box X coordinate")
    region_y: int = Field(description="Bounding box Y coordinate")
    region_w: int = Field(description="Bounding box width")
    region_h: int = Field(description="Bounding box height")
    num_nodes: int = Field(description="Number of minutiae nodes in the graph")
    num_edges: int = Field(description="Number of ridge edges connecting the nodes")
    core_x: int | None = Field(description="Core singularity X coordinate, if found")
    core_y: int | None = Field(description="Core singularity Y coordinate, if found")
    delta_x: int | None = Field(description="Delta singularity X coordinate, if found")
    delta_y: int | None = Field(description="Delta singularity Y coordinate, if found")
    singularity_type: str | None = Field(description="Type of singularity (e.g. 'core', 'delta')")


class CaptureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(description="Capture internal UUIDv7")
    fingerprint_id: uuid.UUID = Field(description="Parent fingerprint slot UUID")
    capture_index: int = Field(description="Index of this capture within the slot")
    image_uri: str = Field(description="URI of the original image (e.g., MinIO path)")
    image_dpi: int | None = Field(description="Resolution of the image in DPI")
    image_quality_score: float | None = Field(description="Automated quality score (e.g., NFIQ-2)")
    algorithm_version: str = Field(description="Algorithm version used for extraction")
    processed_at: datetime = Field(description="When the image was processed")
    num_minutiae: int | None = Field(description="Total minutiae extracted")
    num_graphs: int | None = Field(description="Total disconnected graph components extracted")
    is_reference: bool = Field(description="Whether this is the designated reference capture")
    is_exemplar: bool = Field(description="Whether this is an exemplar (known) or latent capture")
    notes: str | None = Field(description="Forensic notes")
    graphs: list[RidgeGraphResponse] = Field(default=[], description="Nested ridge graphs")


class CaptureUpdate(BaseModel):
    """PATCH /api/v1/captures/{id} body."""

    is_reference: bool | None = Field(default=None, description="Set as reference capture")
    notes: str | None = Field(default=None, description="Update forensic notes")


class CaptureUploadResponse(BaseModel):
    """Response of POST /fingerprints/{id}/captures (multipart)."""

    capture: CaptureResponse = Field(description="The created capture record")
    graphs_created: int = Field(description="Number of ridge graphs extracted")

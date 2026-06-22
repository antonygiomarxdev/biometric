from __future__ import annotations

from pydantic import BaseModel


class EmbeddingCandidate(BaseModel):
    person_id: str
    score: float
    full_name: str | None = None
    external_id: str | None = None
    image_url: str | None = None
    capture_id: str
    finger_name: str | None = None


class EmbeddingSearchResponse(BaseModel):
    query_time_ms: int
    probe_gradcam_b64: str | None = None
    enhance_applied: bool
    search_mode: str = "single"
    total_candidates: int
    candidates: list[EmbeddingCandidate]

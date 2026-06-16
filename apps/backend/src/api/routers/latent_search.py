"""
Qdrant-backed latent fingerprint search router.

Phase 15+ (Qdrant Chunked Indexing): the search endpoint accepts
a latent image and returns a ranked list of candidate persons
whose enrolled chunks most closely match the local invariants
extracted from the query.

The forensic rule (``SearchValidationStrategy``) accepts as few
as 2 minutiae. The matcher is robust to noisy fragments because
noise produces triangles that simply do not match any enrolled
chunk in the database.

The underlying ``QdrantRagMatchingService`` replaces the deprecated
``RagMatchingService`` (pgvector). No SQLAlchemy ``Session`` needed.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from src.api.dependencies import resources
from src.services.rag_matching_service import QdrantRagMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/matching",
    tags=["matching"],
)


def _get_qdrant_matching_service() -> QdrantRagMatchingService:
    return QdrantRagMatchingService(pool=resources.process_pool)


@router.post("/search")
async def search_latent(
    file: UploadFile = File(..., description="Latent fingerprint image"),
    top_k_per_chunk: int = 5,
    top_k_persons: int = 10,
    matching: QdrantRagMatchingService = Depends(_get_qdrant_matching_service),
) -> dict[str, Any]:
    """Search the Qdrant chunk store for a latent fingerprint.

    Returns the ranked candidates (person_id, aggregated weighted
    score, hit count) ordered by ``total_score`` descending.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    hits = await matching.search_async(
        image_bytes=image_bytes,
        top_k_per_chunk=top_k_per_chunk,
        top_k_persons=top_k_persons,
    )
    return {
        "success": True,
        "candidates": [
            {
                "person_id": h.person_id,
                "total_score": h.total_score,
                "hits": h.hits,
            }
            for h in hits
        ],
        "query_chunks": len(hits),
    }

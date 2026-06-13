"""
Router for latent fingerprint matching (``/api/v1/matching``).

Per D-02, D-03:
  - One router per resource, mounted under ``/api/v1/matching``.
  - Injects ``get_db`` and the lifespan-managed ``ProcessPoolExecutor``
    via ``Depends(get_db)`` and the ``MatchingService`` dependency.

Endpoints:
  - ``POST /search`` — Upload a latent fingerprint image and search for
    Top-K candidates from the known-prints gallery.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, resources
from src.core.config import config
from src.services.matching_service import MatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/matching",
    tags=["matching"],
)


def _get_matching_service() -> MatchingService:
    """
    Dependency provider for ``MatchingService``.

    Uses the application-scoped ``ProcessPoolExecutor`` from the lifespan
    manager so CPU-heavy processing never blocks the event loop.
    """
    return MatchingService(pool=resources.process_pool)


@router.post("/search")
async def search_latent(
    file: UploadFile = File(..., description="Latent fingerprint image (BMP, PNG, JPEG)"),
    top_k: int = Query(
        default=config.top_k_matches,
        ge=1,
        le=100,
        description="Number of Top-K candidates to return",
    ),
    db: Session = Depends(get_db),
    matching: MatchingService = Depends(_get_matching_service),
) -> dict[str, Any]:
    """
    Upload a latent fingerprint and search for Top-K candidates.

    The image is:
      1. Decoded and processed by ``FingerprintService`` inside the
         ``ProcessPoolExecutor`` (CPU-bound).
      2. Converted to a fixed-dimension query vector.
      3. Searched against the ``fingerprint_vectors`` table using the
         pgvector HNSW L2 distance operator (``<->``).

    Returns the Top-K candidates ranked by similarity (ascending L2 distance).
    """
    logger.info(
        "Latent matching request — file=%s, size_hint=%s, top_k=%d",
        file.filename,
        file.size,
        top_k,
    )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    candidates = await matching.search_latent(
        image_bytes=image_bytes,
        top_k=top_k,
        db=db,
    )

    results_list: list[dict[str, Any]] = []
    for c in candidates:
        results_list.append({
            "person_id": c.person_id,
            "name": c.name,
            "document": c.document,
            "evidence_id": c.evidence_id,
            "l2_distance": c.l2_distance,
            "score": round(c.score, 6),
        })

    logger.info("Latent search complete — %d candidates returned", len(results_list))

    return {
        "success": True,
        "top_k": top_k,
        "candidates": results_list,
        "total": len(results_list),
    }

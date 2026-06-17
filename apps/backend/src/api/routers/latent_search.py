"""
Fingerprint latent search router — Phase 21 (MCC production backend).

Accepts a latent/probe fingerprint image, processes it through the
extraction pipeline, builds MCC cylinder descriptors (144-D), and
searches the Qdrant MCC store for matching enrolled persons.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db, get_mcc_matching_service
from src.api.prefix import API_PREFIX
from src.db.models import Person
from src.services.mcc_matching_service import MccMatchingService, MccSearchHit

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{API_PREFIX}/matching",
    tags=["matching"],
)


@router.post(
    "/search",
    summary="Search enrolled prints for a latent/probe image",
    description=(
        "Accepts a latent/probe fingerprint image, runs the full extraction "
        "pipeline (Gabor enhancement → skeletonization → minutiae → MCC cylinder "
        "descriptors), and returns the top-K enrolled candidates ranked by "
        "cosine-similarity with per-fingerprint normalization.\n\n"
        "Each minutia is represented by a 144-D MCC cylinder vector (12 angular "
        "sectors × 4 radial rings × 3 structural features), L2-normalized and "
        "rotation-invariant.\n\n"
        "Search is cosine KNN per cylinder, votes aggregated per person, then "
        "normalized by the number of enrolled cylinders to remove bias toward "
        "enrollees with more minutiae."
    ),
    responses={
        200: {"description": "Ranked candidates with scores and person metadata"},
        400: {"description": "Empty file or invalid image bytes"},
    },
)
async def search_latent(
    file: UploadFile = File(..., description="Latent/probe fingerprint image"),
    top_k: int = 10,
    matching: MccMatchingService = Depends(get_mcc_matching_service),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, Any]:
    """Search enrolled fingerprints for matches to a probe image."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    hits: list[MccSearchHit] = matching.search(image_bytes, top_k=top_k)

    candidates: list[dict[str, Any]] = []
    for hit in hits:
        person_info: dict[str, Any] = {"person_id": hit.person_id}
        try:
            person_uuid = UUID(hit.person_id)
        except ValueError:
            person_uuid = None

        if person_uuid is not None:
            person = await session.get(Person, person_uuid)
        else:
            result = await session.execute(
                select(Person).where(Person.external_id == hit.person_id)
            )
            person = result.scalar_one_or_none()

        if person is not None:
            person_info["full_name"] = person.full_name
            person_info["external_id"] = person.external_id

        candidates.append({
            "person_id": hit.person_id,
            "total_score": round(hit.total_score, 4),
            "hits": hit.hits,
            "full_name": person_info.get("full_name"),
            "external_id": person_info.get("external_id"),
        })

    return {
        "success": True,
        "query_time_ms": 0,
        "total_candidates": len(candidates),
        "candidates": candidates,
    }

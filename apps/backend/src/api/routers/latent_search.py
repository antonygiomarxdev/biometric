"""
Fingerprint latent search router — Phase 18.

Accepts a latent/probe fingerprint image, processes it through the
extraction pipeline, chunk-vectorises it via Delaunay triangulation,
and searches the Qdrant chunk store for matching enrolled persons.

Uses the same QdrantChunkRepository collection that
FingerprintEnrollmentService writes to, ensuring searches find
previously enrolled prints.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db, get_rag_matching_service
from src.api.prefix import API_PREFIX
from src.db.models import Person
from src.services.rag_matching_service import (
    QdrantRagMatchingService,
    SearchHit,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{API_PREFIX}/matching",
    tags=["matching"],
)


@router.post("/search")
async def search_latent(
    file: UploadFile = File(..., description="Latent/probe fingerprint image"),
    top_k: int = 10,
    matching: QdrantRagMatchingService = Depends(get_rag_matching_service),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, Any]:
    """Search enrolled fingerprints for matches to a probe image.

    1. Decode image bytes
    2. Run extraction pipeline (enhance → skeletonize → minutiae → triangles)
    3. Search Qdrant for matching chunk signatures
    4. Aggregate hits by person
    5. Enrich with person names from the database

    Returns ranked candidates ordered by total_score descending.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    hits: list[SearchHit] = await matching.search_async(
        image_bytes,
        top_k_persons=top_k,
    )

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

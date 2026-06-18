"""
Fingerprint latent search router — Phase 25 (triplet-based matching).

Accepts a latent/probe fingerprint image, runs the quality pipeline,
extracts triplet feature vectors (6-D per triplet),
and searches the Qdrant triplet_features store for matching persons.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select

from src.api.dependencies import get_async_db, get_mcc_matching_service
from src.api.prefix import API_PREFIX
from src.db.models import Person
from src.dev.logger import dev_log

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.services.mcc_matching_service import MccMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{API_PREFIX}/matching",
    tags=["matching"],
)


@router.post(
    "/search",
    summary="Search enrolled prints using triplet-based matching",
    description=(
        "Uses the thinning + Crossing Number pipeline for minutiae extraction, "
        "triplet feature vectors (6-D per triplet), and KNN search against the "
        "triplet_features collection. Returns top-K candidates with score, "
        "peak_votes, supporting_triplets, and probe_minutiae."
    ),
    responses={
        200: {"description": "Ranked candidates with triplet-based scores"},
        400: {"description": "Empty file or invalid image bytes"},
    },
)
async def search_latent(
    file: UploadFile = File(..., description="Probe fingerprint image"),
    top_k: int = 10,
    matching: MccMatchingService = Depends(get_mcc_matching_service),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    t0 = time.monotonic()

    # Fetch enrolled OF records for pre-filter (Phase 26)
    try:
        from src.db.of_registry import OFRegistry
        registry = OFRegistry(session)
        enrolled_ofs_raw = await registry.get_all()
        enrolled_ofs: dict[str, Any] | None = enrolled_ofs_raw
    except Exception:
        enrolled_ofs = None

    result = matching.search_by_triplets(
        image_bytes, top_k=top_k,
        enrolled_ofs=enrolled_ofs,
    )
    query_time_ms = int((time.monotonic() - t0) * 1000)

    candidates = result["candidates"]
    probe_minutiae = result["probe_minutiae"]

    dev_log(
        "search.endpoint",
        image_bytes=len(image_bytes),
        top_k=top_k,
        candidates=len(candidates),
        probe_minutiae=len(probe_minutiae),
        query_time_ms=query_time_ms,
    )

    # Resolve person names from DB
    for c in candidates:
        pid = c["person_id"]
        try:
            from uuid import UUID
            person_uuid = UUID(pid)
        except ValueError:
            person_uuid = None

        if person_uuid is not None:
            person = await session.get(Person, person_uuid)
        else:
            result = await session.execute(
                select(Person).where(Person.external_id == pid)
            )
            person = result.scalar_one_or_none()

        if person is not None:
            c["full_name"] = person.full_name
            c["external_id"] = person.external_id
        else:
            c["full_name"] = None
            c["external_id"] = None

    return {
        "success": True,
        "query_time_ms": query_time_ms,
        "total_candidates": len(candidates),
        "probe_minutiae": probe_minutiae,
        "candidates": candidates,
    }

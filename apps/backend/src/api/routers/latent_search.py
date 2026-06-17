"""
Fingerprint latent search router — Phase 21 (MCC production backend).

Accepts a latent/probe fingerprint image, processes it through the
extraction pipeline, builds MCC cylinder descriptors (144-D), and
searches the Qdrant MCC store for matching enrolled persons.
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db, get_mcc_matching_service
from src.api.prefix import API_PREFIX
from src.db.models import Person
from src.services.mcc_matching_service import MccMatchingService

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
        "enrollees with more minutiae.\n\n"
        "Phase 23 extension: response includes top-level ``probe_minutiae`` "
        "(for frontend canvas rendering without an extra round-trip to /extract) "
        "and per-candidate ``match_trace`` (cylinder-level connecting-line data)."
    ),
    responses={
        200: {"description": "Ranked candidates with scores, match_trace, and probe_minutiae"},
        400: {"description": "Empty file or invalid image bytes"},
    },
)
async def search_latent(
    file: UploadFile = File(..., description="Latent/probe fingerprint image"),
    top_k: int = 10,
    matching: MccMatchingService = Depends(get_mcc_matching_service),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, Any]:
    """Search enrolled fingerprints for matches to a probe image.

    Returns a JSON envelope with:
      - ``success`` (bool): always true on 2xx.
      - ``query_time_ms`` (int): wall-clock duration of the search.
      - ``total_candidates`` (int): number of candidates in the response.
      - ``probe_minutiae`` (list[dict]): the probe's minutiae
        (x, y, angle, type) so the frontend can draw the probe canvas
        without an extra round-trip to /extract.
      - ``candidates`` (list[dict]): ranked candidates; each includes
        ``match_trace`` (list[dict]) for cylinder-level connecting-line
        rendering and ``contributing_fingerprints`` for D-08.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    t0 = time.monotonic()
    probe_minutiae, hits = matching.search(image_bytes, top_k=top_k)
    query_time_ms = int((time.monotonic() - t0) * 1000)

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

        match_trace = [
            {
                "probe_cylinder_index": e.probe_cylinder_index,
                "probe_x": e.probe_x,
                "probe_y": e.probe_y,
                "probe_angle": e.probe_angle,
                "candidate_capture_id": e.candidate_capture_id,
                "candidate_fingerprint_id": e.candidate_fingerprint_id,
                "candidate_x": e.candidate_x,
                "candidate_y": e.candidate_y,
                "candidate_angle": e.candidate_angle,
                "similarity": round(e.similarity, 4),
            }
            for e in hit.match_trace
        ]
        candidates.append({
            "person_id": hit.person_id,
            "total_score": round(hit.total_score, 4),
            "hits": hit.hits,
            "full_name": person_info.get("full_name"),
            "external_id": person_info.get("external_id"),
            "contributing_fingerprints": list(hit.contributing_fingerprints),
            "match_trace": match_trace,
        })

    return {
        "success": True,
        "query_time_ms": query_time_ms,
        "total_candidates": len(candidates),
        "probe_minutiae": [
            {
                "x": m.x,
                "y": m.y,
                "angle": round(m.angle, 4),
                "type": m.type,
            }
            for m in probe_minutiae
        ],
        "candidates": candidates,
    }

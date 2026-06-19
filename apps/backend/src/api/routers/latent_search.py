"""
Fingerprint latent search router — NIST-style MCC matching.

Accepts a latent/probe fingerprint image, runs the quality pipeline,
extracts MCC cylinder descriptors (144-D per minutia), and searches
the Qdrant mcc_cylinders store with cosine KNN + Hough voting.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select

from src.api.dependencies import get_async_db, get_mcc_matching_service
from src.api.prefix import API_PREFIX
from src.core.config import config
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
    summary="Search enrolled prints using MCC cylinders + Hough voting",
    description=(
        "NIST Bozorth3-style matching: extracts MCC cylinder descriptors "
        "(144-D) per minutia, runs cosine KNN against the mcc_cylinders "
        "collection, then Hough-votes the spatial alignment to keep only "
        "geometrically consistent hits. Returns ranked candidates with "
        "score, hits, match trace, and probe minutiae."
    ),
    responses={
        200: {"description": "Ranked candidates with NIST MCC scores"},
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

    # NIST-style MCC matching: cylinders + Hough voting.
    # No OF pre-filter — it is not well calibrated for altered prints.
    probe_minutiae_summaries, hits = await asyncio.get_running_loop().run_in_executor(
        None, matching.search, image_bytes, top_k,
    )

    query_time_ms = int((time.monotonic() - t0) * 1000)

    # Convert probe minutiae summaries to dicts for the response
    probe_minutiae = [
        {
            "x": int(m.x),
            "y": int(m.y),
            "angle": float(m.angle),
            "type": int(m.type),
        }
        for m in probe_minutiae_summaries
    ]

    # Build candidates list, resolving person names from DB
    confidence_threshold = config.matching.confidence_threshold
    candidates: list[dict[str, Any]] = []
    for hit in hits:
        # Filter weak candidates — show only those above the
        # confidence threshold. Real matches in NIST practice
        # typically produce 80%+ confidence; below 50% is noise.
        if float(hit.total_score) < confidence_threshold:
            continue
        pid = hit.person_id
        try:
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

        full_name = person.full_name if person is not None else None
        external_id = person.external_id if person is not None else None

        # The match_trace carries ``candidate_capture_id`` and
        # ``candidate_fingerprint_id`` directly from the Qdrant
        # payload. These are whatever the enrollment script passed
        # in. The cylinder enrollment accepts opaque strings for
        # both, so they are NOT guaranteed to match rows in
        # PostgreSQL — re-enrollments across Phase 25 left orphan
        # cylinder payloads whose fingerprint_id/capture_id refer
        # to rows that no longer exist.
        #
        # The image endpoint serves the Gabor-enhanced PNG from
        # PostgreSQL, so we need a ``FingerprintCapture.id`` that
        # exists there. Since the per-person fingerprint_id can
        # be stale, we resolve by ``person_id`` (the most stable
        # identifier) and pick the most recent FingerprintCapture
        # for that person.
        from src.db.models import Fingerprint, FingerprintCapture

        person_capture_id: str | None = None
        try:
            person_uuid = UUID(pid)
        except ValueError:
            person_uuid = None
        lookup_id = person_uuid if person_uuid is not None else (person.id if person is not None else None)
        if lookup_id is not None:
            cap = (
                await session.execute(
                    select(FingerprintCapture)
                    .join(Fingerprint, FingerprintCapture.fingerprint_id == Fingerprint.id)
                    .where(Fingerprint.person_id == lookup_id)
                    .order_by(FingerprintCapture.created_at.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if cap is not None:
                person_capture_id = str(cap.id)

        # Map match_trace entries to the frontend's supporting_pairs shape.
        # Each trace entry is a (probe_cylinder, candidate_cylinder) match.
        supporting_pairs: list[dict[str, Any]] = []
        for t in hit.match_trace:
            # Prefer the trace's own capture_id if it exists in PostgreSQL
            # (preserves alignment between cylinder and its image). Fall
            # back to the person-level latest capture. Last resort: keep
            # the trace's stale id (frontend will 404, but at least the
            # other endpoints still work).
            trace_capture_id_str = str(t.candidate_capture_id)
            try:
                trace_uuid = UUID(trace_capture_id_str)
                trace_capture = await session.get(FingerprintCapture, trace_uuid)
            except (ValueError, TypeError):
                trace_capture = None
            if trace_capture is not None:
                capture_id = trace_capture_id_str
            else:
                capture_id = person_capture_id or trace_capture_id_str
            supporting_pairs.append({
                "probe_mi_idx": int(t.probe_cylinder_index),
                "candidate_mi_x": float(t.candidate_x),
                "candidate_mi_y": float(t.candidate_y),
                "candidate_mi_angle": float(t.candidate_angle),
                "candidate_fingerprint_id": str(t.candidate_fingerprint_id),
                "candidate_capture_id": capture_id,
                "similarity": float(t.similarity),
            })

        candidates.append({
            "person_id": pid,
            "score": float(hit.total_score),
            "peak_votes": int(hit.hits),
            "supporting_pairs": supporting_pairs,
            "num_probe_pairs": len(probe_minutiae_summaries),
            "full_name": full_name,
            "external_id": external_id,
        })

    dev_log(
        "search.endpoint",
        image_bytes=len(image_bytes),
        top_k=top_k,
        candidates=len(candidates),
        probe_minutiae=len(probe_minutiae),
        query_time_ms=query_time_ms,
    )

    return {
        "success": True,
        "query_time_ms": query_time_ms,
        "total_candidates": len(candidates),
        "probe_minutiae": probe_minutiae,
        "candidates": candidates,
    }

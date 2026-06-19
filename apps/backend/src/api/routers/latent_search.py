"""
Fingerprint latent search router — NIST Bozorth3 pair matching (Phase 27).

Accepts a latent/probe fingerprint image, runs the quality pipeline,
extracts 5-D pair descriptors, runs KNN against the pair_features
collection, then Bozorth3 linking groups geometrically compatible
matches. Returns ranked candidates with score, hits, match trace,
and probe minutiae.
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
    summary="Search enrolled prints using NIST Bozorth3 pair linking",
    description=(
        "NIST Bozorth3-style matching: extracts 5-D pair descriptors per "
        "minutia pair, runs KNN against the pair_features collection, then "
        "Union-Find links geometrically compatible matches. Returns ranked "
        "candidates with score, hits, match trace, and probe minutiae."
    ),
    responses={
        200: {"description": "Ranked candidates with NIST Bozorth3 scores"},
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
    result = await asyncio.get_running_loop().run_in_executor(
        None, matching.search_by_pairs, image_bytes, top_k,
    )
    query_time_ms = int((time.monotonic() - t0) * 1000)

    probe_minutiae = [
        {
            "x": int(m["x"]),
            "y": int(m["y"]),
            "angle": float(m["angle"]),
            "type": int(m.get("type", 2)),
        }
        for m in result["probe_minutiae"]
    ]

    candidates: list[dict[str, Any]] = []
    for hit in result["candidates"]:
        pid = hit["person_id"]
        try:
            person_uuid = UUID(pid)
        except ValueError:
            person_uuid = None

        if person_uuid is not None:
            person = await session.get(Person, person_uuid)
        else:
            result_q = await session.execute(
                select(Person).where(Person.external_id == pid)
            )
            person = result_q.scalar_one_or_none()

        full_name = person.full_name if person is not None else None
        external_id = person.external_id if person is not None else None

        from src.db.models import Fingerprint, FingerprintCapture

        person_capture_id: str | None = None
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

        supporting_pairs = list(hit.get("supporting_pairs", []))
        for sp in supporting_pairs:
            tc = str(sp.get("candidate_capture_id", ""))
            try:
                trace_uuid = UUID(tc)
                trace_capture = await session.get(FingerprintCapture, trace_uuid)
            except (ValueError, TypeError):
                trace_capture = None
            if trace_capture is not None:
                sp["candidate_capture_id"] = tc
            else:
                sp["candidate_capture_id"] = person_capture_id or tc

        score = float(hit["score"])
        if score >= 0.7:
            confidence = "alta"
        elif score >= 0.4:
            confidence = "media"
        else:
            confidence = "baja"

        # Find the most frequent capture_id from supporting_pairs
        capture_ids: dict[str, int] = {}
        for sp in supporting_pairs:
            cid = str(sp.get("candidate_capture_id", ""))
            capture_ids[cid] = capture_ids.get(cid, 0) + 1
        best_capture_id = max(capture_ids, key=lambda k: capture_ids[k]) if capture_ids else None

        candidate_minutiae: list[dict[str, float | int]] = []
        if best_capture_id:
            try:
                cap_uuid = UUID(best_capture_id)
                cap_entity = await session.get(FingerprintCapture, cap_uuid)
                if cap_entity is not None:
                    from src.db.repositories.capture_minutia_repository import (
                        CaptureMinutiaRepository,
                    )
                    rows = await CaptureMinutiaRepository.list_for_capture(session, cap_uuid)
                    # rows have x, y as normalized (0-1); convert to 256x256 pixel coords
                    candidate_minutiae = [
                        {
                            "x": round(float(r.x) * 256),
                            "y": round(float(r.y) * 256),
                            "angle": float(r.angle),
                            "type": int(r.type),
                        }
                        for r in rows
                    ]
            except (ValueError, TypeError):
                pass

        candidates.append({
            "person_id": pid,
            "score": score,
            "peak_votes": int(hit["peak_votes"]),
            "supporting_pairs": supporting_pairs,
            "num_probe_pairs": int(hit.get("num_probe_pairs", 0)),
            "full_name": full_name,
            "external_id": external_id,
            "confidence": confidence,
            "capture_id": best_capture_id,
            "candidate_minutiae": candidate_minutiae,
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

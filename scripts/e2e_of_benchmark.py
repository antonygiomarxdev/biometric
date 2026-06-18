"""E2E benchmark for Phase 26 OF pre-filter pipeline.

Validates the Plan 26-01 acceptance gate:
  - Self-match: 5/5 SOCOFing persons match themselves > 0.5
  - 50% center crop: >=4/5 match > 0.4
  - 25% corner crop: >=3/5 match > 0.3
  - Search latency: < 3s per query

Adapted from ``e2e_triplet_benchmark.py`` (Phase 25).

Usage (from apps/backend):
    uv run python ../../scripts/e2e_of_benchmark.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import cv2
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import config
from src.db.qdrant_mcc_repository import QdrantMccRepository, TRIPLET_COLLECTION_NAME
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)
from src.services.mcc_matching_service import MccMatchingService
from src.services.person_service import PersonService

SOCOFING_REAL = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "backend"
    / "static"
    / "SOCOFing"
    / "Real"
)

PERSONS = ["SOC_0100", "SOC_0101", "SOC_0102", "SOC_0103", "SOC_0104"]
FINGER_NAME = "index"

# Plan 26-01 acceptance thresholds
SELF_MATCH_THRESHOLD = 0.5
CROP_50_THRESHOLD = 0.4
CROP_25_THRESHOLD = 0.3

# Latency target
MAX_LATENCY_MS = 3000


def find_socofing_file(person_external_id: str) -> Path:
    pid = person_external_id.replace("SOC_", "").lstrip("0")
    for path in sorted(SOCOFING_REAL.glob(f"{pid}__*_{FINGER_NAME}_finger.BMP")):
        return path
    raise FileNotFoundError(
        f"No {FINGER_NAME} BMP found for {person_external_id} (pid={pid})",
    )


def crop_center(img_bytes: bytes, fraction: float) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    y_off = (h - ch) // 2
    x_off = (w - cw) // 2
    cropped = img[y_off:y_off + ch, x_off:x_off + cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


def crop_corner(img_bytes: bytes, fraction: float) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    cropped = img[:ch, :cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


async def _fetch_all_ofs(
    session: AsyncSession,
) -> dict[str, dict[str, object]] | None:
    try:
        from src.db.of_registry import OFRegistry

        registry = OFRegistry(session)
        raw = await registry.get_all()
        result: dict[str, dict[str, object]] = {}
        for fid, rec in raw.items():
            result[fid] = {
                "fingerprint_id": rec["fingerprint_id"],
                "of_ori": rec["of_ori"],
                "of_coh": rec["of_coh"],
                "block_size": rec["block_size"],
                "pseudo_core": (
                    list(rec["pseudo_core"]) if rec["pseudo_core"] else None
                ),
            }
        return result
    except Exception as exc:
        print(f"  WARNING: OF fetch failed: {exc}")
        return None


async def main() -> int:
    engine = create_async_engine(config.async_database_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    mcc = MccMatchingService()

    # Clean up old data (DB + Qdrant)
    print("=" * 60)
    print("Cleaning state for", PERSONS)
    print("=" * 60)

    # Drop Qdrant triplet_features collection
    repo = QdrantMccRepository.from_host()
    collections = repo._client.get_collections().collections
    existing = [c.name for c in collections]
    if TRIPLET_COLLECTION_NAME in existing:
        print(f"  Dropping existing {TRIPLET_COLLECTION_NAME} collection...")
        repo._client.delete_collection(TRIPLET_COLLECTION_NAME)
    repo.ensure_triplet_collection()
    print(f"  Empty {TRIPLET_COLLECTION_NAME} collection ready.")

    # Clean DB
    async with engine.begin() as conn:
        for pid in PERSONS:
            await conn.execute(
                text("""
                    DELETE FROM fingerprint_of_index
                    WHERE fingerprint_id IN (
                        SELECT fp.id FROM fingerprints fp
                        JOIN persons p ON p.id = fp.person_id
                        WHERE p.external_id = :pid
                    )
                """),
                {"pid": pid},
            )
            await conn.execute(
                text("""
                    DELETE FROM fingerprint_captures
                    WHERE fingerprint_id IN (
                        SELECT fp.id FROM fingerprints fp
                        JOIN persons p ON p.id = fp.person_id
                        WHERE p.external_id = :pid
                    )
                """),
                {"pid": pid},
            )
            await conn.execute(
                text("""
                    DELETE FROM fingerprints
                    WHERE person_id IN (
                        SELECT id FROM persons WHERE external_id = :pid
                    )
                """),
                {"pid": pid},
            )
            await conn.execute(
                text("DELETE FROM persons WHERE external_id = :pid"),
                {"pid": pid},
            )

    # Enroll 5 persons via create_capture (production path)
    print()
    print("=" * 60)
    print("Enrolling 5 SOCOFing persons")
    print("=" * 60)
    enrolled: list[tuple[str, bytes]] = []

    try:
        async with Session() as session:
            person_svc = PersonService(session)
            enroll_svc = FingerprintEnrollmentService(session, mcc)
            for ext_id in PERSONS:
                person = await person_svc.find_or_create_person(
                    external_id=ext_id,
                    full_name=f"Sujeto SOCOFing {ext_id.split('_')[1]}",
                    doc_type="cedula",
                    doc_number=f"DOC_{ext_id.split('_')[1].zfill(8)}",
                )
                await session.commit()
                await session.refresh(person)

                image_path = find_socofing_file(ext_id)
                img_bytes = image_path.read_bytes()

                slot = await FingerprintRepository.create(
                    session,
                    person_id=person.id,
                    finger_position=0,
                    capture_type="rolled",
                )
                await session.commit()
                await session.refresh(slot)

                capture, _graphs = await enroll_svc.create_capture(
                    fingerprint_id=slot.id,
                    image_bytes=img_bytes,
                    image_dpi=500,
                )
                await session.commit()
                await session.refresh(capture)
                enrolled.append((ext_id, img_bytes))
                print(
                    f"  {ext_id}: capture={str(capture.id)[:8]} "
                    f"minutiae={capture.num_minutiae} "
                    f"enhanced={'yes' if capture.enhanced_image else 'no'}",
                )
    finally:
        await engine.dispose()

    print(f"\nEnrolled {len(enrolled)} persons")

    # Fetch enrolled OF records for the pre-filter
    print()
    print("=" * 60)
    print("Fetching enrolled OF records for pre-filter")
    print("=" * 60)
    engine2 = create_async_engine(config.async_database_url)
    Session2 = async_sessionmaker(engine2, class_=AsyncSession, expire_on_commit=False)
    async with Session2() as session:
        enrolled_ofs = await _fetch_all_ofs(session)
    await engine2.dispose()
    if enrolled_ofs:
        print(f"  Fetched {len(enrolled_ofs)} OF records")
    else:
        print("  WARNING: No OF records found — filter will be a no-op")

    print()

    # ---- Test 1: Self-match ----
    print("=" * 60)
    print("Test 1: Self-match (threshold > 0.5)")
    print("=" * 60)
    self_ok = 0
    latencies: list[float] = []
    for ext_id, img_bytes in enrolled:
        t0 = time.monotonic()
        result = mcc.search_by_triplets(
            img_bytes, top_k=5,
            enrolled_ofs=enrolled_ofs,
        )
        elapsed = (time.monotonic() - t0) * 1000
        latencies.append(elapsed)
        candidates = result.get("candidates", [])
        if not candidates:
            print(f"  {ext_id} -> NO CANDIDATES  FAIL  ({elapsed:.0f}ms)")
            continue
        top = candidates[0]
        match = top["person_id"] == ext_id
        threshold_ok = top["score"] >= SELF_MATCH_THRESHOLD
        if match and threshold_ok:
            self_ok += 1
            print(
                f"  {ext_id} -> {top['person_id'][:8]} "
                f"score={top['score']:.3f} "
                f"confirmed={top['confirming_triplets']}  OK  ({elapsed:.0f}ms)",
            )
        else:
            print(
                f"  {ext_id} -> {top['person_id'][:8]} "
                f"score={top['score']:.3f} "
                f"confirmed={top['confirming_triplets']}  "
                f"{'FAIL (wrong person)' if not match else 'FAIL (low score)'}"
                f"  ({elapsed:.0f}ms)",
            )
    print(f"  Self-match: {self_ok}/{len(enrolled)}")

    # ---- Test 2: 50% center crop ----
    print()
    print("=" * 60)
    print("Test 2: 50% center crop (threshold > 0.4)")
    print("=" * 60)
    crop50_ok = 0
    for ext_id, img_bytes in enrolled:
        crop_bytes = crop_center(img_bytes, 0.5)
        t0 = time.monotonic()
        result = mcc.search_by_triplets(
            crop_bytes, top_k=5,
            enrolled_ofs=enrolled_ofs,
        )
        elapsed = (time.monotonic() - t0) * 1000
        latencies.append(elapsed)
        candidates = result.get("candidates", [])
        if not candidates:
            print(f"  crop50({ext_id}) -> NO CANDIDATES  FAIL  ({elapsed:.0f}ms)")
            continue
        top = candidates[0]
        match = top["person_id"] == ext_id
        threshold_ok = top["score"] >= CROP_50_THRESHOLD
        if match and threshold_ok:
            crop50_ok += 1
            print(
                f"  crop50({ext_id}) -> {top['person_id'][:8]} "
                f"score={top['score']:.3f} "
                f"confirmed={top['confirming_triplets']}  OK  ({elapsed:.0f}ms)",
            )
        else:
            print(
                f"  crop50({ext_id}) -> {top['person_id'][:8]} "
                f"score={top['score']:.3f} "
                f"confirmed={top['confirming_triplets']}  "
                f"{'FAIL (wrong person)' if not match else 'FAIL (low score)'}"
                f"  ({elapsed:.0f}ms)",
            )
    print(f"  50% crop: {crop50_ok}/{len(enrolled)}")

    # ---- Test 3: 25% corner crop ----
    print()
    print("=" * 60)
    print("Test 3: 25% corner crop (threshold > 0.3)")
    print("=" * 60)
    crop25_ok = 0
    for ext_id, img_bytes in enrolled:
        crop_bytes = crop_corner(img_bytes, 0.25)
        t0 = time.monotonic()
        result = mcc.search_by_triplets(
            crop_bytes, top_k=5,
            enrolled_ofs=enrolled_ofs,
        )
        elapsed = (time.monotonic() - t0) * 1000
        latencies.append(elapsed)
        candidates = result.get("candidates", [])
        if not candidates:
            print(f"  crop25({ext_id}) -> NO CANDIDATES  FAIL  ({elapsed:.0f}ms)")
            continue
        top = candidates[0]
        match = top["person_id"] == ext_id
        threshold_ok = top["score"] >= CROP_25_THRESHOLD
        if match and threshold_ok:
            crop25_ok += 1
            print(
                f"  crop25({ext_id}) -> {top['person_id'][:8]} "
                f"score={top['score']:.3f} "
                f"confirmed={top['confirming_triplets']}  OK  ({elapsed:.0f}ms)",
            )
        else:
            print(
                f"  crop25({ext_id}) -> {top['person_id'][:8]} "
                f"score={top['score']:.3f} "
                f"confirmed={top['confirming_triplets']}  "
                f"{'FAIL (wrong person)' if not match else 'FAIL (low score)'}"
                f"  ({elapsed:.0f}ms)",
            )
    print(f"  25% crop: {crop25_ok}/{len(enrolled)}")

    # ---- Summary ----
    max_lat = max(latencies) if latencies else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print()
    print("=" * 60)
    print("Summary (Plan 26-01 acceptance gate)")
    print("=" * 60)
    print(f"  Self-match:    {self_ok}/{len(enrolled)}  (target: 5/5 > 0.5)")
    print(f"  50% crop:      {crop50_ok}/{len(enrolled)}  (target: >=4/5 > 0.4)")
    print(f"  25% crop:      {crop25_ok}/{len(enrolled)}  (target: >=3/5 > 0.3)")
    print(f"  Max latency:   {max_lat:.0f}ms  (target: < 3000ms)")
    print(f"  Avg latency:   {avg_lat:.0f}ms")

    passed = (
        self_ok == len(enrolled)
        and crop50_ok >= 4
        and crop25_ok >= 3
        and max_lat < MAX_LATENCY_MS
    )
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

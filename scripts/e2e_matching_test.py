"""E2E integration test for Phase 24 matching pipeline.

Enrolls 5 SOCOFing persons via the production `create_capture` path
(same as the API), then runs self-match and crop-match tests using
the pair-based search endpoint.

Usage (from apps/backend):
    uv run python ../../scripts/e2e_matching_test.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import cv2
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import config
from src.db.qdrant_mcc_repository import QdrantMccRepository
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


async def main() -> int:
    engine = create_async_engine(config.async_database_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    mcc = MccMatchingService()

    # Clean up old data
    async with engine.begin() as conn:
        for pid in PERSONS:
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

    # Clean Qdrant pair features
    repo = QdrantMccRepository.from_host()
    repo.ensure_pair_collection()
    for pid in PERSONS:
        repo.delete_pairs_by_person(pid)

    # Enroll 5 persons via create_capture
    print("=" * 60)
    print("Enrolling 5 SOCOFing persons via create_capture")
    print("=" * 60)
    enrolled: list[tuple[str, bytes]] = []  # (external_id, image_bytes)

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

    # ---- Self-match ----
    print()
    print("=" * 60)
    print("Test 1: Self-match (pair-based)")
    print("=" * 60)
    self_match_ok = 0
    for ext_id, img_bytes in enrolled:
        candidates = mcc.search_by_pairs(img_bytes, top_k=5).get("candidates", [])
        if candidates and candidates[0]["person_id"] == ext_id:
            print(
                f"  {ext_id} -> {candidates[0]['person_id']} "
                f"(score={candidates[0]['score']:.3f}, "
                f"peak={candidates[0]['peak_votes']})  OK",
            )
            self_match_ok += 1
        else:
            top = candidates[0]["person_id"] if candidates else "NONE"
            print(f"  {ext_id} -> {top}  FAIL")
    print(f"  Self-match: {self_match_ok}/{len(enrolled)}")

    # ---- 50% center crop ----
    print()
    print("=" * 60)
    print("Test 2: 50% center crop -> full enrollment")
    print("=" * 60)
    crop_50_ok = 0
    for ext_id, img_bytes in enrolled:
        crop_bytes = crop_center(img_bytes, 0.5)
        candidates = mcc.search_by_pairs(crop_bytes, top_k=5).get("candidates", [])
        if candidates and candidates[0]["person_id"] == ext_id:
            print(
                f"  crop50({ext_id}) -> {candidates[0]['person_id']} "
                f"(score={candidates[0]['score']:.3f}, "
                f"peak={candidates[0]['peak_votes']})  OK",
            )
            crop_50_ok += 1
        else:
            top = candidates[0]["person_id"] if candidates else "NONE"
            print(f"  crop50({ext_id}) -> {top}  FAIL")
    print(f"  50% crop match: {crop_50_ok}/{len(enrolled)}")

    # ---- 25% corner crop ----
    print()
    print("=" * 60)
    print("Test 3: 25% corner crop -> full enrollment")
    print("=" * 60)
    crop_25_ok = 0
    for ext_id, img_bytes in enrolled:
        crop_bytes = crop_corner(img_bytes, 0.25)
        candidates = mcc.search_by_pairs(crop_bytes, top_k=5).get("candidates", [])
        if candidates and candidates[0]["person_id"] == ext_id:
            print(
                f"  crop25({ext_id}) -> {candidates[0]['person_id']} "
                f"(score={candidates[0]['score']:.3f}, "
                f"peak={candidates[0]['peak_votes']})  OK",
            )
            crop_25_ok += 1
        else:
            top = candidates[0]["person_id"] if candidates else "NONE"
            print(f"  crop25({ext_id}) -> {top}  FAIL")
    print(f"  25% crop match: {crop_25_ok}/{len(enrolled)}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Self-match:    {self_match_ok}/{len(enrolled)}")
    print(f"  50% crop:      {crop_50_ok}/{len(enrolled)}")
    print(f"  25% crop:      {crop_25_ok}/{len(enrolled)}")

    passed = self_match_ok == len(enrolled)
    print(f"\nOVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

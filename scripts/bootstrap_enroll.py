"""One-shot bootstrap: enroll 5 SOCOFing persons via the proper service path.

Uses `FingerprintEnrollmentService.create_capture` (NOT a hand-rolled
mcc.enroll call) so that the `enhanced_image` PNG is persisted in the
fingerprint_captures row — without it, /api/v1/captures/{id}/image
returns 503 and the perito cannot see the candidate's image.

Pre-run cleanup:
  - DELETE persons SOC_0100..SOC_0104 (cascades to fingerprints/captures)
  - DROP Qdrant collection mcc_cylinders (recreated on first enroll)

Usage (from apps/backend):
    uv run python ../../scripts/bootstrap_enroll.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
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
        f"No {FINGER_NAME} BMP found for {person_external_id} (pid={pid}) in {SOCOFING_REAL}",
    )


async def cleanup_pgsql(engine) -> int:
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                DELETE FROM fingerprint_captures
                WHERE fingerprint_id IN (
                    SELECT fp.id FROM fingerprints fp
                    JOIN persons p ON p.id = fp.person_id
                    WHERE p.external_id = ANY(:ids)
                )
                """,
            ),
            {"ids": PERSONS},
        )
        cap_del = result.rowcount or 0

        result = await conn.execute(
            text(
                """
                DELETE FROM fingerprints
                WHERE person_id IN (
                    SELECT id FROM persons WHERE external_id = ANY(:ids)
                )
                """,
            ),
            {"ids": PERSONS},
        )
        fp_del = result.rowcount or 0

        result = await conn.execute(
            text("DELETE FROM persons WHERE external_id = ANY(:ids)"),
            {"ids": PERSONS},
        )
        p_del = result.rowcount or 0
    return p_del, fp_del, cap_del


def cleanup_qdrant() -> bool:
    with httpx.Client(timeout=10) as client:
        r = client.delete("http://localhost:6333/collections/mcc_cylinders")
    return r.status_code in (200, 404)


async def main() -> int:
    engine = create_async_engine(config.async_database_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    mcc = MccMatchingService()

    p_del, fp_del, cap_del = await cleanup_pgsql(engine)
    print(f"PG cleanup: {p_del} personas, {fp_del} fingerprints, {cap_del} captures")
    qdrant_ok = cleanup_qdrant()
    print(f"Qdrant cleanup: {'OK' if qdrant_ok else 'failed'}")

    qdrant_repo = QdrantMccRepository.from_host()
    qdrant_repo.ensure_collection()
    print("Qdrant collection ready")

    enrolled: list[tuple[str, str, int, int]] = []

    try:
        async with Session() as session:
            person_svc = PersonService(session)
            enroll_svc = FingerprintEnrollmentService(session, mcc)
            for external_id in PERSONS:
                person = await person_svc.find_or_create_person(
                    external_id=external_id,
                    full_name=f"Sujeto SOCOFing {external_id.split('_')[1]}",
                    doc_type="cedula",
                    doc_number=f"DOC_{external_id.split('_')[1].zfill(8)}",
                )
                assert person is not None
                await session.commit()
                await session.refresh(person)

                image_path = find_socofing_file(external_id)
                image_bytes = image_path.read_bytes()
                print(f"  {external_id}: {image_path.name} ({len(image_bytes)} bytes)")

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
                    image_bytes=image_bytes,
                    image_dpi=500,
                )
                await session.commit()
                await session.refresh(capture)

                has_img = capture.enhanced_image is not None
                enrolled.append(
                    (
                        external_id,
                        str(capture.id),
                        capture.num_minutiae or 0,
                        len(capture.enhanced_image) if has_img else 0,
                    ),
                )
                print(
                    f"    → capture={str(capture.id)[:8]} "
                    f"minutiae={capture.num_minutiae} "
                    f"enhanced_png={len(capture.enhanced_image) if has_img else 0}B",
                )
    finally:
        await engine.dispose()

    print(f"\nOK {len(enrolled)} personas re-enroladas via FingerprintEnrollmentService")
    bad = [e for e in enrolled if e[3] == 0]
    if bad:
        print(f"  ⚠ {len(bad)} capturas sin enhanced_image: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

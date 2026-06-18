"""Re-enroll 5 SOCOFing persons and verify self-match in the SAME process.

The pipeline (RidgeGraphExtractor + sknw) is non-deterministic across
separate Python processes — the same image produces different minutiae
positions in different runs. So enrollment and query must happen in
the SAME process for the positions to match.

This script:
  1. Cleans up the 5 SOC persons and Qdrant
  2. Re-enrolls them via FingerprintEnrollmentService
  3. Runs the same query test in the same process
  4. Reports which self-matches work

If the issue is truly process-level non-determinism, all 5 should
match after this single-process enrollment. If not, there's still
some intrinsic issue.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from src.core.config import config
from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)
from src.services.mcc_matching_service import MccMatchingService
from src.services.person_service import PersonService
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

PERSONS = ["SOC_0100", "SOC_0101", "SOC_0102", "SOC_0103", "SOC_0104"]
SOCOFING_REAL = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "backend"
    / "static"
    / "SOCOFing"
    / "Real"
)


def find_socofing_file(person_external_id: str) -> Path:
    pid = person_external_id.replace("SOC_", "").lstrip("0")
    for path in sorted(SOCOFING_REAL.glob(f"{pid}__*_index_finger.BMP")):
        return path
    raise FileNotFoundError(person_external_id)


async def cleanup_pgsql(engine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM fingerprint_captures WHERE fingerprint_id IN "
                "(SELECT fp.id FROM fingerprints fp JOIN persons p ON p.id=fp.person_id "
                "WHERE p.external_id = ANY(:ids))",
            ),
            {"ids": PERSONS},
        )
        await conn.execute(
            text(
                "DELETE FROM fingerprints WHERE person_id IN "
                "(SELECT id FROM persons WHERE external_id = ANY(:ids))",
            ),
            {"ids": PERSONS},
        )
        await conn.execute(
            text("DELETE FROM persons WHERE external_id = ANY(:ids)"),
            {"ids": PERSONS},
        )


def cleanup_qdrant() -> None:
    import httpx
    with httpx.Client(timeout=10) as c:
        c.delete("http://localhost:6333/collections/mcc_cylinders")


async def main() -> int:
    engine = create_async_engine(config.async_database_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    mcc = MccMatchingService()

    await cleanup_pgsql(engine)
    cleanup_qdrant()
    repo = QdrantMccRepository.from_host()
    repo.ensure_collection()

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
            await session.commit()
            await session.refresh(person)

            image_path = find_socofing_file(external_id)
            image_bytes = image_path.read_bytes()

            slot = await FingerprintRepository.create(
                session,
                person_id=person.id,
                finger_position=0,
                capture_type="rolled",
            )
            await session.commit()
            await session.refresh(slot)

            capture, _ = await enroll_svc.create_capture(
                fingerprint_id=slot.id,
                image_bytes=image_bytes,
                image_dpi=500,
            )
            await session.commit()
            await session.refresh(capture)
            print(f"  enrolled {external_id} capture={str(capture.id)[:8]}")

        # SAME-PROCESS query test
        print("\n=== Self-match in same process ===")
        for external_id in PERSONS:
            image_path = find_socofing_file(external_id)
            image_bytes = image_path.read_bytes()
            probe, hits = mcc.search(image_bytes, top_k=3)
            if not hits:
                winner = "NO_MATCH"
                score = 0.0
            else:
                winner = hits[0].person_id
                score = hits[0].total_score
            ok = winner == external_id
            marker = "OK" if ok else "XX"
            print(f"  {marker} {external_id} -> {winner} (score={score:.3f})")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

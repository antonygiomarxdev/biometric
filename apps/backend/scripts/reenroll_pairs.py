"""Re-enroll all SOCOFing Real fingerprints into pair_features collection.

Wipes the existing pair_features collection first, then re-enrolls
every Real image via the MccMatchingService.enroll_pairs pipeline.

Usage:

    uv run python scripts/reenroll_pairs.py

NOTE: Requires PostgreSQL + Qdrant to be running.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.core.config import config
from src.db.qdrant_mcc_repository import QdrantMccRepository, PAIR_COLLECTION_NAME
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.services.mcc_matching_service import MccMatchingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reenroll_pairs")

SOCOFING_REAL = Path(__file__).resolve().parent.parent / "static" / "SOCOFing" / "Real"

# NIST FGP codes: 0=unknown, 1=R_thumb, 2=R_index, 3=R_middle,
# 4=R_ring, 5=R_little, 6=L_thumb, 7=L_index, 8=L_middle,
# 9=L_ring, 10=L_little
_FINGER_FGP: dict[str, int] = {
    "Right_thumb": 1, "Right_index": 2, "Right_middle": 3,
    "Right_ring": 4, "Right_little": 5,
    "Left_thumb": 6, "Left_index": 7, "Left_middle": 8,
    "Left_ring": 9, "Left_little": 10,
}


def _parse_socofing_filename(stem: str) -> dict:
    parts = stem.split("__")
    person_id = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    hand_finger = rest.rsplit("_finger", 1)[0] if "_finger" in rest else rest
    fgp = _FINGER_FGP.get(hand_finger, 0)
    return {"person_id": person_id, "fgp": fgp}


async def _reenroll() -> None:
    mcc_service = MccMatchingService()
    repo = QdrantMccRepository.from_host()

    collections = repo._client.get_collections().collections
    existing = [c.name for c in collections]
    if PAIR_COLLECTION_NAME in existing:
        logger.info("Dropping existing %s collection...", PAIR_COLLECTION_NAME)
        repo._client.delete_collection(PAIR_COLLECTION_NAME)

    repo.ensure_pair_collection()
    # ensure the pair collection exists for the service as well
    mcc_service._mcc_repo.ensure_pair_collection()

    all_files = sorted(SOCOFING_REAL.glob("*.BMP")) + sorted(SOCOFING_REAL.glob("*.bmp")) + sorted(SOCOFING_REAL.glob("*.png"))
    # Only enroll subjects 1-5 (for the benchmark)
    image_files = [f for f in all_files if f.stem.split("__")[0].lstrip("0").isdigit() and 1 <= int(f.stem.split("__")[0].lstrip("0")) <= 5]
    if not image_files:
        logger.warning("No images found in %s", SOCOFING_REAL)
        return

    logger.info("Found %d images to enroll (subjects 1-5)", len(image_files))

    total_pairs = 0

    async_engine = create_async_engine(config.async_database_url)
    async_session_factory = async_sessionmaker(
        bind=async_engine,
        expire_on_commit=False,
    )
    async with async_session_factory() as session:

        for img_path in image_files:
            stem = img_path.stem
            info = _parse_socofing_filename(stem)

            person_ext_id = info["person_id"]
            person = await PersonRepository.find_by_external_id(session, person_ext_id)
            if person is None:
                from src.db.models import Person as PersonModel
                person = PersonModel(
                    external_id=person_ext_id,
                    full_name=f"SOCOFing Person {person_ext_id}",
                )
                session.add(person)
                await session.commit()
                await session.refresh(person)
                logger.info("Created person %s (ext_id=%s)", person.id, person_ext_id)

            person_id = str(person.id)

            fgp = info["fgp"]
            fp = await FingerprintRepository.find_slot(
                session, person.id, fgp, "rolled",
            )
            if fp is None:
                fp = await FingerprintRepository.create(
                    session, person_id=person.id, finger_position=fgp,
                )
                logger.info("Created fingerprint %s for person %s", fp.id, person.id)

            capture_id = f"reenroll_pairs_{img_path.stem}"
            image_bytes = img_path.read_bytes()

            loop = asyncio.get_running_loop()
            num_pairs = await loop.run_in_executor(
                None,
                mcc_service.enroll_pairs,
                capture_id,
                str(fp.id),
                person_id,
                image_bytes,
            )
            total_pairs += num_pairs
            logger.info("  %s → %d pairs", img_path.name, num_pairs)

        logger.info(
            "\nDone. Enrolled %d images → %d total pairs",
            len(image_files), total_pairs,
        )


def main() -> None:
    asyncio.run(_reenroll())


if __name__ == "__main__":
    main()

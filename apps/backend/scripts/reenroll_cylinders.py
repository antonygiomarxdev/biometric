"""Re-enroll all SOCOFing Real fingerprints into mcc_cylinders collection.

Wipes the existing mcc_cylinders collection first, then re-enrolls
every Real image via the MccMatchingService.enroll pipeline.

Usage:

    uv run python scripts/reenroll_cylinders.py

NOTE: Requires PostgreSQL + Qdrant to be running.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.core.config import config
from src.db.qdrant_mcc_repository import COLLECTION_NAME
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.services.mcc_matching_service import MccMatchingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reenroll_cylinders")

SOCOFING_REAL = Path(__file__).resolve().parent.parent / "static" / "SOCOFing" / "Real"


def _parse_socofing_filename(stem: str) -> dict:
    parts = stem.split("__")
    person_id = parts[0]
    return {"person_id": person_id}


async def _reenroll() -> None:
    mcc_service = MccMatchingService()

    collections = mcc_service._mcc_repo._client.get_collections().collections
    existing = [c.name for c in collections]
    if COLLECTION_NAME in existing:
        logger.info("Dropping existing %s collection...", COLLECTION_NAME)
        mcc_service._mcc_repo._client.delete_collection(COLLECTION_NAME)

    mcc_service._mcc_repo.ensure_collection()

    all_files = sorted(SOCOFING_REAL.glob("*.BMP")) + sorted(SOCOFING_REAL.glob("*.bmp")) + sorted(SOCOFING_REAL.glob("*.png"))
    image_files = [
        f for f in all_files
        if f.stem.split("__")[0].lstrip("0").isdigit()
        and (1 <= int(f.stem.split("__")[0].lstrip("0")) <= 5 or int(f.stem.split("__")[0].lstrip("0")) == 100)
    ]
    if not image_files:
        logger.warning("No images found in %s", SOCOFING_REAL)
        return

    logger.info("Found %d images to enroll (subjects 1-5 + 100)", len(image_files))

    total_cylinders = 0

    async_engine = create_async_engine(config.async_database_url)
    async_session_factory = async_sessionmaker(bind=async_engine, expire_on_commit=False)
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

            fgp = 2  # default to index
            fp = await FingerprintRepository.find_slot(session, person.id, fgp, "rolled")
            if fp is None:
                fp = await FingerprintRepository.create(
                    session, person_id=person.id, finger_position=fgp,
                )
                logger.info("Created fingerprint %s for person %s", fp.id, person.id)

            capture_id = f"reenroll_cyl_{img_path.stem}"
            image_bytes = img_path.read_bytes()

            loop = asyncio.get_running_loop()
            num_cyl = await loop.run_in_executor(
                None,
                mcc_service.enroll,
                capture_id,
                str(fp.id),
                person_id,
                image_bytes,
            )
            total_cylinders += num_cyl
            logger.info("  %s → %d cylinders", img_path.name, num_cyl)

        logger.info(
            "\nDone. Enrolled %d images → %d total cylinders",
            len(image_files), total_cylinders,
        )


def main() -> None:
    asyncio.run(_reenroll())


if __name__ == "__main__":
    main()

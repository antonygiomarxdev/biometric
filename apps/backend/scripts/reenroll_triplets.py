"""Re-enroll all SOCOFing Real fingerprints into triplet_features collection.

Wipes the existing triplet_features collection first, then re-enrolls
every Real image via the full MccMatchingService pipeline (quality scoring
+ triplet extraction + Qdrant insertion).

Usage:

    uv run python scripts/reenroll_triplets.py

NOTE: Requires PostgreSQL + Qdrant to be running.  Uses the service layer,
so person records are created via PersonService.find_or_create_person
just like the regular enrollment path.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import config
from src.db.database import get_db
from src.db.qdrant_mcc_repository import QdrantMccRepository, TRIPLET_COLLECTION_NAME
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.db.repositories.person_repository import PersonRepository
from src.services.mcc_matching_service import MccMatchingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reenroll_triplets")

SOCOFING_REAL = Path(__file__).resolve().parent.parent / "static" / "SOCOFing" / "Real"

# Mapping from SOCOFing filename stem to person info
# Format: <id>__<Left|Right>_<finger>_<variant>
# e.g., "1__Left_thumb_finger" → person "1", finger "Left_thumb"
FINGER_MAP: dict[str, str] = {
    "little": "little",
    "ring": "ring",
    "middle": "middle",
    "index": "index",
    "thumb": "thumb",
}


def _parse_socofing_filename(stem: str) -> dict:
    """Extract person_id, hand, finger from a SOCOFing filename stem."""
    parts = stem.split("__")
    person_id = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    hand_finger = rest.rsplit("_finger", 1)[0] if "_finger" in rest else rest
    hand = "left" if "Left" in hand_finger else "right"
    finger = "unknown"
    for key, val in FINGER_MAP.items():
        if key in hand_finger.lower():
            finger = val
            break
    return {"person_id": person_id, "hand": hand, "finger": finger}


async def _reenroll() -> None:
    mcc_service = MccMatchingService()
    repo = QdrantMccRepository.from_host()

    # Wipe existing triplet_features collection
    collections = repo._client.get_collections().collections
    existing = [c.name for c in collections]
    if TRIPLET_COLLECTION_NAME in existing:
        logger.info("Dropping existing %s collection...", TRIPLET_COLLECTION_NAME)
        repo._client.delete_collection(TRIPLET_COLLECTION_NAME)

    # Re-create empty
    repo.ensure_triplet_collection()

    # Collect image files
    image_files = sorted(SOCOFING_REAL.glob("*.bmp")) + sorted(SOCOFING_REAL.glob("*.png"))
    if not image_files:
        logger.warning("No images found in %s", SOCOFING_REAL)
        return

    logger.info("Found %d images to enroll", len(image_files))

    total_triplets = 0
    total_minutiae = 0

    async for session in get_db():
        session: AsyncSession
        break

    for img_path in image_files:
        stem = img_path.stem
        info = _parse_socofing_filename(stem)

        # Find or create person
        person_ext_id = info["person_id"]
        person = await PersonRepository.get_by_external_id(session, person_ext_id)
        if person is None:
            # PersonService has a find_or_create_person method, but it requires
            # the seeded SOCOFing person lookup. If not seeded, create manually.
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

        # Create a fingerprint slot
        finger_name = f"{info['hand']}_{info['finger']}"
        fp = await FingerprintRepository.get_by_person_and_name(
            session, person.id, finger_name,
        )
        if fp is None:
            from src.db.models import Fingerprint as FingerprintModel
            fp = FingerprintModel(
                person_id=person.id,
                finger_name=finger_name,
            )
            session.add(fp)
            await session.commit()
            await session.refresh(fp)
            logger.info("Created fingerprint %s for person %s", fp.id, person.id)

        capture_id = f"reenroll_{img_path.stem}"
        image_bytes = img_path.read_bytes()

        loop = asyncio.get_running_loop()
        num_min, num_trip = await loop.run_in_executor(
            None,
            mcc_service.enroll_triplets,
            capture_id,
            str(fp.id),
            person_id,
            image_bytes,
        )
        total_minutiae += num_min
        total_triplets += num_trip
        logger.info(
            "  %s → %d minutiae, %d triplets (total: %d triplets)",
            img_path.name, num_min, num_trip, total_triplets,
        )

    logger.info(
        "\nDone. Enrolled %d images → %d total minutiae, %d total triplets",
        len(image_files), total_minutiae, total_triplets,
    )


def main() -> None:
    asyncio.run(_reenroll())


if __name__ == "__main__":
    main()

"""Seed Person records from SOCOFing Real subset (Phase 23).

Replaces the legacy ``scripts/load_socofing.py`` (which used removed
APIs ``db_manager.create_tables`` and ``repository.register``) with
a focused, idempotent script that creates ONLY ``Person`` records.

Per D-20/D-21/D-22:
  * Reads ``apps/backend/static/SOCOFing/Real/`` (6000 BMPs).
  * Deduplicates by person_id (numeric prefix) -> 600 unique subjects.
  * Uses ``PersonService.find_or_create_person(external_id=...)`` so
    re-running is a no-op.
  * Does NOT insert Fingerprints (those are enrolled interactively
    via the UI; see /enroll page).

Usage::

    # From the project root:
    cd apps/backend && uv run python ../../scripts/seed_socofing.py
    # Or with a limit for local testing:
    cd apps/backend && uv run python ../../scripts/seed_socofing.py --limit 50
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import config
from src.services.person_service import PersonService

logger = logging.getLogger(__name__)

# Resolve SOCOFing Real directory relative to this script (portable across CWDs).
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
SOCOFING_REAL = _REPO_ROOT / "apps" / "backend" / "static" / "SOCOFing" / "Real"

# Filename regex per D-20 / RESEARCH §Pattern 6:
#   100__M_Left_index_finger.BMP
#   101__F_Right_thumb_finger.BMP
# Group pid is the numeric subject id; gender is M or F.
FILENAME_RE = re.compile(
    r"^(?P<pid>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_(?P<finger>[a-z]+_finger)\.BMP$"
)


def collect_subject_ids(limit: int | None) -> list[str]:
    """Read SOCOFing Real filenames; return sorted list of unique subject ids.

    Args:
        limit: If set, stop after collecting this many unique subjects.

    Returns:
        Sorted list of zero-padded 4-digit subject ids (e.g. ``"0100"``).
    """
    if not SOCOFING_REAL.exists():
        raise SystemExit(
            f"SOCOFing Real directory not found: {SOCOFING_REAL}. "
            "Verify the dataset is mounted under apps/backend/static/SOCOFing/Real/."
        )

    seen: set[str] = set()
    for path in sorted(SOCOFING_REAL.glob("*.BMP")):
        match = FILENAME_RE.match(path.name)
        if match is None:
            logger.debug("Skipping non-matching filename: %s", path.name)
            continue
        seen.add(match["pid"].zfill(4))
        if limit is not None and len(seen) >= limit:
            break
    return sorted(seen)


async def seed(limit: int | None) -> int:
    """Idempotently create Person records for each SOCOFing subject.

    Returns the count of persons created (existing persons are skipped,
    so re-running returns 0).
    """
    subject_ids = collect_subject_ids(limit)
    if not subject_ids:
        logger.warning("No subject ids found in %s", SOCOFING_REAL)
        return 0
    logger.info("Found %d unique subjects in SOCOFing/Real", len(subject_ids))

    engine = create_async_engine(config.async_database_url)
    session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False,
    )
    created = 0
    try:
        async with session_factory() as session:
            svc = PersonService(session)
            for pid in subject_ids:
                ext_id = f"SOC_{pid}"
                try:
                    person = await svc.find_or_create_person(
                        external_id=ext_id,
                        full_name=f"Sujeto SOCOFing {pid}",
                        doc_type="cedula",
                        doc_number=f"DOC_{pid.zfill(8)}",
                    )
                except Exception as exc:
                    logger.error("Failed to seed person %s: %s", ext_id, exc)
                    continue
                if person is not None:
                    # find_or_create_person returns the Person either way;
                    # increment only if it was just inserted. Use
                    # created_at == updated_at as the heuristic.
                    if person.created_at == person.updated_at:
                        created += 1
            await session.commit()
    finally:
        await engine.dispose()

    logger.info(
        "Done. %d new persons created, %d already existed.",
        created,
        len(subject_ids) - created,
    )
    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed Person records from SOCOFing Real subset (Phase 23).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N subjects (for local testing); default = all 600.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for filename parsing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    n = asyncio.run(seed(args.limit))
    print(f"Done. {n} new persons seeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

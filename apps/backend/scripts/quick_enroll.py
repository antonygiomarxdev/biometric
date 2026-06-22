#!/usr/bin/env python3
"""Batch enroll SOCOFing Real fingerprints via REST API.

Usage:
    uv run python scripts/quick_enroll.py [--concurrency N]

Default concurrency is 16. The script:

  1. Calls POST /api/v1/persons/         (create or fetch by external_id)
  2. Calls POST /persons/{id}/fingerprints (idempotent slot creation)
  3. Calls POST /fingerprints/{id}/captures (uploads image; backend
     computes AFR-Net embedding and upserts to Qdrant)

Parsing: filename ``30__F_Left_thumb_finger.BMP`` →
external_id='30', finger_name='F_Left_thumb_finger'.

The script uses ``asyncio.Semaphore`` to keep N HTTP requests in
flight at once.  The backend is event-driven: ``create_capture`` is
non-blocking, every CPU step is dispatched to a dedicated
``ThreadPoolExecutor``, and the inference path is serialised by an
``asyncio.Lock`` on the shared model.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = os.getenv("QUICK_ENROLL_BASE_URL", "http://localhost:8765/api/v1")
SOCOFING_REAL = Path(__file__).parent.parent / "static" / "SOCOFing" / "Real"
if not SOCOFING_REAL.exists():
    SOCOFING_REAL = Path(
        "/home/ksante/dev/biometric/apps/backend/static/SOCOFing/Real"
    )

FINGER_POSITIONS = {
    "Right_thumb": 1, "Right_index": 2, "Right_middle": 3,
    "Right_ring": 4, "Right_little": 5,
    "Left_thumb": 6, "Left_index": 7, "Left_middle": 8,
    "Left_ring": 9, "Left_little": 10,
}


def _finger_position_from_name(hand_finger: str) -> int:
    """Match SOCOFing hand+name (e.g. ``M_Left_index_finger``) to a NIST FGP code.

    The filename pattern is ``M|F_<Hand>_<finger>_finger`` so we
    try every key in ``FINGER_POSITIONS`` and return the first one
    that appears as a substring.  Falls back to 0 (unknown) when
    the name doesn't match.
    """
    for name, code in FINGER_POSITIONS.items():
        if name in hand_finger:
            return code
    return 0


def parse_filename(stem: str) -> tuple[str, str, int]:
    parts = stem.split("__", 2)
    person_ext_id = parts[0]
    hand_finger = parts[1] if len(parts) > 1 else "Unknown"
    finger_pos = _finger_position_from_name(hand_finger)
    return person_ext_id, hand_finger, finger_pos


async def get_or_create_person(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    person_ext_id: str,
    cache: dict[str, str],
) -> str:
    async with semaphore:
        if person_ext_id in cache:
            return cache[person_ext_id]
        resp = await client.post(
            "/persons/",
            json={
                "external_id": person_ext_id,
                "full_name": f"Person {person_ext_id}",
            },
        )
        if resp.status_code in (200, 201):
            pid = resp.json()["id"]
        elif resp.status_code == 409:
            r = await client.get(f"/persons/by-external-id/{person_ext_id}")
            r.raise_for_status()
            pid = r.json()["id"]
        else:
            resp.raise_for_status()
            msg = f"Failed to create person {person_ext_id}: {resp.text}"
            raise RuntimeError(msg)
        cache[person_ext_id] = pid
        return pid


async def create_fingerprint_slot(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    person_id: str,
    finger_pos: int,
) -> str:
    async with semaphore:
        resp = await client.post(
            f"/persons/{person_id}/fingerprints",
            json={"finger_position": finger_pos, "capture_type": "rolled"},
        )
        resp.raise_for_status()
        return resp.json()["id"]


async def upload_capture(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    fingerprint_id: str,
    image_path: Path,
) -> int:
    async with semaphore:
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/bmp")}
            resp = await client.post(
                f"/fingerprints/{fingerprint_id}/captures",
                files=files,
                data={"is_exemplar": "true"},
            )
        return resp.status_code


async def enroll_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    person_cache: dict[str, str],
    path: Path,
) -> bool:
    person_ext_id, _finger_name, finger_pos = parse_filename(path.stem)
    try:
        person_id = await get_or_create_person(client, semaphore, person_ext_id, person_cache)
        slot_id = await create_fingerprint_slot(client, semaphore, person_id, finger_pos)
        status = await upload_capture(client, semaphore, slot_id, path)
        return status == 201
    except Exception as exc:
        logger.warning("  Failed %s: %s", path.name, exc)
        return False


async def run(bmp_files: list[Path], concurrency: int) -> tuple[int, int]:
    semaphore = asyncio.Semaphore(concurrency)
    person_cache: dict[str, str] = {}
    enrolled = 0
    failed = 0
    counter_lock = asyncio.Lock()

    timeout = httpx.Timeout(60.0, connect=10.0)
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(
        base_url=BASE_URL, timeout=timeout, limits=limits,
    ) as client:
        async def _one(path: Path) -> None:
            nonlocal enrolled, failed
            ok = await enroll_one(client, semaphore, person_cache, path)
            async with counter_lock:
                if ok:
                    enrolled += 1
                else:
                    failed += 1

        tasks = [asyncio.create_task(_one(p)) for p in bmp_files]
        t0 = time.monotonic()
        for done_count, fut in enumerate(asyncio.as_completed(tasks), start=1):
            await fut
            if done_count % 500 == 0 or done_count == len(tasks):
                elapsed = time.monotonic() - t0
                rate = done_count / elapsed
                logger.info(
                    "  %d/%d done (%.1f img/s, %d ok / %d failed)",
                    done_count, len(tasks), rate, enrolled, failed,
                )

    return enrolled, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--concurrency", "-c", type=int, default=16,
        help="Max concurrent HTTP requests (default: 16)",
    )
    args = parser.parse_args()

    bmp_files = sorted(SOCOFING_REAL.glob("*.BMP"))
    if not bmp_files:
        logger.error("No BMP files found in %s", SOCOFING_REAL)
        sys.exit(1)

    logger.info(
        "Enrolling %d images from %s via %s (concurrency=%d)...",
        len(bmp_files), SOCOFING_REAL, BASE_URL, args.concurrency,
    )
    t_start = time.monotonic()
    enrolled, failed = asyncio.run(run(bmp_files, args.concurrency))
    total = time.monotonic() - t_start
    rate = (enrolled + failed) / total if total else 0.0
    logger.info(
        "Done: %d enrolled, %d failed, %.1f min, %.1f img/s",
        enrolled, failed, total / 60, rate,
    )


if __name__ == "__main__":
    main()

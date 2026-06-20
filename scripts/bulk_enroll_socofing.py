"""Bulk-enroll all 600 SOCOFing subjects (×10 fingers) — parallel workers.

Creates Fingerprint slots and FingerprintCaptures for every Real SOCOFing
image (6000 BMPs) using ``FingerprintEnrollmentService.create_capture``,
persisting minutiae in ``capture_minutiae`` and index pairs in Qdrant.

Idempotent — safe to re-run / resume after interruption. Skips existing
captures (matched by SHA-256 of the source image).

Usage (from ``apps/backend``)::

    uv run python ../../scripts/bulk_enroll_socofing.py              # all 600 subjects
    uv run python ../../scripts/bulk_enroll_socofing.py --workers 8  # 8 parallel workers
    uv run python ../../scripts/bulk_enroll_socofing.py --limit 10   # first 10 subjects only

Finger naming (SOCOFing → NIST FGP)::

    Right_thumb  → 0   Left_thumb   → 5
    Right_index  → 1   Left_index   → 6
    Right_middle → 2   Left_middle  → 7
    Right_ring   → 3   Left_ring    → 8
    Right_little → 4   Left_little  → 9
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import config
from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository
from src.services.fingerprint_enrollment_service import (
    FingerprintEnrollmentService,
)
from src.services.mcc_matching_service import MccMatchingService
from src.services.person_service import PersonService

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
SOCOFING_REAL = _REPO_ROOT / "apps" / "backend" / "static" / "SOCOFing" / "Real"

_DEFAULT_WORKERS = os.cpu_count() or 4

# ---------------------------------------------------------------------------
# SOCOFing filename → NIST FGP mapping
# ---------------------------------------------------------------------------

FILENAME_RE = re.compile(
    r"^(?P<pid>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_(?P<finger>[a-z]+_finger)\.BMP$",
)

FGP_MAP: dict[str, int] = {
    "Right_thumb_finger": 0,
    "Right_index_finger": 1,
    "Right_middle_finger": 2,
    "Right_ring_finger": 3,
    "Right_little_finger": 4,
    "Left_thumb_finger": 5,
    "Left_index_finger": 6,
    "Left_middle_finger": 7,
    "Left_ring_finger": 8,
    "Left_little_finger": 9,
}

FINGER_LABELS: dict[int, str] = {
    0: "pulgar derecho",
    1: "índice derecho",
    2: "medio derecho",
    3: "anular derecho",
    4: "meñique derecho",
    5: "pulgar izquierdo",
    6: "índice izquierdo",
    7: "medio izquierdo",
    8: "anular izquierdo",
    9: "meñique izquierdo",
}


@dataclass
class SocofingImage:
    pid: str
    pid_z4: str
    gender: str
    hand: str
    finger: str
    fgp: int
    path: Path


def collect_images(
    limit_subjects: int | None = None,
    offset_subjects: int = 0,
) -> list[SocofingImage]:
    if not SOCOFING_REAL.exists():
        raise SystemExit(
            f"SOCOFing Real directory not found: {SOCOFING_REAL}. "
            "Verify the dataset is mounted under apps/backend/static/SOCOFing/Real/."
        )

    pid_order: list[str] = []
    pid_images: dict[str, list[SocofingImage]] = {}

    for path in sorted(SOCOFING_REAL.glob("*.BMP")):
        m = FILENAME_RE.match(path.name)
        if m is None:
            continue
        pid = m["pid"]
        if pid not in pid_order:
            pid_order.append(pid)
        img = _make_image(m, path)
        if img is not None:
            pid_images.setdefault(pid, []).append(img)

    selected_pids = pid_order[offset_subjects:]
    if limit_subjects is not None:
        selected_pids = selected_pids[:limit_subjects]

    images: list[SocofingImage] = []
    for pid in selected_pids:
        images.extend(pid_images.get(pid, []))
    return images


def _make_image(m: re.Match, path: Path) -> SocofingImage | None:
    hand_finger = f"{m['hand']}_{m['finger']}"
    fgp = FGP_MAP.get(hand_finger)
    if fgp is None:
        log.warning("Unrecognised hand/finger combo: %s → %s", path.name, hand_finger)
        return None
    return SocofingImage(
        pid=m["pid"],
        pid_z4=m["pid"].zfill(4),
        gender=m["gender"],
        hand=m["hand"],
        finger=m["finger"],
        fgp=fgp,
        path=path,
    )


# ---------------------------------------------------------------------------
# Per-finger enrollment
# ---------------------------------------------------------------------------

@dataclass
class EnrollResult:
    external_id: str
    fgp: int
    finger_label: str
    status: str           # "ok" | "skip" | "error"
    capture_id: str = ""
    minutiae: int = 0
    error: str = ""


async def enroll_one_finger(
    session: AsyncSession,
    person_svc: PersonService,
    enroll_svc: FingerprintEnrollmentService,
    img: SocofingImage,
    *,
    dry_run: bool = False,
) -> EnrollResult:
    ext_id = f"SOC_{img.pid_z4}"
    person = await person_svc.find_or_create_person(
        external_id=ext_id,
        full_name=f"Sujeto SOCOFing {img.pid_z4}",
        doc_type="cedula",
        doc_number=f"DOC_{img.pid_z4.zfill(8)}",
        sex=img.gender,
    )
    if person is None:
        return EnrollResult(ext_id, img.fgp, FINGER_LABELS[img.fgp], "error", error="person not found")

    slot = await FingerprintRepository.find_slot(session, person.id, img.fgp, "rolled")
    if slot is None:
        slot = await FingerprintRepository.create(
            session,
            person_id=person.id,
            finger_position=img.fgp,
            capture_type="rolled",
        )

    image_bytes = img.path.read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    existing = await FingerprintCaptureRepository.find_by_image_hash(session, image_hash)
    if existing is not None:
        return EnrollResult(
            ext_id, img.fgp, FINGER_LABELS[img.fgp], "skip",
            capture_id=str(existing.id),
            minutiae=existing.num_minutiae or 0,
        )

    if dry_run:
        return EnrollResult(ext_id, img.fgp, FINGER_LABELS[img.fgp], "skip", error="dry-run")

    try:
        capture, _graphs = await enroll_svc.create_capture(
            fingerprint_id=slot.id,
            image_bytes=image_bytes,
            image_dpi=500,
        )
        return EnrollResult(
            ext_id, img.fgp, FINGER_LABELS[img.fgp], "ok",
            capture_id=str(capture.id),
            minutiae=capture.num_minutiae or 0,
        )
    except Exception as exc:
        log.exception("Failed to enroll %s finger %s: %s", ext_id, FINGER_LABELS[img.fgp], exc)
        return EnrollResult(ext_id, img.fgp, FINGER_LABELS[img.fgp], "error", error=str(exc)[:120])


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    worker_id: int
    results: list[EnrollResult]


async def worker(
    worker_id: int,
    subject_ids: list[str],
    by_subject: dict[str, list[SocofingImage]],
    session_factory: async_sessionmaker,
    *,
    dry_run: bool = False,
) -> WorkerResult:
    """Process a chunk of subjects serially, each with its own DB session + Qdrant client."""
    worker_results: list[EnrollResult] = []
    async with session_factory() as session:
        mcc = MccMatchingService()
        person_svc = PersonService(session)
        enroll_svc = FingerprintEnrollmentService(session, mcc)

        for pid in subject_ids:
            subject_images = by_subject[pid]
            n_sub = len(subject_images)
            ok_count = 0
            skip_count = 0
            err_count = 0

            for img in subject_images:
                r = await enroll_one_finger(
                    session, person_svc, enroll_svc, img, dry_run=dry_run,
                )
                worker_results.append(r)
                if r.status == "ok":
                    ok_count += 1
                elif r.status == "skip":
                    skip_count += 1
                else:
                    err_count += 1

            print(
                f"[W{worker_id}] SOC_{pid} "
                f"✅{ok_count} ⏭️{skip_count} ❌{err_count} "
                f"({n_sub} dedos)",
                flush=True,
            )

    return WorkerResult(worker_id=worker_id, results=worker_results)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def main(
    limit_subjects: int | None,
    offset_subjects: int,
    dry_run: bool,
    num_workers: int,
) -> int:
    images = collect_images(limit_subjects, offset_subjects)

    by_subject: dict[str, list[SocofingImage]] = {}
    for img in images:
        by_subject.setdefault(img.pid_z4, []).append(img)

    subject_ids = sorted(by_subject.keys())
    n_subjects = len(subject_ids)
    n_images = len(images)

    print(f"Found {n_images} images across {n_subjects} subjects in {SOCOFING_REAL}")
    print(f"Workers: {num_workers}")
    if dry_run:
        print("DRY RUN — no captures will be created")
    print()

    # Chunk subjects among workers (contiguous slices so logs are grouped)
    chunks: list[list[str]] = [[] for _ in range(num_workers)]
    for i, pid in enumerate(subject_ids):
        chunks[i % num_workers].append(pid)

    # Raise pool size to handle concurrent workers + headroom
    pool_size = max(num_workers + 2, 10)
    engine = create_async_engine(
        config.async_database_url,
        pool_size=pool_size,
        max_overflow=pool_size,
    )
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    t_start = time.monotonic()
    all_results: list[EnrollResult] = []

    try:
        tasks = [
            worker(wid, chunks[wid], by_subject, Session, dry_run=dry_run)
            for wid in range(num_workers)
            if chunks[wid]
        ]
        worker_results = await asyncio.gather(*tasks)
        for wr in worker_results:
            all_results.extend(wr.results)
    finally:
        await engine.dispose()

    elapsed = time.monotonic() - t_start

    ok = sum(1 for r in all_results if r.status == "ok")
    skipped = sum(1 for r in all_results if r.status == "skip")
    errors = sum(1 for r in all_results if r.status == "error")
    print(f"\n{'='*60}")
    print(f"Enrollment complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  ✅ Enrolled:   {ok}")
    print(f"  ⏭️  Skipped:   {skipped}")
    print(f"  ❌ Errors:     {errors}")
    print(f"  📸 Total:      {len(all_results)}")
    print(f"  👤 Subjects:   {n_subjects}")
    print(f"  ⚡ Workers:    {num_workers}")
    if errors:
        print(f"\n⚠️  {errors} errors — inspect logs above for details")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bulk-enroll all 600 SOCOFing subjects (×10 fingers) — parallel.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit to N subjects (for testing); default = all 600.",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip first N subjects (to resume from a point).",
    )
    parser.add_argument(
        "--workers", type=int, default=_DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {_DEFAULT_WORKERS} = all CPUs).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be enrolled without creating anything.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    sys.exit(asyncio.run(main(
        limit_subjects=args.limit,
        offset_subjects=args.offset,
        dry_run=args.dry_run,
        num_workers=args.workers,
    )))

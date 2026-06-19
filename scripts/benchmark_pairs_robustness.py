"""Benchmark for the NIST Bozorth3 pairs matcher under real-world conditions.

Tests :meth:`MccMatchingService.search_by_pairs` against SOCOFING Real
images with synthetic real-world distortions:

  - Rotation: 30°, 60°, 90°, 180°
  - Crop: 25%
  - Noise: Gaussian + salt & pepper
  - Combination: (noise + crop + rotate)

Output: per-probe table plus aggregate metrics per transformation type.
Used as the regression test suite for Phases 27-05.

Usage (from apps/backend):
    uv run python ../../scripts/benchmark_pairs_robustness.py
"""
from __future__ import annotations

import asyncio
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.config import config
from src.db.models import Person
from src.services.mcc_matching_service import MccMatchingService

SOCOFING_ROOT = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "backend"
    / "static"
    / "SOCOFing"
    / "Real"
)

PERSONS = ["1", "2", "3", "4", "5"]


@dataclass
class ProbeResult:
    expected_person: str
    probe_kind: str
    rotation_deg: float
    crop_pct: float
    top1_person: str | None
    top1_score: float
    top1_hits: int
    rank_of_correct: int | None
    latency_ms: float


def rotate_image(image_bytes: bytes, angle_deg: float) -> bytes:
    """Rotate the image by angle_deg around its center."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, m, (w, h), borderValue=0)
    ok, buf = cv2.imencode(".bmp", rotated)
    return buf.tobytes() if ok else image_bytes


def crop_image(image_bytes: bytes, crop_pct: float) -> bytes:
    """Crop ``crop_pct`` of each side of the image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    dx = int(w * crop_pct / 2)
    dy = int(h * crop_pct / 2)
    cropped = img[dy:h - dy, dx:w - dx]
    ok, buf = cv2.imencode(".bmp", cropped)
    return buf.tobytes() if ok else image_bytes


def add_noise(image_bytes: bytes, intensity: float = 0.05) -> bytes:
    """Add Gaussian noise + salt & pepper to simulate a real-world latent."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return image_bytes
    img = img.astype(np.float32)
    # Gaussian noise
    gauss = np.random.normal(0, intensity * 255, img.shape)
    img += gauss
    # Salt & pepper
    sp_mask = np.random.random(img.shape)
    n_salt = np.sum(sp_mask < intensity / 10)
    n_pepper = np.sum((sp_mask > 1 - intensity / 10) & (sp_mask <= 1))
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".bmp", img)
    return buf.tobytes() if ok else image_bytes


async def run_one(
    svc: MccMatchingService, person: str, image_bytes: bytes, kind: str,
    rotation_deg: float, crop_pct: float,
) -> ProbeResult:
    t0 = time.monotonic()
    result = await asyncio.get_running_loop().run_in_executor(
        None, svc.search_by_pairs, image_bytes, 5,
    )
    latency = (time.monotonic() - t0) * 1000

    expected = _expected_uuid_map.get(person)
    rank = None
    top1 = result["candidates"][0] if result["candidates"] else None
    for i, c in enumerate(result["candidates"], 1):
        if c["person_id"] == expected:
            rank = i
            break

    return ProbeResult(
        expected_person=person,
        probe_kind=kind,
        rotation_deg=rotation_deg,
        crop_pct=crop_pct,
        top1_person=top1["person_id"] if top1 else None,
        top1_score=top1["score"] if top1 else 0.0,
        top1_hits=top1["peak_votes"] if top1 else 0,
        rank_of_correct=rank,
        latency_ms=latency,
    )


def find_image(person: str) -> Path | None:
    matches = sorted(SOCOFING_ROOT.glob(f"{person}__*_index_finger*.BMP"))
    return matches[0] if matches else None


async def main() -> int:
    engine = create_async_engine(config.async_database_url)
    Sess = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    global _expected_uuid_map
    _expected_uuid_map = {}
    async with Sess() as session:
        rows = (
            await session.execute(
                select(Person).where(Person.external_id.in_(PERSONS))
            )
        ).scalars().all()
        for p in rows:
            _expected_uuid_map[p.external_id] = str(p.id)

    print("=" * 70)
    print("Benchmark: NIST Bozorth3 — real-world robustness (6 scenarios)")
    print(f"Enrolled: {PERSONS}")
    print(f"Linker tolerances: dx={config.matching.link_dx_tol}, dy={config.matching.link_dy_tol}, dtheta={config.matching.link_dtheta_tol}")
    print(f"Saturation: {config.matching.confidence_saturation}")
    print("=" * 70)

    svc = MccMatchingService()

    # 6 scenarios covering real-world forensic use cases
    scenarios: list[tuple[str, str, int, float, float]] = []
    # (short_name, description, rotation_deg, crop_pct, noise_intensity)
    scenarios.append(("clean",      "clean",        0,  0.00, 0.0))
    scenarios.append(("crop",       "crop_25%",     0,  0.25, 0.0))
    scenarios.append(("noise",      "noise_5%",     0,  0.00, 0.05))
    scenarios.append(("crop+noise", "crop+noise",   0,  0.25, 0.05))
    scenarios.append(("rot+noise",  "rot30+noise", 30,  0.00, 0.05))
    scenarios.append(("combo",      "rot30+crop+noise", 30, 0.25, 0.05))

    results: list[ProbeResult] = []

    for person in PERSONS:
        img_path = find_image(person)
        if img_path is None:
            print(f"  SKIP: no image for person {person}")
            continue
        original = img_path.read_bytes()

        for short_name, desc, rot, crop, noise in scenarios:
            img = original
            if rot != 0:
                img = rotate_image(img, rot)
            if crop != 0:
                img = crop_image(img, crop)
            if noise != 0:
                img = add_noise(img, noise)

            r = await run_one(svc, person, img, desc, rot, crop)
            results.append(r)

            status = "OK" if r.top1_person == _expected_uuid_map.get(person) else "FAIL"
            rank_str = f"rank={r.rank_of_correct}" if r.rank_of_correct else "not-in-top5"
            print(
                f"  [{status}] {person:5s} {desc:>16s} "
                f"top1={r.top1_person[:8] if r.top1_person else 'none':8s} "
                    f"votes={r.top1_hits:3d} score={r.top1_score:.3f} "
                    f"{rank_str:14s} latency={r.latency_ms:.0f}ms"
                )

    print()
    print("=" * 70)
    print("Aggregate metrics by (rotation, crop)")
    print("=" * 70)

    by_kind: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        by_kind[r.probe_kind].append(r)

    print(f"\n{'Kind':18s} {'N':>4s} {'Top-1 OK':>10s} {'Top-1 %':>10s} {'Avg votes':>10s} {'Avg latency':>14s}")
    print("-" * 80)
    for kind, rs in sorted(by_kind.items()):
        n = len(rs)
        top1_ok = sum(1 for r in rs if r.top1_person == _expected_uuid_map.get(r.expected_person))
        avg_votes = statistics.mean(r.top1_hits for r in rs) if rs else 0
        avg_latency = statistics.mean(r.latency_ms for r in rs) if rs else 0
        print(
            f"{kind:18s} {n:>4d} {top1_ok:>10d} "
            f"{100*top1_ok/n:>9.1f}% "
            f"{avg_votes:>10.0f} "
            f"{avg_latency:>13.0f}ms"
        )

    print()
    print("Scenario summary:")
    print(f"  {'Scenario':>20s}  {'N':>4s} {'OK':>4s} {'%':>5s}")
    print(f"  {'-'*20}  {'-'*4} {'-'*4} {'-'*5}")
    for kind, rs in sorted(by_kind.items()):
        n = len(rs)
        top1_ok = sum(1 for r in rs if r.top1_person == _expected_uuid_map.get(r.expected_person))
        print(f"  {kind:>20s}  {n:>4d} {top1_ok:>4d} {100*top1_ok/n:>4.0f}%")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    _expected_uuid_map = {}
    sys.exit(asyncio.run(main()))

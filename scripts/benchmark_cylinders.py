"""Benchmark for the current NIST MCC cylinder matcher.

Tests the latent search endpoint (the production path) against
SOCOFing Altered-Easy (CR/Obl/Zcut) and the original Real
(self-match) for the 5 enrolled subjects. Uses the data already
in PostgreSQL + Qdrant — no re-enrollment.

Output: per-probe table of (top-1 candidate, score, latency,
correct/rank) plus aggregate metrics (top-1 accuracy, EER-style
threshold sweep, score distribution).

Usage (from apps/backend):
    uv run python ../../scripts/benchmark_cylinders.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

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
)

PERSONS = ["SOC_0100", "SOC_0101", "SOC_0102", "SOC_0103", "SOC_0104"]


@dataclass
class ProbeResult:
    """One row of benchmark output."""
    expected_person: str
    probe_file: str
    probe_kind: str  # "real" | "altered_cr" | "altered_obl" | "altered_zcut"
    top1_person: str | None
    top1_score: float
    top1_hits: int  # raw count of aligned cylinder hits
    rank_of_correct: int | None  # 1-based; None if not in top-10
    num_candidates: int
    latency_ms: float
    all_top10: list[tuple[str, float, int]] = field(default_factory=list)


def find_socofing_file(person_external_id: str, subdir: str, finger: str = "index") -> Path | None:
    pid = person_external_id.replace("SOC_", "").lstrip("0")
    matches = sorted(
        (SOCOFING_ROOT / subdir).glob(f"{pid}__*_{finger}_finger*.BMP")
    )
    return matches[0] if matches else None


async def run_one(svc: MccMatchingService, person: str, file_path: Path, kind: str) -> ProbeResult:
    img_bytes = file_path.read_bytes()
    t0 = time.monotonic()
    probe_summary, hits = await asyncio.get_running_loop().run_in_executor(
        None, svc.search, img_bytes, 10,
    )
    latency = (time.monotonic() - t0) * 1000

    # Hits are sorted by score desc. Find rank of correct subject.
    # h.person_id is the external_id (e.g. "SOC_0100") because that's
    # what was passed to enroll() and stored in Qdrant. Compare
    # directly against the expected external_id.
    rank_of_correct: int | None = None
    top1 = hits[0] if hits else None
    for idx, h in enumerate(hits, start=1):
        if h.person_id == person:
            rank_of_correct = idx
            break

    return ProbeResult(
        expected_person=person,
        probe_file=file_path.name,
        probe_kind=kind,
        top1_person=top1.person_id if top1 else None,
        top1_score=top1.total_score if top1 else 0.0,
        top1_hits=top1.hits if top1 else 0,
        rank_of_correct=rank_of_correct,
        num_candidates=len(hits),
        latency_ms=latency,
        all_top10=[(h.person_id, h.total_score, h.hits) for h in hits],
    )


async def main() -> int:
    engine = create_async_engine(config.async_database_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Build UUID map: external_id -> person UUID, so we can identify
    # which hit is the "correct" one.
    global _expected_uuid_map
    _expected_uuid_map = {}
    async with Session() as session:
        rows = (
            await session.execute(
                select(Person).where(Person.external_id.in_(PERSONS))
            )
        ).scalars().all()
        for p in rows:
            _expected_uuid_map[p.external_id] = str(p.id)

    print("=" * 70)
    print("Benchmark: NIST MCC cylinders + Hough voting")
    print(f"Enrolled subjects: {PERSONS}")
    print(f"  UUID map: {_expected_uuid_map}")
    print("=" * 70)

    svc = MccMatchingService()

    # Define probe sets:
    # 1. Real self-match (5 probes)
    # 2. Altered-Easy CR (5 probes)
    # 3. Altered-Easy Obl (5 probes)
    # 4. Altered-Easy Zcut (5 probes)
    # Total: 20 probes
    probe_specs: list[tuple[str, str, str]] = []  # (person, subdir, kind)
    for person in PERSONS:
        probe_specs.append((person, "Real", "real"))
        for kind_name, alt_name in [
            ("altered_cr", "Altered/Altered-Easy"),
            ("altered_obl", "Altered/Altered-Easy"),
            ("altered_zcut", "Altered/Altered-Easy"),
        ]:
            # All 3 files for the same person in Altered-Easy
            # (the existing e2e_matching_test.py picks the first
            # match — we'll pick all CR/Obl/Zcut by suffix).
            probe_specs.append((person, alt_name, kind_name))

    results: list[ProbeResult] = []
    for person, subdir, kind in probe_specs:
        # Pick the file matching the kind
        if kind == "real":
            file_path = find_socofing_file(person, subdir)
        else:
            suffix_map = {
                "altered_cr": "_CR.BMP",
                "altered_obl": "_Obl.BMP",
                "altered_zcut": "_Zcut.BMP",
            }
            pid = person.replace("SOC_", "").lstrip("0")
            pattern = f"{pid}__*index_finger{suffix_map[kind]}"
            candidates = sorted((SOCOFING_ROOT / subdir).glob(pattern))
            file_path = candidates[0] if candidates else None

        if file_path is None or not file_path.exists():
            print(f"  SKIP: no file for {person} / {kind}")
            continue

        r = await run_one(svc, person, file_path, kind)
        results.append(r)

        status = "OK" if r.top1_person == person else "FAIL"
        rank_str = f"rank={r.rank_of_correct}" if r.rank_of_correct else "not-in-top10"
        print(
            f"  [{status}] {person:10s} {kind:14s} "
            f"top1={r.top1_person:10s} "
            f"score={r.top1_score:.3f} hits={r.top1_hits:3d} "
            f"{rank_str:14s} "
            f"latency={r.latency_ms:.0f}ms"
        )

    # Aggregate metrics
    print()
    print("=" * 70)
    print("Aggregate metrics")
    print("=" * 70)

    by_kind: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        by_kind[r.probe_kind].append(r)

    print(f"\n{'Kind':16s} {'N':>4s} {'Top-1 OK':>10s} {'Top-1 %':>10s} {'Avg hits':>10s} {'Avg score':>12s} {'Avg latency':>14s}")
    print("-" * 80)
    for kind, rs in sorted(by_kind.items()):
        n = len(rs)
        top1_ok = sum(1 for r in rs if r.top1_person == r.expected_person)
        avg_hits = statistics.mean(r.top1_hits for r in rs)
        avg_score = statistics.mean(r.top1_score for r in rs)
        avg_latency = statistics.mean(r.latency_ms for r in rs)
        print(
            f"{kind:16s} {n:>4d} {top1_ok:>10d} "
            f"{100*top1_ok/n:>9.1f}% "
            f"{avg_hits:>10.0f} "
            f"{avg_score:>12.3f} {avg_latency:>13.0f}ms"
        )

    # Rank distribution
    print()
    print("Rank distribution (where is the correct subject?):")
    for kind, rs in sorted(by_kind.items()):
        rank_counts: dict[int, int] = defaultdict(int)
        for r in rs:
            rank_counts[r.rank_of_correct or 0] += 1
        dist_str = " ".join(f"r{rank}={n}" for rank, n in sorted(rank_counts.items()))
        print(f"  {kind:16s} {dist_str}")

    # Score distribution by raw hits (not the saturated score)
    print()
    print("Hits distribution (top-1 raw hits, all probes):")
    correct_hits = [r.top1_hits for r in results if r.top1_person == r.expected_person]
    wrong_hits = [r.top1_hits for r in results if r.top1_person != r.expected_person]
    if correct_hits:
        print(f"  Correct top-1:   n={len(correct_hits)}  min={min(correct_hits):3d}  max={max(correct_hits):3d}  mean={statistics.mean(correct_hits):6.1f}")
    if wrong_hits:
        print(f"  Wrong top-1:     n={len(wrong_hits)}  min={min(wrong_hits):3d}  max={max(wrong_hits):3d}  mean={statistics.mean(wrong_hits):6.1f}")
    if correct_hits and wrong_hits:
        # Threshold sweep on raw hits (better than saturated score)
        for thresh in [5, 10, 15, 20, 30, 50]:
            n_correct_above = sum(1 for s in correct_hits if s >= thresh)
            n_wrong_above = sum(1 for s in wrong_hits if s >= thresh)
            print(f"  hits >= {thresh:2d}: correct_above={n_correct_above}/{len(correct_hits)}  wrong_above={n_wrong_above}/{len(wrong_hits)}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    _expected_uuid_map = {}
    sys.exit(asyncio.run(main()))

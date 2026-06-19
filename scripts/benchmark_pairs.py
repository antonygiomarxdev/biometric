"""Benchmark for the Bozorth3 pairs matcher (Phase 27, Plan 27-01).

Tests ``MccMatchingService.search_by_pairs`` against SOCOFing Real
(self-match) and Altered-Easy (CR/Obl/Zcut) for the 5 enrolled
subjects. Uses the data already in Qdrant.

Output: per-probe table of (top-1 candidate, votes, score, latency)
plus aggregate metrics (top-1 accuracy per probe type).

Usage (from apps/backend):
    uv run python ../../scripts/benchmark_pairs.py
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

PERSONS = ["1", "2", "3", "4", "5"]


@dataclass
class ProbeResult:
    expected_person: str
    probe_file: str
    probe_kind: str  # "real" | "altered_cr" | "altered_obl" | "altered_zcut"
    top1_person: str | None
    top1_score: float
    top1_hits: int
    rank_of_correct: int | None
    num_candidates: int
    latency_ms: float


def find_probe(person: str, subdir: str, suffix: str = "") -> Path | None:
    matches = sorted(
        (SOCOFING_ROOT / subdir).glob(f"{person}__*_index_finger{suffix}*.BMP")
    )
    return matches[0] if matches else None


async def run_one(svc: MccMatchingService, person: str, file_path: Path, kind: str) -> ProbeResult:
    img_bytes = file_path.read_bytes()
    t0 = time.monotonic()
    result = await asyncio.get_running_loop().run_in_executor(
        None, svc.search_by_pairs, img_bytes, 10,
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
        probe_file=file_path.name,
        probe_kind=kind,
        top1_person=top1["person_id"] if top1 else None,
        top1_score=top1["score"] if top1 else 0.0,
        top1_hits=top1["peak_votes"] if top1 else 0,
        rank_of_correct=rank,
        num_candidates=len(result["candidates"]),
        latency_ms=latency,
    )


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
    print("Benchmark: NIST Bozorth3 pair linking")
    print(f"Tolerances: dx={config.matching.link_dx_tol}, dy={config.matching.link_dy_tol}, dtheta={config.matching.link_dtheta_tol}")
    print(f"Saturation: {config.matching.confidence_saturation}")
    print(f"Enrolled subjects: {PERSONS}")
    print("=" * 70)

    svc = MccMatchingService()

    probe_specs: list[tuple[str, str, str, str]] = []
    for person in PERSONS:
        probe_specs.append((person, "Real", "real", ""))
        for kind_name, suffix in [
            ("altered_cr", "_CR"),
            ("altered_obl", "_Obl"),
            ("altered_zcut", "_Zcut"),
        ]:
            probe_specs.append((person, "Altered/Altered-Easy", kind_name, suffix))

    results: list[ProbeResult] = []
    for person, subdir, kind, suffix in probe_specs:
        file_path = find_probe(person, subdir, suffix)
        if file_path is None or not file_path.exists():
            print(f"  SKIP: no file for {person} / {kind}")
            continue

        r = await run_one(svc, person, file_path, kind)
        results.append(r)

        expected = _expected_uuid_map.get(person)
        ok = r.top1_person == expected
        status = "OK" if ok else "FAIL"
        rank_str = f"rank={r.rank_of_correct}" if r.rank_of_correct else "not-in-top10"
        print(
            f"  [{status}] {person:5s} {kind:14s} "
            f"top1={r.top1_person[:8] if r.top1_person else 'none':8s} "
            f"votes={r.top1_hits:3d} score={r.top1_score:.3f} "
            f"{rank_str:14s} latency={r.latency_ms:.0f}ms"
        )

    print()
    print("=" * 70)
    print("Aggregate metrics")
    print("=" * 70)

    by_kind: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        by_kind[r.probe_kind].append(r)

    print(f"\n{'Kind':16s} {'N':>4s} {'Top-1 OK':>10s} {'Top-1 %':>10s} {'Avg votes':>10s} {'Avg score':>10s} {'Avg latency':>14s}")
    print("-" * 80)
    for kind, rs in sorted(by_kind.items()):
        n = len(rs)
        top1_ok = sum(1 for r in rs if r.top1_person == _expected_uuid_map.get(r.expected_person))
        avg_votes = statistics.mean(r.top1_hits for r in rs)
        avg_score = statistics.mean(r.top1_score for r in rs)
        avg_latency = statistics.mean(r.latency_ms for r in rs)
        print(
            f"{kind:16s} {n:>4d} {top1_ok:>10d} "
            f"{100*top1_ok/n:>9.1f}% "
            f"{avg_votes:>10.0f} "
            f"{avg_score:>10.3f} {avg_latency:>13.0f}ms"
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

    # Votes distribution
    print()
    print("Votes distribution (top-1 votes, all probes):")
    correct_votes = [r.top1_hits for r in results if r.top1_person == _expected_uuid_map.get(r.expected_person)]
    wrong_votes = [r.top1_hits for r in results if r.top1_person != _expected_uuid_map.get(r.expected_person)]
    if correct_votes:
        print(f"  Correct top-1:   n={len(correct_votes):2d}  min={min(correct_votes):3d}  max={max(correct_votes):3d}  mean={statistics.mean(correct_votes):6.1f}")
    if wrong_votes:
        print(f"  Wrong top-1:     n={len(wrong_votes):2d}  min={min(wrong_votes):3d}  max={max(wrong_votes):3d}  mean={statistics.mean(wrong_votes):6.1f}")
    if correct_votes and wrong_votes:
        for thresh in [5, 10, 15, 20, 30, 50]:
            n_correct = sum(1 for v in correct_votes if v >= thresh)
            n_wrong = sum(1 for v in wrong_votes if v >= thresh)
            print(f"  votes >= {thresh:2d}: correct_above={n_correct}/{len(correct_votes)}  wrong_above={n_wrong}/{len(wrong_votes)}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    _expected_uuid_map = {}
    sys.exit(asyncio.run(main()))

"""
Phase 21 — SOCOFing benchmark for production MccMatchingService.

Validates the 80% Rank-1 (3 minutiae) and 100% Rank-1 (15 minutiae) claim
from the Phase 20 spike (``scripts/spike_mcc.py``) against the production
MccMatchingService wrapping QdrantMccRepository.

Methodology
-----------
1. Enroll N=10 prints from SOCOFing/Real via ``MccMatchingService.enroll()``
   (production path).
2. For each minutiae count N in {3, 5, 8, 15}:
     * Re-run the full spike-style pipeline to get all minutiae (cached,
       so the pipeline runs once per image, not per N).
     * Take a random subset of N minutiae, add Gauss(0, 3 px) noise to
       coordinates and Gauss(0, 0.1 rad) to angle.
     * Build cylinders from the perturbed subset via the production
       ``mcc_descriptor.extract_cylinders``.
     * Query ``MccMatchingService._search_cylinders()`` and measure
       Rank-1/5/10 + avg search time.
3. Report a table matching the spike's output format.

If the production pipeline returns 0 minutiae for a print (currently the
case for SOCOFing — see findings printed at the end), the script also
runs a "spike-fallback" benchmark that uses RidgeGraph nodes (the same
source the spike used) so we can still measure the underlying MCC
algorithm. The two measurements are reported side-by-side.
"""
from __future__ import annotations

import math
import random
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.interfaces import PipelineContext
from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.processing.enhancer import create_enhancer
from src.processing.gabor import QualityMaskStep
from src.processing.graph_extractor import RidgeGraphExtractor
from src.processing.mcc_descriptor import extract_cylinders
from src.processing.pre_hooks import (
    OrientationFieldAnalyzer,
    SingularityDetector,
)
from src.processing.skeletonize_step import SkeletonizationStep
from src.processing.spurious_filter import SkeletonCleanerStep
from src.services.mcc_matching_service import MccMatchingService
from qdrant_client import QdrantClient

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

SOCOF = Path(__file__).resolve().parents[1] / "static" / "SOCOFing" / "Real"
N_ENROLL = 10
MINUTIAE_COUNTS = (3, 5, 8, 15)
SEED = 42
COLLECTION = "mcc_phase21_bench"
COORD_NOISE = 3.0
ANGLE_NOISE = 0.1


def _spike_pipeline(img: np.ndarray):
    """Spike-style pipeline: returns (minutiae_dicts, skeleton, orient, freq).

    Replicates ``spike_mcc.pipeline`` so we get the same RidgeGraph-based
    minutiae the Phase 20 spike used.
    """
    ctx = PipelineContext(raw_image=img, fingerprint_id="x")
    enhanced = create_enhancer().enhance(img, resize=True)
    ctx.enhanced_image = enhanced
    ctx.preprocessed_image = enhanced

    OrientationFieldAnalyzer().process(ctx)
    QualityMaskStep().process(ctx)

    SingularityDetector(roi_radius=140).process(ctx)
    SkeletonizationStep(min_island_size=20).process(ctx)
    SkeletonCleanerStep().process(ctx)
    RidgeGraphExtractor().process(ctx)

    rg = ctx.ridge_graph
    if rg is None:
        return [], np.zeros((1, 1)), None, None
    nodes = [
        {"x": float(n.x), "y": float(n.y), "angle": float(n.angle)}
        for n in rg.nodes
    ]
    skel = ctx.skeleton if ctx.skeleton is not None else np.zeros((1, 1))
    return nodes, skel, ctx.orientation_field, ctx.freq_image


def _build_repo() -> tuple[QdrantClient, QdrantMccRepository]:
    """Build a QdrantMccRepository on a fresh in-memory Qdrant.

    Note: the local Qdrant at localhost:6333 has 'too many open files'
    errors, so we use in-memory. The algorithm under test (cosine KNN +
    per-person aggregation) is identical between the two.
    """
    client = QdrantClient(location=":memory:")
    repo = QdrantMccRepository(client, collection=COLLECTION)
    repo.ensure_collection()
    return client, repo


def _cache_pipeline(paths: list[Path]) -> dict[str, tuple]:
    """Run spike pipeline once per image, cache the result."""
    cache: dict[str, tuple] = {}
    t0 = time.monotonic()
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        nodes, skel, orient, freq = _spike_pipeline(img)
        cache[p.stem] = (nodes, skel, orient, freq)
    print(
        f"  Pipeline cache built in {time.monotonic() - t0:.1f}s "
        f"({len(cache)} images)"
    )
    return cache


def _enroll_via_service(
    svc: MccMatchingService,
    paths: list[Path],
) -> dict[str, int]:
    """Enroll N prints via MccMatchingService.enroll(). Returns {stem: n}."""
    counts: dict[str, int] = {}
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ok, buf = cv2.imencode(".bmp", img)
        n = svc.enroll(
            capture_id=f"cap_{p.stem}",
            fingerprint_id=p.stem,
            person_id=p.stem,
            image_bytes=bytes(buf),
        )
        counts[p.stem] = n
    return counts


def _enroll_via_repo_direct(
    repo: QdrantMccRepository,
    cache: dict[str, tuple],
    stems: list[str],
) -> dict[str, int]:
    """Insert RidgeGraph-based cylinders directly into the repo (spike-style).

    This is the fallback enrollment: it uses the same minutiae source
    the Phase 20 spike used, and the same QdrantMccRepository the
    production MccMatchingService wraps. The only thing that differs
    from the spike is that we go through the production repo, not
    direct Qdrant upserts.
    """
    counts: dict[str, int] = {}
    for stem in stems:
        nodes, skel, orient, freq = cache[stem]
        if not nodes:
            counts[stem] = 0
            continue
        cylinders = extract_cylinders(nodes, skel, orient, freq)
        n = repo.bulk_insert_cylinders(
            person_id=stem,
            fingerprint_id=stem,
            capture_id=f"spike_cap_{stem}",
            vectors=cylinders,
        )
        counts[stem] = n
    return counts


def _run_benchmark(
    svc: MccMatchingService,
    cache: dict[str, tuple],
    stems: list[str],
    label: str,
) -> dict[int, dict[str, float]]:
    """Run the rank-vs-minutiae benchmark. Returns {n_min: {rank1, ...}}."""
    rng = random.Random(SEED)
    results: dict[int, dict[str, float]] = {}
    print(
        f"\n{'Minutiae':>8s}  {'Rank-1':>7s}  {'Rank-5':>7s}  "
        f"{'Rank-10':>7s}  {'AvgTime':>8s}"
    )
    print("-" * 48)

    for n_min in MINUTIAE_COUNTS:
        hits = {1: 0, 5: 0, 10: 0}
        total = 0
        t_tot = 0.0
        for stem in stems:
            nodes, skel, orient, freq = cache[stem]
            if len(nodes) < n_min:
                continue
            idx = rng.sample(range(len(nodes)), n_min)
            perturbed = [
                {
                    "x": nodes[i]["x"] + rng.gauss(0, COORD_NOISE),
                    "y": nodes[i]["y"] + rng.gauss(0, COORD_NOISE),
                    "angle": (nodes[i]["angle"] + rng.gauss(0, ANGLE_NOISE))
                    % (2 * math.pi),
                }
                for i in idx
            ]
            cylinders = extract_cylinders(perturbed, skel, orient, freq)
            if len(cylinders) < 3:
                continue

            t0 = time.monotonic()
            ranked = svc._search_cylinders(cylinders, top_k=10)
            t_tot += time.monotonic() - t0

            ranked_ids = [h.person_id for h in ranked]
            try:
                rank = ranked_ids.index(stem) + 1
            except ValueError:
                rank = -1
            for k in hits:
                if 0 < rank <= k:
                    hits[k] += 1
            total += 1

        avg_ms = t_tot / max(total, 1) * 1000
        row = {
            "rank1": hits[1] / max(total, 1) * 100,
            "rank5": hits[5] / max(total, 1) * 100,
            "rank10": hits[10] / max(total, 1) * 100,
            "avg_ms": avg_ms,
            "total": total,
        }
        results[n_min] = row
        print(
            f"  {n_min:>8d}  {row['rank1']:6.1f}%  {row['rank5']:6.1f}%  "
            f"{row['rank10']:6.1f}%  {avg_ms:6.0f}ms"
        )
    return results


def main() -> int:
    print("=" * 60)
    print("Phase 21 — SOCOFing benchmark for MccMatchingService")
    print("=" * 60)

    all_imgs = sorted(SOCOF.glob("*.BMP"))
    rng = random.Random(SEED)
    rng.shuffle(all_imgs)
    selected = [p for p in all_imgs if cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) is not None][:N_ENROLL]
    print(f"Selected {len(selected)} SOCOFing/Real prints")

    # Pre-compute the spike-style pipeline once per image
    print("\nPre-computing spike-style pipeline for each image...")
    cache = _cache_pipeline(selected)

    # ---- Production benchmark: use MccMatchingService.enroll() ----
    print("\n--- Production benchmark (MccMatchingService.enroll) ---")
    _, prod_repo = _build_repo()
    prod_svc = MccMatchingService(mcc_repo=prod_repo)
    prod_counts = _enroll_via_service(prod_svc, selected)
    prod_total = sum(prod_counts.values())
    prod_enrolled = [s for s, n in prod_counts.items() if n > 0]
    print(
        f"Enrolled {len(prod_enrolled)}/{len(selected)} prints with cylinders "
        f"({prod_total} total)"
    )
    if prod_enrolled:
        prod_results = _run_benchmark(
            prod_svc, cache, prod_enrolled, label="production",
        )
    else:
        prod_results = {}
        print(
            "SKIPPED: production enrollment returned 0 cylinders. "
            "Falling back to spike-style enrollment below."
        )

    # ---- Fallback benchmark: enroll cylinders via the production repo,
    #      but with RidgeGraph-based minutiae (matches the spike) ----
    print("\n--- Fallback benchmark (spike-style enrollment + production repo) ---")
    _, fb_repo = _build_repo()
    fb_svc = MccMatchingService(mcc_repo=fb_repo)
    fb_counts = _enroll_via_repo_direct(fb_repo, cache, [p.stem for p in selected])
    fb_total = sum(fb_counts.values())
    fb_enrolled = [s for s, n in fb_counts.items() if n > 0]
    print(
        f"Enrolled {len(fb_enrolled)}/{len(selected)} prints with cylinders "
        f"({fb_total} total)"
    )
    fallback_results = _run_benchmark(
        fb_svc, cache, [p.stem for p in selected], label="fallback",
    )

    # ---- Findings ----
    print("\n" + "=" * 60)
    print("Findings")
    print("=" * 60)
    print(
        f"Production enrollment: {len(prod_enrolled)}/{len(selected)} prints "
        f"({prod_total} total cylinders)"
    )
    print(
        f"Fallback  enrollment: {len(fb_enrolled)}/{len(selected)} prints "
        f"({fb_total} total cylinders)"
    )
    if prod_total == 0:
        print(
            "\nROOT CAUSE: MccMatchingService.enroll() returns 0 cylinders\n"
            "for SOCOFing because the underlying FingerprintService pipeline\n"
            "produces 0 MinutiaCandidates from SOCOFing images. The\n"
            "RidgeGraphExtractor extracts 80+ ridge-graph nodes per image,\n"
            "but the SkeletonMinutiaeExtractor (the default) binarizes the\n"
            "input with `image > 127` and returns 0 candidates when fed the\n"
            "binary (0/1) skeleton — i.e. the production pipeline is\n"
            "functionally broken for SOCOFing-like datasets.\n"
        )

    # ---- Acceptance comparison ----
    print("=" * 60)
    print("Acceptance comparison (target: 80%@3, 100%@15, <500ms)")
    print("=" * 60)
    if prod_results:
        r3 = prod_results.get(3, {})
        r15 = prod_results.get(15, {})
        print(
            f"Production: Rank-1@3 = {r3.get('rank1', 0):.1f}%, "
            f"Rank-1@15 = {r15.get('rank1', 0):.1f}%, "
            f"avg time = {r15.get('avg_ms', 0):.0f}ms"
        )
    else:
        print("Production: NO DATA (0 cylinders enrolled)")
    if fallback_results:
        r3 = fallback_results.get(3, {})
        r15 = fallback_results.get(15, {})
        print(
            f"Fallback : Rank-1@3 = {r3.get('rank1', 0):.1f}%, "
            f"Rank-1@15 = {r15.get('rank1', 0):.1f}%, "
            f"avg time = {r15.get('avg_ms', 0):.0f}ms"
        )

    # Acceptance gates (spike comparison): 80%@3 and 100%@15 within 10%
    spike_match_3 = fallback_results.get(3, {}).get("rank1", 0) >= 70
    spike_match_15 = fallback_results.get(15, {}).get("rank1", 0) >= 90
    print(
        f"\nFallback within 10% of spike claim? "
        f"3min: {'YES' if spike_match_3 else 'NO'}, "
        f"15min: {'YES' if spike_match_15 else 'NO'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

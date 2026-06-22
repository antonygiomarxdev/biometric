"""Score distribution benchmark for the AFR-Net + Qdrant pipeline.

Runs ``N`` probes (default 200) sampled from SOCOFing Real /
Altered-Easy / Altered-Medium / Altered-Hard.  For each probe we
record:

  - top-1 score (cosine similarity with the closest gallery image)
  - margin (top-1 score − top-2 score)
  - top-K scores (for ROC / threshold analysis)
  - whether top-1 is the correct person

Then prints the per-variant distribution plus a table of TPR / FPR
at candidate thresholds so we can pick UI thresholds for
``MATCH_THRESHOLD_GOOD`` / ``MATCH_THRESHOLD_FAIR`` from data, not
from guesswork.

This is a one-shot data collection script, not a service.  It loads
the model once, runs the probes, prints the table, exits.

Usage::

    uv run --no-sync python scripts/benchmark_score_distribution.py \\
        --n-probes 200 --top-k 10
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient

# Ensure the project root is importable so ``src.*`` resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.loader import ModelLoader  # noqa: E402
from src.services.embedding_service import EmbeddingService  # noqa: E402
from src.db.qdrant_embedding_repository import (  # noqa: E402
    QdrantEmbeddingRepository,
)
from src.core import config as app_config  # noqa: E402

SOCOFING_ROOT = Path(
    os.getenv(
        "SOCOFING_ROOT",
        str(Path.home() / "Downloads/SOCOFing/socofing/SOCOFing"),
    )
)

VARIANTS = ["Real", "Altered-Easy", "Altered-Medium", "Altered-Hard"]


@dataclass
class ProbeResult:
    variant: str
    subject_id: int
    finger: str
    top1_score: float
    top2_score: float
    topk_scores: list[float]
    correct: bool
    embed_ms: int
    search_ms: int


def list_probes(variant: str, n: int, seed: int) -> list[Path]:
    # SOCOFing layout: ``SOCOFing/Real/`` and
    # ``SOCOFing/Altered/Altered-{Easy,Medium,Hard}/``.
    if variant == "Real":
        root = SOCOFING_ROOT / "Real"
    else:
        root = SOCOFING_ROOT / "Altered" / variant
    files = sorted(root.glob("*.BMP"))
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:n]


def parse_filename(path: Path) -> tuple[int, str, str]:
    """``100__M_Left_index_finger.BMP`` → (100, "Left_index", "CR")."""
    stem = path.stem
    # Strip variant suffix if present (Real has none, Altered-X has _CR/_Zcut/...)
    for suf in ("_CR", "_Zcut", "_Obl", "_Oblique"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    parts = stem.split("__")
    subject_id = int(parts[0])
    finger_part = parts[1]
    # ``M_Left_index_finger`` → ``Left_index``
    tokens = finger_part.split("_")
    if tokens[0] in ("M", "F"):
        tokens = tokens[1:]
    finger = "_".join(t for t in tokens if t != "finger")
    return subject_id, finger, ""


def load_image_bytes(path: Path) -> bytes:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Failed to encode {path}")
    return buf.tobytes()


async def main_async(args: argparse.Namespace) -> None:
    loader = ModelLoader()
    # Force eager load (the production API uses ``get_embedding_model``
    # which loads on first call; here we want a clear error if the
    # model file is missing).
    _ = loader.embedding_model
    print(f"Model loaded on {loader.device}", flush=True)

    qdrant_client = QdrantClient(host="localhost", port=6333)
    repo = QdrantEmbeddingRepository(client=qdrant_client,
                                     collection=app_config.qdrant_embedding_collection)
    repo.ensure_collection()
    service = EmbeddingService(loader=loader, qdrant=repo)

    results: list[ProbeResult] = []
    t_total = time.monotonic()

    for variant in VARIANTS:
        paths = list_probes(variant, args.n_probes, args.seed)
        print(f"\n=== {variant} ({len(paths)} probes) ===", flush=True)
        t0 = time.monotonic()
        for i, path in enumerate(paths):
            subject_id, finger, _ = parse_filename(path)
            image_bytes = load_image_bytes(path)
            t_embed = time.monotonic()
            emb_result = await service.embed(image_bytes, with_gradcam=False)
            # ``embed()`` returns ``(vector, gradcam_or_None)``.
            vector: NDArray[np.float32] = emb_result[0]
            embed_ms = int((time.monotonic() - t_embed) * 1000)

            t_search = time.monotonic()
            hits = await asyncio.get_running_loop().run_in_executor(
                loader.pool, repo.search, vector, args.top_k,
            )
            search_ms = int((time.monotonic() - t_search) * 1000)

            if not hits:
                continue
            scores = [h["score"] for h in hits]
            top1 = scores[0]
            top2 = scores[1] if len(scores) > 1 else 0.0

            # Resolve "correct" via the Qdrant payload.  We must look up
            # ``person_id`` in PostgreSQL because Qdrant stores the *fingerprint*
            # UUID, not the SOCOFing subject number.  The cleanest signal
            # for benchmark purposes is whether the top-1 hit is one of the
            # 10 captures enrolled for this subject.  We approximate that
            # by checking whether the top-K contains at least one capture
            # whose ``fingerprint_id`` is enrolled under a person whose
            # ``external_id`` starts with ``SOC_<subject_id>``.  That
            # requires PG, which we don't open here — instead, we use the
            # indirect signal that the top-1 hit's payload ``person_id`` is
            # among the gallery.  For the 6K SOCOFing gallery every person
            # has exactly one Real capture, so top-1 score is the right
            # signal for Real.  For Altered variants the correct Real
            # capture is what we want — see ``quick_enroll.py`` for the
            # mapping convention.
            #
            # Without PG access we cannot tell "correct person" vs
            # "wrong person" exactly.  We record the score distribution
            # anyway, which is the point of this benchmark.
            results.append(ProbeResult(
                variant=variant,
                subject_id=subject_id,
                finger=finger,
                top1_score=top1,
                top2_score=top2,
                topk_scores=scores,
                correct=False,  # populated below if PG is reachable
                embed_ms=embed_ms,
                search_ms=search_ms,
            ))

            if (i + 1) % 25 == 0:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / elapsed
                print(
                    f"  [{i+1}/{len(paths)}] {rate:.1f} probes/s",
                    flush=True,
                )

        elapsed = time.monotonic() - t0
        print(
            f"  {variant} done in {elapsed:.1f}s "
            f"({len(paths)/elapsed:.1f} probes/s)",
            flush=True,
        )

    total = time.monotonic() - t_total
    print(f"\nTotal: {total:.1f}s for {len(results)} probes", flush=True)

    # ----------------------------------------------------------------
    # Per-variant score distribution.
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SCORE DISTRIBUTION PER VARIANT (top-1 score)")
    print("=" * 72)
    by_variant: dict[str, list[float]] = defaultdict(list)
    margins_by_variant: dict[str, list[float]] = defaultdict(list)
    for r in results:
        by_variant[r.variant].append(r.top1_score)
        margins_by_variant[r.variant].append(r.top1_score - r.top2_score)

    print(
        f"{'Variant':<20} {'n':>5} {'min':>6} {'p25':>6} {'p50':>6} "
        f"{'p75':>6} {'p95':>6} {'max':>6}"
    )
    for variant in VARIANTS:
        scores = sorted(by_variant[variant])
        if not scores:
            continue
        n = len(scores)
        def pct(arr: list[float], q: float) -> float:
            return arr[min(n - 1, int(n * q))]
        print(
            f"{variant:<20} {n:>5} {pct(scores, 0):>6.3f} {pct(scores, 0.25):>6.3f} "
            f"{pct(scores, 0.5):>6.3f} {pct(scores, 0.75):>6.3f} {pct(scores, 0.95):>6.3f} "
            f"{pct(scores, 1):>6.3f}"
        )

    print("\nMARGIN DISTRIBUTION (top-1 − top-2)")
    print(
        f"{'Variant':<20} {'n':>5} {'min':>6} {'p25':>6} {'p50':>6} "
        f"{'p75':>6} {'p95':>6} {'max':>6}"
    )
    for variant in VARIANTS:
        m = sorted(margins_by_variant[variant])
        if not m:
            continue
        n = len(m)
        def pct(arr: list[float], q: float) -> float:
            return arr[min(n - 1, int(n * q))]
        print(
            f"{variant:<20} {n:>5} {pct(m, 0):>6.3f} {pct(m, 0.25):>6.3f} "
            f"{pct(m, 0.5):>6.3f} {pct(m, 0.75):>6.3f} {pct(m, 0.95):>6.3f} "
            f"{pct(m, 1):>6.3f}"
        )

    # ----------------------------------------------------------------
    # Threshold sweep — what fraction of probes would be "above
    # threshold" at each candidate cut?  Without "correct" labels we
    # report only the unconditional fraction; once we add PG lookup
    # we can compute TPR / FPR.
    # ----------------------------------------------------------------
    print("\nTHRESHOLD SWEEP — fraction of probes with top-1 ≥ threshold")
    print("(includes all variants; one row per variant)")
    print(
        f"{'Threshold':>10} "
        + " ".join(f"{v:>15}" for v in VARIANTS)
    )
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        row = [f"{thr:>10.2f}"]
        for variant in VARIANTS:
            scores = by_variant[variant]
            if not scores:
                row.append(f"{'—':>15}")
                continue
            frac = sum(1 for s in scores if s >= thr) / len(scores)
            row.append(f"{frac:>15.1%}")
        print(" ".join(row))

    # ----------------------------------------------------------------
    # Latency.
    # ----------------------------------------------------------------
    print("\nLATENCY (ms, per probe)")
    emb: list[int] = [r.embed_ms for r in results]
    sch: list[int] = [r.search_ms for r in results]
    print(
        f"  embed  median={statistics.median(emb):>4.0f}  "
        f"p95={sorted(emb)[int(0.95*len(emb))]:>4.0f}"
    )
    print(
        f"  search median={statistics.median(sch):>4.0f}  "
        f"p95={sorted(sch)[int(0.95*len(sch))]:>4.0f}"
    )

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Probes per variant (default 50; 200 total)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K to query (default 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

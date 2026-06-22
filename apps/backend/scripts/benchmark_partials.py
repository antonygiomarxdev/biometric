"""Partial-print benchmark for the AFR-Net + Qdrant pipeline.

The user reported that a manually cropped fingerprint (a piece of
the full print) produces many false positives at low scores.
This benchmark quantifies the problem.

For each probe we:
  1. Take a full Real SOCOFing print
  2. Generate 4 crops at 25 % / 50 % / 75 % of the print:
     - ``center`` (the centre 25 / 50 / 75 % square)
     - ``topleft``, ``topright``, ``bottomleft``, ``bottomright`` (quadrants)
  3. Embed the crop
  4. Query Qdrant for top-K nearest neighbours
  5. Mark the probe as ``correct`` if any of the top-K hits' ``person_id``
     matches the SOCOFing subject we expect.

We do NOT yet have a "correct" label that we can trust without PG
access.  We *do* have the SOCOFing subject id in the filename and
the Qdrant hit's ``person_id`` is a UUID.  Without PG we cannot
resolve UUID → subject.  To still get a useful number we report:

  - The top-1 score distribution per crop size
  - The fraction of crops where the top-1 score is below 0.5
    (i.e. the model "gave up")
  - The fraction where the top-1 score is above 0.5 but the model
    is "not sure" (margin #1 − #2 below 0.05)

The intent is to see whether cropping drives the model into the
"noise" regime.  If it does, sliding window + multi-crop aggregation
is justified.  If it doesn't, the crop preprocessing we already
have (bbox-based centering) may be enough.

Usage::

    uv run --no-sync python scripts/benchmark_partials.py \\
        --n-probes 50 --crop-sizes 25 50 75
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.loader import ModelLoader  # noqa: E402
from src.services.embedding_service import EmbeddingService  # noqa: E402
from src.db.qdrant_embedding_repository import (  # noqa: E402
    QdrantEmbeddingRepository,
)
from src.core import config as app_config  # noqa: E402

SOCOFING_REAL = Path(
    os.getenv(
        "SOCOFING_REAL",
        str(Path.home() / "Downloads/SOCOFing/socofing/SOCOFing/Real"),
    )
)


@dataclass
class CropResult:
    crop_kind: str  # "center_25", "topleft_50", etc.
    crop_size: int  # 25, 50, 75
    top1_score: float
    top2_score: float
    margin: float
    top1_in_top10: bool  # is the matching capture in top-10?
    embed_ms: int
    search_ms: int


CROP_KINDS_PER_SIZE = ("center", "topleft", "topright", "bottomleft", "bottomright")


def make_crop(img: NDArray[np.uint8], kind: str, pct: int) -> NDArray[np.uint8]:
    """Crop ``img`` to ``pct`` % of its shorter side, anchored by
    ``kind`` (centre or one of the 4 corners)."""
    h, w = img.shape[:2]
    side = int(min(h, w) * pct / 100)
    if kind == "center":
        y0 = (h - side) // 2
        x0 = (w - side) // 2
    elif kind == "topleft":
        y0, x0 = 0, 0
    elif kind == "topright":
        y0, x0 = 0, w - side
    elif kind == "bottomleft":
        y0, x0 = h - side, 0
    elif kind == "bottomright":
        y0, x0 = h - side, w - side
    else:
        raise ValueError(f"unknown kind: {kind}")
    return img[y0:y0 + side, x0:x0 + side]


def parse_filename(path: Path) -> int:
    """``100__M_Left_index_finger.BMP`` → 100."""
    return int(path.stem.split("__")[0])


def load_image(path: Path) -> NDArray[np.uint8]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read {path}")
    return np.asarray(img, dtype=np.uint8)


def encode_png(img: NDArray[np.uint8]) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Failed to encode PNG")
    return buf.tobytes()


async def main_async(args: argparse.Namespace) -> None:
    loader = ModelLoader()
    _ = loader.embedding_model
    print(f"Model loaded on {loader.device}", flush=True)

    qdrant_client = QdrantClient(host="localhost", port=6333)
    repo = QdrantEmbeddingRepository(
        client=qdrant_client,
        collection=app_config.qdrant_embedding_collection,
    )
    repo.ensure_collection()
    service = EmbeddingService(loader=loader, qdrant=repo)

    files = sorted(SOCOFING_REAL.glob("*.BMP"))
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[: args.n_probes]
    print(f"{len(files)} probes × {len(CROP_KINDS_PER_SIZE)} crops × "
          f"{len(args.crop_sizes)} sizes = "
          f"{len(files) * len(CROP_KINDS_PER_SIZE) * len(args.crop_sizes)} queries",
          flush=True)

    results: list[CropResult] = []
    t_total = time.monotonic()

    for i, path in enumerate(files):
        # ``subject_id`` would be used once we have a PG lookup
        # to verify the top-1 hit.  We keep the parse so the
        # file is exercised (catches malformed names).
        _subject_id = parse_filename(path)
        full_img = load_image(path)
        for size in args.crop_sizes:
            for kind in CROP_KINDS_PER_SIZE:
                crop = make_crop(full_img, kind, size)
                image_bytes = encode_png(crop)

                t_embed = time.monotonic()
                emb_result = await service.embed(image_bytes, with_gradcam=False)
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

                results.append(CropResult(
                    crop_kind=f"{kind}_{size}",
                    crop_size=size,
                    top1_score=top1,
                    top2_score=top2,
                    margin=top1 - top2,
                    top1_in_top10=bool(hits),  # we cannot verify w/o PG
                    embed_ms=embed_ms,
                    search_ms=search_ms,
                ))

        if (i + 1) % 10 == 0:
            elapsed = time.monotonic() - t_total
            rate = (i + 1) / elapsed
            print(
                f"  [{i+1}/{len(files)}] {rate:.1f} probes/s",
                flush=True,
            )

    total = time.monotonic() - t_total
    print(f"\nTotal: {total:.1f}s for {len(results)} crops", flush=True)

    # ----------------------------------------------------------------
    # Per-crop-size score distribution.
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TOP-1 SCORE BY CROP SIZE")
    print("=" * 72)
    by_size: dict[int, list[float]] = defaultdict(list)
    by_kind: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in results:
        by_size[r.crop_size].append(r.top1_score)
        by_kind[(r.crop_kind.split("_")[0], r.crop_size)].append(r.top1_score)

    print(
        f"{'Crop size':>10} {'n':>5} {'min':>6} {'p25':>6} {'p50':>6} "
        f"{'p75':>6} {'p95':>6} {'max':>6}"
    )
    for size in args.crop_sizes:
        scores = sorted(by_size[size])
        if not scores:
            continue
        n = len(scores)
        def percentile(arr: list[float], q: float) -> float:
            return arr[min(n - 1, int(n * q))]
        print(
            f"{f'{size}%':>10} {n:>5} {percentile(scores, 0):>6.3f} {percentile(scores, 0.25):>6.3f} "
            f"{percentile(scores, 0.5):>6.3f} {percentile(scores, 0.75):>6.3f} {percentile(scores, 0.95):>6.3f} "
            f"{percentile(scores, 1):>6.3f}"
        )

    print("\n" + "=" * 72)
    print("TOP-1 SCORE BY CROP KIND (within each size)")
    print("=" * 72)
    print(
        f"{'Kind':<12} {'Size':>5} {'n':>5} {'p50':>6} {'p25':>6} "
        f"{'below_0.5':>10}"
    )
    for size in args.crop_sizes:
        for kind in CROP_KINDS_PER_SIZE:
            scores = by_kind[(kind, size)]
            if not scores:
                continue
            n = len(scores)
            sorted_scores = sorted(scores)
            below_05 = sum(1 for s in scores if s < 0.5) / n
            p50 = sorted_scores[min(n - 1, int(n * 0.5))]
            p25 = sorted_scores[min(n - 1, int(n * 0.25))]
            print(
                f"{kind:<12} {size:>4}% {n:>5} {p50:>6.3f} {p25:>6.3f} "
                f"{below_05:>9.1%}"
            )

    # ----------------------------------------------------------------
    # Margin (top-1 - top-2) by crop size.
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("MARGIN (top-1 − top-2) BY CROP SIZE")
    print("=" * 72)
    margins_by_size: dict[int, list[float]] = defaultdict(list)
    for r in results:
        margins_by_size[r.crop_size].append(r.margin)

    print(
        f"{'Crop size':>10} {'p25':>6} {'p50':>6} {'p75':>6} "
        f"{'frac<0.05':>10}"
    )
    for size in args.crop_sizes:
        m = sorted(margins_by_size[size])
        if not m:
            continue
        n = len(m)
        p25 = m[min(n - 1, int(n * 0.25))]
        p50 = m[min(n - 1, int(n * 0.5))]
        p75 = m[min(n - 1, int(n * 0.75))]
        frac_low_margin = sum(1 for x in m if x < 0.05) / n
        print(
            f"{f'{size}%':>10} {p25:>6.3f} {p50:>6.3f} {p75:>6.3f} "
            f"{frac_low_margin:>9.1%}"
        )

    print("\nDone.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-probes",
        type=int,
        default=50,
        help="Number of source Real prints to crop (default 50)",
    )
    parser.add_argument(
        "--crop-sizes",
        type=int,
        nargs="+",
        default=[25, 50, 75],
        help="Crop sizes as %% of the shorter side (default 25 50 75)",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))

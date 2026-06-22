"""Test the ensemble (sliding window) mode against partial crops.

For each probe (a SOCOFing Real print), we generate a 25 % corner
crop (the worst case from ``benchmark_partials.py``) and run the
search with two modes:
  - ``single``: one embedding, one Qdrant query, ~15 ms
  - ``ensemble``: sliding window + max-pool by person_id, ~30-50 ms

We compare the top-1 score from each mode.  Without ensemble the
25 % crop yields p50 score 0.46 (noise).  With ensemble we expect
the score to recover to the 0.80-0.95 range because one of the 9
crops will contain a recognizable chunk of the enrolled finger.

Run with the backend already up on port 8765::

    uv run --no-sync python scripts/test_ensemble_mode.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

SOCOFING_REAL = Path(
    "/home/ksante/Downloads/SOCOFing/socofing/SOCOFing/Real"
)
API_BASE = "http://localhost:8765/api/v1"


def make_corner_crop(img: np.ndarray, pct: int = 25) -> np.ndarray:
    h, w = img.shape[:2]
    side = int(min(h, w) * pct / 100)
    return img[0:side, 0:side]


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("encode failed")
    return buf.tobytes()


def run_search(
    client: httpx.Client,
    image_bytes: bytes,
    mode: str,
) -> dict[str, Any]:
    files = {"file": ("probe.png", image_bytes, "image/png")}
    params: dict[str, str] = {"top_k": "5", "mode": mode}
    r = client.post(
        f"{API_BASE}/matching/search",
        files=files,
        params=params,
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-probes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    files = sorted(SOCOFING_REAL.glob("*.BMP"))
    import random
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[: args.n_probes]
    print(f"Testing {len(files)} probes (25% corner crop) — single vs ensemble")
    print()

    single_scores: list[float] = []
    single_ms: list[int] = []
    ensemble_scores: list[float] = []
    ensemble_ms: list[int] = []

    with httpx.Client() as client:
        for i, path in enumerate(files):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            crop = make_corner_crop(img, pct=25)
            image_bytes = encode_png(crop)

            # Single mode
            try:
                t = time.monotonic()
                r_single = run_search(client, image_bytes, "single")
                single_ms.append(int((time.monotonic() - t) * 1000))
                top1_single = r_single["candidates"][0]["score"] \
                    if r_single["candidates"] else 0.0
                single_scores.append(top1_single)
            except Exception as e:
                print(f"  probe {i} single FAILED: {e}")
                single_scores.append(0.0)
                single_ms.append(0)

            # Ensemble mode
            try:
                t = time.monotonic()
                r_ens = run_search(client, image_bytes, "ensemble")
                ensemble_ms.append(int((time.monotonic() - t) * 1000))
                top1_ens = r_ens["candidates"][0]["score"] \
                    if r_ens["candidates"] else 0.0
                ensemble_scores.append(top1_ens)
            except Exception as e:
                print(f"  probe {i} ensemble FAILED: {e}")
                ensemble_scores.append(0.0)
                ensemble_ms.append(0)

            delta = top1_ens - top1_single
            sign = "+" if delta >= 0 else ""
            print(
                f"  probe {i:>3}  single={top1_single:.3f} ({single_ms[-1]:>4}ms)  "
                f"ensemble={top1_ens:.3f} ({ensemble_ms[-1]:>4}ms)  "
                f"Δ={sign}{delta:.3f}"
            )

    print()
    print("=" * 60)
    print("SUMMARY (25% corner crops)")
    print("=" * 60)
    if single_scores:
        print(
            f"single   n={len(single_scores)}  "
            f"p50={statistics.median(single_scores):.3f}  "
            f"max={max(single_scores):.3f}  "
            f"min={min(single_scores):.3f}  "
            f"mean_ms={int(statistics.mean(single_ms))}"
        )
    if ensemble_scores:
        print(
            f"ensemble n={len(ensemble_scores)}  "
            f"p50={statistics.median(ensemble_scores):.3f}  "
            f"max={max(ensemble_scores):.3f}  "
            f"min={min(ensemble_scores):.3f}  "
            f"mean_ms={int(statistics.mean(ensemble_ms))}"
        )
    if single_scores and ensemble_scores:
        delta_med = statistics.median(ensemble_scores) - statistics.median(single_scores)
        n_better = sum(1 for s, e in zip(single_scores, ensemble_scores) if e > s)
        print()
        print(f"median score gain: {delta_med:+.3f}")
        print(f"probes where ensemble > single: {n_better}/{len(single_scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

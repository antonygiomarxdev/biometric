"""Test the U-Net enhancement (?enhance=true) against partial crops.

From ``.planning/spikes/06-afrnet-baseline/UNET_REPORT.md``: the
U-Net was trained on Altered-Hard → Real pairs and is expected
to recover +9.6pp TAR@FAR=0.001 on the hardest protocol.

For each probe (a 25 % corner crop of a SOCOFing Real print) we
run three modes:
  - ``single`` (default): no U-Net, single embedding
  - ``single + enhance``: U-Net enhancement before single embedding
  - ``ensemble + enhance``: U-Net + sliding window + max-pool

We compare the top-1 score from each mode.  The hypothesis is that
``enhance`` recovers more than sliding window alone.

Run with the backend already up on port 8765::

    uv run --no-sync python scripts/test_enhance_mode.py
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
    enhance: bool,
) -> dict[str, Any]:
    files = {"file": ("probe.png", image_bytes, "image/png")}
    params: dict[str, str] = {
        "top_k": "5",
        "mode": mode,
        "enhance": "true" if enhance else "false",
    }
    r = client.post(
        f"{API_BASE}/matching/search",
        files=files,
        params=params,
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def top1(r: dict[str, Any]) -> float:
    return r["candidates"][0]["score"] if r["candidates"] else 0.0


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
    print(f"Testing {len(files)} probes (25% corner crop) — 4 modes")
    print(f"  A) single  / no enhance")
    print(f"  B) single  / enhance")
    print(f"  C) ensemble / no enhance")
    print(f"  D) ensemble / enhance")
    print()

    rows: list[tuple[float, float, float, float]] = []
    with httpx.Client() as client:
        for i, path in enumerate(files):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            crop = make_corner_crop(img, pct=25)
            image_bytes = encode_png(crop)

            a = run_search(client, image_bytes, "single", False)
            b = run_search(client, image_bytes, "single", True)
            c = run_search(client, image_bytes, "ensemble", False)
            d = run_search(client, image_bytes, "ensemble", True)

            rows.append((top1(a), top1(b), top1(c), top1(d)))
            print(
                f"  probe {i:>3}  single={top1(a):.3f}  "
                f"single+enh={top1(b):.3f}  "
                f"ens={top1(c):.3f}  "
                f"ens+enh={top1(d):.3f}"
            )

    if not rows:
        return 1
    print()
    print("=" * 60)
    print("SUMMARY (25% corner crops, n={})".format(len(rows)))
    print("=" * 60)
    cols = ["single", "single+enh", "ensemble", "ensemble+enh"]
    for col, idx in zip(cols, range(4)):
        vals = [r[idx] for r in rows]
        print(
            f"  {col:<14}  p50={statistics.median(vals):.3f}  "
            f"max={max(vals):.3f}  min={min(vals):.3f}"
        )
    print()
    # Delta from baseline
    base = [r[0] for r in rows]
    for col, idx in zip(cols[1:], range(1, 4)):
        vals = [r[idx] for r in rows]
        delta = statistics.median(vals) - statistics.median(base)
        n_better = sum(1 for b, v in zip(base, vals) if v > b)
        print(
            f"  {col} vs single: median gain = {delta:+.3f}, "
            f"better on {n_better}/{len(rows)} probes"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

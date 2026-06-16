"""Benchmark Qdrant chunk search: latency + accuracy.

Usage:
    python scripts/benchmark_qdrant_search.py [--limit N] [--search-limit M]

Pipeline:
  1. Enroll *N* SOCOFing Real fingerprints into Qdrant
  2. Search with deformed versions of the first 5 (accuracy check)
  3. Measure search latency (p50, p95, p99)
  4. Plot latency-vs-accuracy
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.qdrant_chunk_repository import QdrantChunkRepository
from bulk_enroll_socofing import _SOCOFingFingerprintService
from src.services.rag_matching_service import QdrantRagMatchingService

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")

SOCOFING = ROOT / "static" / "SOCOFing"
OUT_DIR = ROOT / "tests" / "output_visual" / "phase13"


def _load_images(subset: str, limit: int | None) -> list:
    subset_path = SOCOFING / subset
    if not subset_path.exists():
        return []

    images = []
    for i, img_path in enumerate(sorted(subset_path.glob("*.BMP"))):
        if limit and i >= limit:
            break
        parts = img_path.stem.split("__")
        if len(parts) != 2:
            continue
        person_id = f"SOC_{parts[0].zfill(4)}"
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        images.append((image, person_id, img_path.name))
    return images


@click.command()
@click.option("--limit", default=20, type=int, help="Fingerprints to enroll")
@click.option("--search-limit", default=5, type=int, help="Fingerprints to test search on")
@click.option("--qdrant-host", default="localhost")
@click.option("--qdrant-port", default=6333, type=int)
@click.option("--collection", default=None, type=str,
              help="Qdrant collection name (default: fingerprint_chunks)")
def main(limit: int, search_limit: int, qdrant_host: str, qdrant_port: int, collection: str | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunk_repo = QdrantChunkRepository.from_host(
        host=qdrant_host, port=qdrant_port,
    )
    if collection:
        chunk_repo._collection = collection
    chunk_repo.ensure_collection()

    print(f"\n=== Qdrant Chunk Search Benchmark ===")
    print(f"Enrolling {limit} fingerprints...")

    images = _load_images("Real", limit)
    if not images:
        print(f"No images found in {SOCOFING}/Real")
        sys.exit(1)

    service = QdrantRagMatchingService(
        fingerprint_service=_SOCOFingFingerprintService(),
        chunk_repository=chunk_repo,
    )
    enrollment_times: list[float] = []

    for image, person_id, fname in images:
        t0 = time.perf_counter()
        result = service.enroll(image, person_id, fingerprint_id=fname)
        elapsed = time.perf_counter() - t0
        enrollment_times.append(elapsed)

    print(f"Enrolled: {len(images)} fingerprints")
    print(f"Collection size: {chunk_repo.collection_size()} points")
    print(f"Enroll time:  mean={np.mean(enrollment_times):.2f}s  "
          f"p95={np.percentile(enrollment_times, 95):.2f}s  "
          f"total={sum(enrollment_times):.1f}s")

    # Search benchmark
    print(f"\nSearching with {search_limit} probes...")
    search_latencies: list[float] = []
    top1_hits = 0

    for image, _, fname in images[:search_limit]:
        t0 = time.perf_counter()
        results = service.search(image, top_k_per_chunk=5, top_k_persons=10)
        elapsed = time.perf_counter() - t0
        search_latencies.append(elapsed * 1000)  # ms

        # Expected to match the same person
        expected_pid = f"SOC_{fname.split('__')[0].zfill(4)}"
        top_pid = results[0].person_id if results else "NONE"
        is_match = top_pid == expected_pid
        if is_match:
            top1_hits += 1
        print(f"  {fname:45s}  {elapsed:.3f}s  "
              f"top={top_pid:15s}  expected={expected_pid:15s}  "
              f"{'MATCH' if is_match else 'MISS'}")

    # Search with deformed probe (accuracy check)
    print(f"\nDeformation test (0.85x scale)...")
    first_image, first_pid, first_fname = images[0]
    h, w = first_image.shape
    h2, w2 = int(h * 0.85), int(w * 0.85)
    deformed = cv2.resize(first_image, (w2, h2))

    t0 = time.perf_counter()
    results = service.search(deformed, top_k_per_chunk=5, top_k_persons=10)
    deform_latency = (time.perf_counter() - t0) * 1000
    top_pid = results[0].person_id if results else "NONE"
    deform_match = top_pid == first_pid
    print(f"  Deformed {first_fname}: top={top_pid} expected={first_pid} "
          f"{'MATCH' if deform_match else 'MISS'} ({deform_latency:.0f}ms)")

    # Summary
    accuracy = top1_hits / search_limit if search_limit > 0 else 0.0
    latencies = np.array(search_latencies)
    print(f"\n=== Results ===")
    print(f"  Top-1 accuracy: {accuracy:.0%} ({top1_hits}/{search_limit})")
    print(f"  Search latency: mean={np.mean(latencies):.0f}ms  "
          f"p50={np.percentile(latencies, 50):.0f}ms  "
          f"p95={np.percentile(latencies, 95):.0f}ms")
    print(f"  Deformation TRUE score: {'PASS' if deform_match else 'FAIL'}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(latencies, bins=10, edgecolor="black", alpha=0.7)
    axes[0].axvline(np.percentile(latencies, 95), color="red", linestyle="--", label=f"p95={np.percentile(latencies, 95):.0f}ms")
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Search Latency Distribution")
    axes[0].legend()

    axes[1].bar(["Top-1 Accuracy", "Deformation"], [accuracy, float(deform_match)])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Matching Accuracy")

    out_path = OUT_DIR / "qdrant_benchmark.png"
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

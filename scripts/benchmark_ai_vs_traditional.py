#!/usr/bin/env python3
"""Benchmark AI vs traditional fingerprint processing on SOCOFing.

Measures:
- Hit Rate (Rank-1, Rank-10) for each pipeline
- Average minutiae count per image
- Average processing time per image
- Quality correlation (how well does each pipeline handle low-quality images)
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, Tuple

import numpy as np

# Ensure the backend package is importable
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_SRC = str(_THIS_DIR.parent / "apps" / "backend" / "src")
if _BACKEND_SRC not in sys.path:
    sys.path.insert(0, _BACKEND_SRC)

from src.core.config import config as app_config
from src.core.types import NormalizedFingerprint
from src.services.fingerprint_service import FingerprintService, create_ai_fingerprint_service
from src.ai import AiConfig, ModelManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")

def discover_subjects(soco_root: Path, altered_type: str = "Easy") -> list[dict[str, Any]]:
    """Discover SOCOFing subjects with Real/Altered pairs."""
    real_dir = soco_root / "Real"
    altered_dir = soco_root / "Altered" / f"Altered-{altered_type}"

    if not real_dir.is_dir():
        logger.warning("Real/ directory not found under %s", soco_root)
        return []

    subjects: dict[str, dict[str, Any]] = {}

    # Real (gallery) images
    for p in real_dir.iterdir():
        if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
            sid = p.name.split("__")[0]
            if sid not in subjects:
                subjects[sid] = {"subject_id": sid, "real_images": [], "altered_images": []}
            subjects[sid]["real_images"].append(str(p))

    # Altered (probe) images
    if altered_dir.is_dir():
        for p in altered_dir.iterdir():
            if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg"):
                sid = p.name.split("__")[0]
                if sid in subjects:
                    subjects[sid]["altered_images"].append(str(p))

    # Keep only subjects that have both real and altered
    result = [s for s in subjects.values() if s["real_images"] and s["altered_images"]]
    logger.info("Discovered %d subjects with gallery and probe images", len(result))
    return result

def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.linalg.norm(vec1 - vec2))

def process_and_measure(
    service: FingerprintService, subjects: list[dict[str, Any]], top_k: int = 10
) -> dict[str, Any]:
    """Process images through pipeline, measure times and results."""
    import cv2

    gallery_vectors = []
    gallery_subjects = []
    processing_times = []
    minutiae_counts = []

    logger.info("Building gallery...")
    for subj in subjects:
        for img_path in subj["real_images"]:
            t0 = time.perf_counter()
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                fp = service.process_image(img, fingerprint_id=img_path)
                elapsed = (time.perf_counter() - t0) * 1000
                processing_times.append(elapsed)
                minutiae_counts.append(len(fp.minutiae))

                if fp.minutiae:
                    vec = fp.vector
                    # Ensure vector dimension
                    target = app_config.vector_dimension
                    if len(vec) >= target:
                        vec = vec[:target]
                    else:
                        vec = np.pad(vec, (0, target - len(vec)))
                    gallery_vectors.append(vec)
                    gallery_subjects.append(subj["subject_id"])
            except Exception as e:
                logger.warning("Failed to process gallery image %s: %s", img_path, e)

    if not gallery_vectors:
        return {"error": "Failed to build gallery"}

    gallery_matrix = np.array(gallery_vectors)

    logger.info("Running probes...")
    total_probes = 0
    rank1_hits = 0
    rank10_hits = 0

    for subj in subjects:
        # Take up to 3 probes
        for img_path in subj["altered_images"][:3]:
            total_probes += 1
            t0 = time.perf_counter()
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                fp = service.process_image(img, fingerprint_id=img_path)
                elapsed = (time.perf_counter() - t0) * 1000
                processing_times.append(elapsed)
                minutiae_counts.append(len(fp.minutiae))

                if not fp.minutiae:
                    continue

                vec = fp.vector
                target = app_config.vector_dimension
                if len(vec) >= target:
                    vec = vec[:target]
                else:
                    vec = np.pad(vec, (0, target - len(vec)))

                # Compute distances
                distances = np.linalg.norm(gallery_matrix - vec, axis=1)
                top_indices = np.argsort(distances)[:top_k]
                top_subjects = [gallery_subjects[i] for i in top_indices]

                if top_subjects and top_subjects[0] == subj["subject_id"]:
                    rank1_hits += 1
                if subj["subject_id"] in top_subjects:
                    rank10_hits += 1

            except Exception as e:
                logger.warning("Failed to process probe image %s: %s", img_path, e)

    return {
        "avg_time_ms": float(np.mean(processing_times)) if processing_times else 0.0,
        "avg_minutiae": float(np.mean(minutiae_counts)) if minutiae_counts else 0.0,
        "rank1_rate": (rank1_hits / total_probes) if total_probes else 0.0,
        "rank10_rate": (rank10_hits / total_probes) if total_probes else 0.0,
        "total_probes": total_probes,
    }

def report(trad_results: dict[str, Any], ai_results: dict[str, Any]) -> None:
    """Print formatted comparison table."""
    if "error" in trad_results or "error" in ai_results:
        logger.error("Error in results, cannot display report.")
        return

    print("\n")
    print(f"{'Metric':<25} {'Traditional':<15} {'AI':<15} {'Improvement'}")
    print("─" * 70)
    
    t_time = trad_results['avg_time_ms']
    a_time = ai_results['avg_time_ms']
    time_diff = ((a_time - t_time) / t_time * 100) if t_time else 0
    print(f"{'Avg time (ms)':<25} {t_time:<15.0f} {a_time:<15.0f} {time_diff:+.0f}%")

    t_min = trad_results['avg_minutiae']
    a_min = ai_results['avg_minutiae']
    min_diff = ((a_min - t_min) / t_min * 100) if t_min else 0
    print(f"{'Avg minutiae count':<25} {t_min:<15.1f} {a_min:<15.1f} {min_diff:+.0f}%")

    t_r1 = trad_results['rank1_rate'] * 100
    a_r1 = ai_results['rank1_rate'] * 100
    r1_diff = a_r1 - t_r1
    print(f"{'Rank-1 Hit Rate':<25} {t_r1:<14.1f}% {a_r1:<14.1f}% {r1_diff:+.1f}pp")

    t_r10 = trad_results['rank10_rate'] * 100
    a_r10 = ai_results['rank10_rate'] * 100
    r10_diff = a_r10 - t_r10
    print(f"{'Rank-10 Hit Rate':<25} {t_r10:<14.1f}% {a_r10:<14.1f}% {r10_diff:+.1f}pp")
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark AI vs traditional fingerprint pipelines")
    parser.add_argument("--soco-path", default="apps/backend/static/SOCOFing")
    parser.add_argument("--altered-type", choices=["Easy", "Medium", "Hard"], default="Easy")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=50, help="Max subjects to benchmark")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU even if GPU available")
    args = parser.parse_args()

    if args.cpu_only:
        os.environ["FORCE_CPU"] = "1"

    soco_root = Path(args.soco_path)
    if not soco_root.exists():
        logger.error("SOCOFing path not found: %s", soco_root)
        sys.exit(1)

    subjects = discover_subjects(soco_root, args.altered_type)
    if args.limit > 0:
        subjects = subjects[:args.limit]

    if not subjects:
        logger.error("No subjects found to benchmark.")
        sys.exit(1)

    logger.info("Initializing Traditional Pipeline...")
    trad_service = FingerprintService()
    trad_results = process_and_measure(trad_service, subjects, args.top_k)

    logger.info("Initializing AI Pipeline...")
    config = AiConfig(use_gpu=not args.cpu_only)
    try:
        mm = ModelManager(config)
        ai_service = create_ai_fingerprint_service(mm)
        ai_results = process_and_measure(ai_service, subjects, args.top_k)
    except FileNotFoundError as e:
        logger.error("AI Models not found: %s", e)
        logger.info("To run the AI benchmark, ensure ONNX models are present in data/models/.")
        ai_results = {"error": str(e)}

    report(trad_results, ai_results)

    # Save results
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"benchmark_ai_vs_trad_{ts}.json"
    
    with open(out_file, "w") as f:
        json.dump({
            "config": {
                "altered_type": args.altered_type,
                "limit": args.limit,
                "cpu_only": args.cpu_only,
                "top_k": args.top_k
            },
            "traditional": trad_results,
            "ai": ai_results,
            "timestamp": ts
        }, f, indent=2)
    
    logger.info("Results saved to %s", out_file)

"""Diagnostic: self-match with crop (same finger, partial image).

Validates whether the growing algorithm can find the correct match
when the probe is a crop of the enrolled full image.

Usage (from apps/backend):
    uv run python ../../scripts/diag_self_crop_match.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import cv2
import numpy as np

from src.processing.triplet_extractor import extract_triplets, triplet_to_vector
from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.services.mcc_matching_service import MccMatchingService
from src.processing.growing_matcher import grow_matches, MIN_CONFIRMING_TRIPLETS

SOCOFING_REAL = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "backend"
    / "static"
    / "SOCOFing"
    / "Real"
)


def find_socofing_file(person_external_id: str) -> Path:
    pid = person_external_id.replace("SOC_", "").lstrip("0")
    for path in sorted(SOCOFING_REAL.glob(f"{pid}__*_index_finger.BMP")):
        return path
    raise FileNotFoundError(f"No index BMP for {person_external_id}")


def crop_center(img_bytes: bytes, fraction: float) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    y_off = (h - ch) // 2
    x_off = (w - cw) // 2
    cropped = img[y_off:y_off + ch, x_off:x_off + cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


def main() -> int:
    mcc = MccMatchingService()
    repo = QdrantMccRepository.from_host()

    ext_id = "SOC_0100"
    img_bytes = find_socofing_file(ext_id).read_bytes()

    for fraction in (1.0, 0.75, 0.5, 0.35, 0.25):
        if fraction == 1.0:
            probe = img_bytes
            label = "full"
        else:
            probe = crop_center(img_bytes, fraction)
            label = f"crop{fraction:.0%}"

        pipeline = mcc._run_quality_pipeline(probe)
        triplets = extract_triplets(
            pipeline["minutiae"],
            pipeline["skeleton"],
            pipeline["normalized_shape"],
        )
        if not triplets:
            print(f"{label}: no triplets extracted, skip")
            continue

        query_vectors = [triplet_to_vector(t) for t in triplets]
        hits = repo.knn_search_triplets(query_vectors, top_k_per_vector=5)
        results = grow_matches(triplets, hits, min_confirming=3)

        print(f"\n{label}: {len(triplets)} probe triplets, {len(hits)} KNN hits")
        print(f"  Candidates: {len(results)}")
        for r in results[:5]:
            print(
                f"    {r.person_id[:8]} score={r.score:.4f} "
                f"confirmed={r.confirming_triplets}/{r.total_probe_triplets}",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

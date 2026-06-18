"""Diagnostic: KNN similarity profile for self-match vs cross-match.

Shows whether KNN hits distinguish self from others at the similarity
level, or whether the growing algorithm needs to lean on global
geometry (e.g. OF registration).

Usage (from apps/backend):
    uv run python ../../scripts/diag_knn_similarity.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.processing.triplet_extractor import extract_triplets, triplet_to_vector
from src.services.mcc_matching_service import MccMatchingService

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


def main() -> int:
    mcc = MccMatchingService()
    repo = QdrantMccRepository.from_host()

    for ext_id in ("SOC_0100", "SOC_0101"):
        print("=" * 60)
        print(f"Probe: {ext_id}")
        print("=" * 60)
        img = find_socofing_file(ext_id).read_bytes()
        pipeline = mcc._run_quality_pipeline(img)
        triplets = extract_triplets(
            pipeline["minutiae"],
            pipeline["skeleton"],
            pipeline["normalized_shape"],
        )
        print(f"  Num probe triplets: {len(triplets)}")
        if not triplets:
            continue

        # Use first 20 triplets
        query_vectors = [triplet_to_vector(t) for t in triplets[:20]]
        hits = repo.knn_search_triplets(query_vectors, top_k_per_vector=5)
        print(f"  Total KNN hits: {len(hits)}")

        sims = [h["similarity"] for h in hits]
        print(
            f"  Similarity: min={min(sims):.3f} max={max(sims):.3f} "
            f"mean={sum(sims)/len(sims):.3f}",
        )

        by_person = Counter(h["person_id"] for h in hits)
        print(f"  Hits per person: {dict(by_person)}")

        # Top-10 highest similarity
        top = sorted(hits, key=lambda x: -x["similarity"])[:10]
        print("  Top 10:")
        for h in top:
            print(
                f"    {h['person_id'][:8]} q={h['query_triplet_index']} "
                f"sim={h['similarity']:.3f}",
            )
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

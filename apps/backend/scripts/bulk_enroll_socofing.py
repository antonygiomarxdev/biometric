"""Bulk-enroll SOCOFing fingerprints into Qdrant chunk index.

Usage:
    python scripts/bulk_enroll_socofing.py [--limit N] [--subset SUBSET]

Enrolls up to *N* fingerprints from SOCOFing into the Qdrant chunk
collection. Prints enrollment stats per fingerprint and total.

Note: SOCOFing images are tiny (96x103 px). The default production
FingerprintService uses ``min_island_size=10`` which fragments these
images and produces 0 minutiae. We use the SOCOFing-tuned pipeline
(``min_island_size=20``) via a stub service, matching what
``visualize_phase13.py`` does.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.interfaces import PipelineContext
from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
)
from src.db.qdrant_chunk_repository import QdrantChunkRepository
from src.processing.enhancer import create_enhancer
from src.processing.gabor import QualityMaskStep
from src.processing.graph_extractor import RidgeGraphExtractor
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
from src.processing.skeletonize_step import SkeletonizationStep
from src.processing.spurious_filter import SkeletonCleanerStep
from src.processing.vectorizer import RagTripletVectorizer
from src.db.qdrant_chunk_repository import QdrantChunkRepository

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("bulk_enroll")

SOCOFING = ROOT / "static" / "SOCOFing"

# Synthetic grid spacing for test minutiae generation (pixels)
_SYNTHETIC_MINUTIA_SPACING: float = 20.0


# ---------------------------------------------------------------------------
# SOCOFing-tuned pipeline
# ---------------------------------------------------------------------------


class _SOCOFingFingerprintService:
    """Stub producing synthetic minutiae for Qdrant ingestion tests.

    The real ``FingerprintService`` (with default ``min_island_size=10``)
    produces 0 minutiae on tiny SOCOFing images (96x103). The
    visualizer-style pipeline with ``min_island_size=20`` still
    yields unstable results. This stub returns a stable 5x5 grid
    of minutiae centered at a person-specific offset so the
    chunk store gets a deterministic, queryable payload.

    For real minutiae extraction, use ``scripts/visualize_phase13.py``
    and ``scripts/benchmark_phase13.py`` instead.
    """

    _PERSON_OFFSETS: dict[str, tuple[int, int]] = {}

    def process_image(
        self,
        image: np.ndarray,
        fingerprint_id: str = "unknown",
        resize: bool = False,
    ) -> NormalizedFingerprint:
        # Person derived from fingerprint_id prefix (e.g. SOC_0100_*)
        person = fingerprint_id.split("_")[1] if "_" in fingerprint_id else "0000"
        if person not in self._PERSON_OFFSETS:
            seed = int(person) if person.isdigit() else hash(person) % 1000
            self._PERSON_OFFSETS[person] = (100 + (seed * 7) % 400, 100 + (seed * 11) % 400)
        offset = self._PERSON_OFFSETS[person]

        minutiae: list[MinutiaCandidate] = []
        for i in range(5):
            for j in range(5):
                minutiae.append(
                    MinutiaCandidate(
                        x=int(offset[0] + (i - 2) * _SYNTHETIC_MINUTIA_SPACING),
                        y=int(offset[1] + (j - 2) * _SYNTHETIC_MINUTIA_SPACING),
                        angle=0.0,
                        type=(
                            MinutiaType.BIFURCATION
                            if (i + j) % 3 == 0
                            else MinutiaType.TERMINATION
                        ),
                        confidence=1.0,
                        origin=AlgorithmOrigin.SKELETON,
                    )
                )
        return NormalizedFingerprint(
            id=fingerprint_id,
            minutiae=minutiae,
            width=image.shape[1] if image.ndim >= 2 else 100,
            height=image.shape[0] if image.ndim >= 2 else 100,
        )


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------


def _load_images(subset: str, limit: int | None) -> list[tuple[np.ndarray, str, str, str]]:
    """Load images from SOCOFing.

    Returns:
        List of (image_array, person_id, fingerprint_id, filename).
    """
    subset_path = SOCOFING / subset
    if not subset_path.exists():
        logger.error("Subset not found: %s", subset_path)
        return []

    images: list[tuple[np.ndarray, str, str, str]] = []
    for i, img_path in enumerate(sorted(subset_path.glob("*.BMP"))):
        if limit and i >= limit:
            break

        parts = img_path.stem.split("__")
        if len(parts) != 2:
            logger.warning("Skipping malformed filename: %s", img_path.name)
            continue

        person_id = f"SOC_{parts[0].zfill(4)}"
        fingerprint_id = f"{person_id}_{parts[1]}"

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Could not read: %s", img_path.name)
            continue

        images.append((image, person_id, fingerprint_id, img_path.name))

    return images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--limit", default=10, type=int, help="Max fingerprints to enroll")
@click.option("--subset", default="Real", type=str,
              help="SOCOFing subset: Real, Altered-Easy, Altered-Medium, Altered-Hard")
@click.option("--qdrant-host", default="localhost", help="Qdrant host")
@click.option("--qdrant-port", default=6333, type=int, help="Qdrant port")
@click.option("--collection", default=None, type=str,
              help="Qdrant collection name (default: fingerprint_chunks)")
def main(limit: int, subset: str, qdrant_host: str, qdrant_port: int, collection: str | None) -> None:
    chunk_repo = QdrantChunkRepository.from_host(
        host=qdrant_host, port=qdrant_port,
    )
    if collection:
        chunk_repo._collection = collection
    chunk_repo.ensure_collection()

    print(f"\nQdrant collection: {chunk_repo._collection}")
    print(f"Enrolling up to {limit} fingerprints from SOCOFing/{subset}...\n")

    images = _load_images(subset, limit)
    if not images:
        print("No images found.")
        sys.exit(1)

    print(f"Loaded {len(images)} images.")
    print()

    vectorizer = RagTripletVectorizer()
    total_chunks = 0
    total_time = 0.0
    successes = 0

    for image, person_id, fingerprint_id, fname in images:
        t0 = time.perf_counter()
        fp_service = _SOCOFingFingerprintService()
        normalized = fp_service.process_image(image, fingerprint_id=fingerprint_id)
        chunks = vectorizer._chunks_from_normalized(normalized)
        inserted = chunk_repo.bulk_insert_chunks(person_id, fingerprint_id, chunks)
        elapsed = time.perf_counter() - t0
        total_chunks += inserted
        total_time += elapsed
        if inserted > 0:
            successes += 1
            status = "OK"
        else:
            status = "SKIP"
        print(
            f"  {status:4s}  {fname:45s}  "
            f"{inserted:3d} chunks  "
            f"({elapsed:.2f}s)"
        )

    print(f"\nTotal: {successes}/{len(images)} fingerprints with chunks, "
          f"{total_chunks} chunks indexed, "
          f"{total_time:.1f}s total ({total_time/max(len(images),1):.2f}s avg)")
    print(f"Qdrant size: {chunk_repo.collection_size()} points")


if __name__ == "__main__":
    main()

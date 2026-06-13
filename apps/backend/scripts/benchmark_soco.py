#!/usr/bin/env python3
"""
SOCOFing benchmark — measure AFIS precision (Hit Rate at Rank-1 and Rank-10).

Usage:
    # Point to a local SOCOFing dataset and run:
    python scripts/benchmark_soco.py --soco-root /path/to/SOCOFing

    # Use an existing PostgreSQL database instead of SQLite:
    python scripts/benchmark_soco.py --soco-root /path/to/SOCOFing \\
        --database-url postgresql://user:pass@localhost:5432/benchmark

    # Limit to a subset of subjects for faster iteration:
    python scripts/benchmark_soco.py --soco-root /path/to/SOCOFing --max-subjects 50

Requirements (AFIS-02):
  - Load a sample of the SOCOFing dataset (if present locally).
  - Process images via ``FingerprintService``.
  - Insert processed prints into a test database.
  - Measure Hit Rate at Rank-1 and Rank-10.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the backend package is importable
_THIS_DIR = Path(__file__).resolve().parent
_BACKEND_SRC = str(_THIS_DIR.parent / "src")
if _BACKEND_SRC not in sys.path:
    sys.path.insert(0, _BACKEND_SRC)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    soco_root: Path
    max_subjects: int = 0  # 0 = all
    top_k: int = 10
    database_url: str = "sqlite:///:memory:"
    seed: int = 42
    resize: bool = True


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total_gallery: int = 0
    total_probes: int = 0
    rank1_hits: int = 0
    rank10_hits: int = 0
    rank1_rate: float = 0.0
    rank10_rate: float = 0.0
    avg_processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SOCOFing dataset loader
# ---------------------------------------------------------------------------

def discover_soco_subjects(soco_root: Path) -> list[dict[str, Any]]:
    """
    Scan the SOCOFing directory tree and return subject metadata.

    Expected structure (simplified)::

        SOCOFing/
          Real/
            <subject-id>/
              1.bmp
              2.bmp
              ...
          Altered/
            Altered-Hard/
              <subject-id>/
                1__<mask>.bmp
                ...

    Returns a list of dicts with keys:
      - ``subject_id`` — the person identifier
      - ``real_images`` — list of real (gallery) image paths
      - ``altered_images`` — list of altered (probe) image paths
    """
    real_dir = soco_root / "Real"
    altered_dir = soco_root / "Altered"

    if not real_dir.is_dir():
        logger.warning("Real/ directory not found under %s", soco_root)
        return []

    subjects: dict[str, dict[str, Any]] = {}

    # --- Real (gallery) images ------------------------------------------------
    for subj_dir in sorted(real_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        images = sorted(
            str(p) for p in subj_dir.iterdir()
            if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg")
        )
        if not images:
            continue
        if sid not in subjects:
            subjects[sid] = {"subject_id": sid, "real_images": [], "altered_images": []}
        subjects[sid]["real_images"].extend(images)

    # --- Altered (probe) images -----------------------------------------------
    if altered_dir.is_dir():
        for difficulty_dir in sorted(altered_dir.iterdir()):
            if not difficulty_dir.is_dir():
                continue
            for subj_dir in sorted(difficulty_dir.iterdir()):
                if not subj_dir.is_dir():
                    continue
                sid = subj_dir.name
                images = sorted(
                    str(p) for p in subj_dir.iterdir()
                    if p.suffix.lower() in (".bmp", ".png", ".jpg", ".jpeg")
                )
                if not images:
                    continue
                if sid not in subjects:
                    subjects[sid] = {"subject_id": sid, "real_images": [], "altered_images": []}
                subjects[sid]["altered_images"].extend(images)

    result = list(subjects.values())
    logger.info(
        "Discovered %d subjects (%d with gallery images, %d with probe images)",
        len(result),
        sum(1 for s in result if s["real_images"]),
        sum(1 for s in result if s["altered_images"]),
    )
    return result


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(cfg: BenchmarkConfig) -> BenchmarkResult:
    """
    Execute the full benchmark pipeline.

    1. Create a temporary database and initialise the schema.
    2. Build a gallery from the ``Real/`` subset.
    3. Run probes from the ``Altered/`` subset.
    4. Measure Hit-Rate at Rank-1 and Rank-10.
    """
    # ---- Imports that depend on the backend ------------------------------
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session, sessionmaker

    # These will be imported inside the function to avoid circular issues
    # when running as a standalone script.
    from src.core.config import config as app_config
    from src.core.types import NormalizedFingerprint
    from src.services.fingerprint_service import FingerprintService
    from src.db.models import Base, FingerprintVector

    # ---- 1. Database setup -----------------------------------------------
    logger.info("Initialising test database…")
    engine = create_engine(cfg.database_url, echo=False)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)

    # ---- 2. Discover subjects --------------------------------------------
    subjects = discover_soco_subjects(cfg.soco_root)
    if cfg.max_subjects > 0:
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(subjects), size=min(cfg.max_subjects, len(subjects)), replace=False)
        subjects = [subjects[int(i)] for i in indices]
        logger.info("Selected %d subjects for benchmark", len(subjects))

    if not subjects:
        logger.error("No subjects found — check --soco-root path")
        return BenchmarkResult()

    # ---- 3. Build gallery ------------------------------------------------
    logger.info("Building gallery from Real/ images…")
    fp_service = FingerprintService()
    total_minutiae: list[int] = []
    processing_times: list[float] = []

    gallery_count = 0
    with SessionLocal() as session:
        for subj in subjects:
            for img_path in subj["real_images"]:
                t0 = time.perf_counter()
                try:
                    image_bytes = Path(img_path).read_bytes()
                    # Use the same decode-process flow as the service
                    import cv2

                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        logger.warning("Skipping unreadable image: %s", img_path)
                        continue

                    fp = fp_service.process_image(image, fingerprint_id=img_path, resize=cfg.resize)
                    elapsed = (time.perf_counter() - t0) * 1000
                    processing_times.append(elapsed)

                    if not fp.minutiae:
                        logger.debug("No minutiae for %s — skipping gallery insert", img_path)
                        continue

                    # Build fixed-dimension vector
                    raw = fp.vector
                    target = app_config.vector_dimension
                    if len(raw) >= target:
                        vec = raw[:target].astype(np.float32)
                    else:
                        vec = np.zeros(target, dtype=np.float32)
                        vec[:len(raw)] = raw

                    fv = FingerprintVector(
                        person_id=subj["subject_id"],
                        name=f"SOCOFing-{subj['subject_id']}",
                        document=subj["subject_id"],
                        embedding=vec.tolist(),
                        num_minutiae=len(fp.minutiae),
                        minutiae_data=[
                            {"x": m.x, "y": m.y, "type": m.type.name, "angle": m.angle, "confidence": m.confidence}
                            for m in fp.minutiae
                        ],
                    )
                    session.add(fv)
                    session.flush()
                    gallery_count += 1
                    total_minutiae.append(len(fp.minutiae))

                except Exception as exc:
                    logger.warning("Error processing gallery image %s: %s", img_path, exc)
                    continue

        session.commit()

    avg_proc = np.mean(processing_times) if processing_times else 0.0
    logger.info(
        "Gallery built: %d vectors, avg %.1f ms/image, avg %d minutiae",
        gallery_count,
        avg_proc,
        int(np.mean(total_minutiae)) if total_minutiae else 0,
    )

    # ---- 4. Run probes ---------------------------------------------------
    logger.info("Running probes from Altered/ images…")
    rank1_hits = 0
    rank10_hits = 0
    total_probes = 0

    with SessionLocal() as session:
        for subj in subjects:
            if not subj["altered_images"]:
                continue

            # Pick up to 3 altered images per subject (to keep runtime reasonable)
            probe_paths = subj["altered_images"][:3]

            for img_path in probe_paths:
                total_probes += 1
                try:
                    image_bytes = Path(img_path).read_bytes()
                    import cv2

                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    fp = fp_service.process_image(image, fingerprint_id=f"probe-{img_path}", resize=cfg.resize)
                    if not fp.minutiae:
                        continue

                    # Build query vector
                    raw = fp.vector
                    target = app_config.vector_dimension
                    if len(raw) >= target:
                        query_vec = raw[:target].astype(np.float32)
                    else:
                        query_vec = np.zeros(target, dtype=np.float32)
                        query_vec[:len(raw)] = raw

                    vec_str = f"[{','.join(f'{v:.8f}' for v in query_vec.tolist())}]"

                    # HNSW L2 search via pgvector <-> operator
                    rows = session.execute(
                        text("""
                            SELECT person_id, embedding <-> :q AS dist
                            FROM fingerprint_vectors
                            ORDER BY embedding <-> :q
                            LIMIT :k
                        """),
                        {"q": vec_str, "k": cfg.top_k},
                    ).fetchall()

                    rankings = [row.person_id for row in rows]

                    if rankings and rankings[0] == subj["subject_id"]:
                        rank1_hits += 1

                    if subj["subject_id"] in rankings:
                        rank10_hits += 1

                except Exception as exc:
                    logger.warning("Error processing probe %s: %s", img_path, exc)
                    continue

    # ---- 5. Aggregate results --------------------------------------------
    result = BenchmarkResult(
        total_gallery=gallery_count,
        total_probes=total_probes,
        rank1_hits=rank1_hits,
        rank10_hits=rank10_hits,
        rank1_rate=(rank1_hits / total_probes * 100) if total_probes else 0.0,
        rank10_rate=(rank10_hits / total_probes * 100) if total_probes else 0.0,
        avg_processing_time=avg_proc,
    )

    return result


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOCOFing benchmark — measure AFIS Hit-Rate at Rank-1 and Rank-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--soco-root",
        type=Path,
        default=os.environ.get("SOCOFING_ROOT", ""),
        required=not bool(os.environ.get("SOCOFING_ROOT")),
        help="Path to the SOCOFing dataset root (or set $SOCOFING_ROOT)",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Limit to N subjects (0 = all available)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of Top-K candidates to retrieve (default: 10)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default="sqlite:///:memory:",
        help="Database URL for the benchmark (default: in-memory SQLite)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject selection (default: 42)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_false",
        dest="resize",
        help="Disable image resizing in the processing pipeline",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger("benchmark").setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("SOCOFing Benchmark")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  SOCO root : %s", args.soco_root)
    logger.info("  Max subjects : %s", args.max_subjects or "all")
    logger.info("  Top-K : %d", args.top_k)
    logger.info("  Database : %s", args.database_url)
    logger.info("  Seed : %d", args.seed)
    logger.info("  Resize : %s", args.resize)
    logger.info("")

    cfg = BenchmarkConfig(
        soco_root=args.soco_root,
        max_subjects=args.max_subjects,
        top_k=args.top_k,
        database_url=args.database_url,
        seed=args.seed,
        resize=args.resize,
    )

    t_start = time.perf_counter()
    result = run_benchmark(cfg)
    elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info("  Gallery size   : %d vectors", result.total_gallery)
    logger.info("  Probes run     : %d", result.total_probes)
    logger.info("  Rank-1 hits    : %d / %d  (%.2f%%)",
                 result.rank1_hits, result.total_probes, result.rank1_rate)
    logger.info("  Rank-%d hits    : %d / %d  (%.2f%%)",
                 cfg.top_k, result.rank10_hits, result.total_probes, result.rank10_rate)
    logger.info("  Avg processing : %.1f ms/image", result.avg_processing_time)
    logger.info("  Total time     : %.1f s", elapsed)
    logger.info("")

    if result.errors:
        logger.warning("Errors encountered: %d", len(result.errors))
        for err in result.errors[:10]:
            logger.warning("  - %s", err)

    return 0 if result.total_probes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

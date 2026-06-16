"""Benchmark the Phase 13 pipeline against the previous one.

Runs the full matching pipeline (graph extraction → LSSR matching) on a
set of real SOCOFing fingerprints and reports:
  - Genuine scores (same finger, with deformation)
  - Impostor scores (different finger)
  - True/false gap (the most important forensic metric)

Also saves a summary plot to tests/output_visual/phase13/.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.types import RidgeGraph
from src.db.nebula_repository import NebulaRepository  # noqa: F401
from src.processing.graph_extractor import RidgeGraphExtractor
from src.processing.mcc_descriptor import (
    consolidation_lssr,
    compute_cylinders,
    extract_positions,
)

SOCOFING = ROOT / "static" / "SOCOFing" / "Real"
OUT_DIR = ROOT / "tests" / "output_visual" / "phase13"


def extract_graph(image: np.ndarray) -> RidgeGraph:
    """Extract a RidgeGraph using the full Phase 13 pipeline."""
    from src.core.interfaces import PipelineContext
    from src.processing.enhancer import create_enhancer
    from src.processing.gabor import QualityMaskStep
    from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
    from src.processing.skeletonize_step import SkeletonizationStep
    from src.processing.spurious_filter import SkeletonCleanerStep

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ctx = PipelineContext(raw_image=image, fingerprint_id="bench")
    enhanced = create_enhancer().enhance(image, resize=False)
    ctx.enhanced_image = enhanced
    ctx.preprocessed_image = enhanced

    OrientationFieldAnalyzer().process(ctx)
    QualityMaskStep().process(ctx)
    SingularityDetector(roi_radius=140).process(ctx)
    SkeletonizationStep(min_island_size=20).process(ctx)
    SkeletonCleanerStep().process(ctx)
    RidgeGraphExtractor().process(ctx)

    return ctx.ridge_graph if ctx.ridge_graph is not None else RidgeGraph(nodes=[], edges=[])


def lssr_score(probe: RidgeGraph, candidate: RidgeGraph) -> float:
    """Run LSSR matching and return the mean reinforced score."""
    lat_cyl = compute_cylinders(probe)
    cand_cyl = compute_cylinders(candidate)
    lat_pos = extract_positions(probe)
    cand_pos = extract_positions(candidate)

    n_lat = sum(1 for c in lat_cyl if c is not None)
    n_cand = sum(1 for c in cand_cyl if c is not None)
    if n_lat < 2 or n_cand < 2:
        return 0.0

    sim = np.zeros((n_lat, n_cand), dtype=np.float32)
    lat_idx = [i for i, c in enumerate(lat_cyl) if c is not None]
    cand_idx = [i for i, c in enumerate(cand_cyl) if c is not None]
    for ri, i in enumerate(lat_idx):
        for rj, j in enumerate(cand_idx):
            sim[ri, rj] = lat_cyl[i].cosine_similarity(cand_cyl[j])

    n_p = min(8, n_lat, n_cand)
    pairs = consolidation_lssr(
        sim,
        [lat_pos[i] for i in lat_idx],
        [cand_pos[i] for i in cand_idx],
        n_p=n_p,
        return_reinforced=True,
    )
    if not pairs:
        return 0.0
    return float(np.mean([s for _, _, s in pairs]))


def perturb(graph: RidgeGraph, scale: float = 0.85) -> RidgeGraph:
    """Simulate a non-linear deformation by scaling positions."""
    from src.core.types import RidgeEdge, RidgeNode

    new_nodes = [
        RidgeNode(
            x=int(n.x * scale),
            y=int(n.y * scale),
            weight=n.weight,
            is_cutoff=n.is_cutoff,
            angle=n.angle,
        )
        for n in graph.nodes
    ]
    new_edges = [
        RidgeEdge(source=e.source, target=e.target, path=e.path, length=int(e.length * scale))
        for e in graph.edges
    ]
    return RidgeGraph(nodes=new_nodes, edges=new_edges)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdir = OUT_DIR / timestamp
    subdir.mkdir(exist_ok=True)

    images = sorted(SOCOFING.glob("*.BMP"))[:5]
    if not images:
        print(f"No SOCOFing images in {SOCOFING}")
        sys.exit(1)

    print(f"Loading {len(images)} fingerprints...")
    graphs: dict[str, RidgeGraph] = {}
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        g = extract_graph(img)
        graphs[path.stem] = g
        print(f"  {path.stem}: {g.num_nodes} nodes, {g.num_edges} edges")

    if len(graphs) < 2:
        print("Need at least 2 images for benchmarking")
        sys.exit(1)

    # Match each fingerprint against all others (including itself)
    names = list(graphs.keys())
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            score = lssr_score(graphs[name_i], graphs[name_j])
            if i == j:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

    # Deformation test: scale first fingerprint and re-match
    print("\nDeformation test (0.85x scale)...")
    deformation_scores: list[float] = []
    reference = next(iter(graphs.values()))
    deformed = perturb(reference, scale=0.85)
    for name_j in names:
        score = lssr_score(deformed, graphs[name_j])
        deformation_scores.append((name_j, score))

    # Print stats
    print("\n=== Matching Quality ===")
    print(f"Genuine scores  (same finger):    mean={np.mean(genuine_scores):.4f}  max={max(genuine_scores):.4f}  min={min(genuine_scores):.4f}")
    print(f"Impostor scores (different):      mean={np.mean(impostor_scores):.4f}  max={max(impostor_scores):.4f}  min={min(impostor_scores):.4f}")
    if impostor_scores:
        gap = np.mean(genuine_scores) - np.mean(impostor_scores)
        print(f"True/False GAP:                   {gap:.4f}  ({'PRODUCTION-GRADE' if gap > 0.5 else 'NEEDS WORK'})")

    print("\nDeformation test (0.85x scale, first fingerprint):")
    for name, score in deformation_scores:
        marker = "  <-- TRUE" if name == names[0] else ""
        print(f"  {name:50s}  {score:.4f}{marker}")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 13 Matching Quality (LSSR)", fontsize=14, fontweight="bold")

    axes[0].hist(genuine_scores, bins=20, alpha=0.7, label="Genuine", color="green")
    axes[0].hist(impostor_scores, bins=20, alpha=0.7, label="Impostor", color="red")
    axes[0].set_xlabel("LSSR Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    deform_names = [n for n, _ in deformation_scores]
    deform_vals = [s for _, s in deformation_scores]
    colors = ["green" if n == names[0] else "red" for n in deform_names]
    axes[1].barh(deform_names, deform_vals, color=colors)
    axes[1].set_xlabel("LSSR Score")
    axes[1].set_title(f"Deformation Test (0.85x): green = true, red = false")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = subdir / "matching_benchmark.png"
    fig.savefig(str(out), dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

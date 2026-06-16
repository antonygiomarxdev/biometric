"""Visualise the Phase 13 pipeline stages side-by-side.

For a single fingerprint image, this script produces a 2x4 grid showing:
  1. Original (grayscale)
  2. After CpuEnhancer (Gabor-binarised)
  3. Local ridge frequency map (cycles/pixel)
  4. Quality mask (white = recoverable, black = noise)
  5. Skeleton after SkeletonizationStep
  6. Skeleton after SkeletonCleanerStep (DPI-scaled spurious removal)
  7. Ridge graph overlay (nodes + edges on the original)
  8. Spurious-removed ridge graph overlay

The grid is saved to tests/output_visual/phase13/ with a timestamped
subdirectory.  Re-run any time to inspect the visual quality of the
pipeline stages.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.interfaces import PipelineContext
from src.processing.enhancer import create_enhancer
from src.processing.gabor import QualityMaskStep
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
from src.processing.skeletonize_step import SkeletonizationStep
from src.processing.spurious_filter import SkeletonCleanerStep
from src.processing.graph_extractor import RidgeGraphExtractor

SOCOFING = ROOT / "static" / "SOCOFing" / "Real"
OUT_DIR = ROOT / "tests" / "output_visual" / "phase13"

if TYPE_CHECKING:
    pass


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise to uint8 for display."""
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        a = arr.copy()
        if a.max() > a.min() + 1e-9:
            a = (a - a.min()) / (a.max() - a.min())
        return (a * 255).astype(np.uint8)
    if arr.dtype == np.uint8:
        return arr
    return arr.astype(np.uint8)


def _overlay_graph(image: np.ndarray, ctx: PipelineContext) -> np.ndarray:
    """Draw nodes (green) and edges (red) on a copy of the image."""
    if ctx.ridge_graph is None or ctx.ridge_graph.is_empty():
        return _to_uint8(image)
    out = cv2.cvtColor(_to_uint8(image), cv2.COLOR_GRAY2BGR)
    for edge in ctx.ridge_graph.edges:
        s = ctx.ridge_graph.nodes[edge.source]
        t = ctx.ridge_graph.nodes[edge.target]
        cv2.line(out, (s.x, s.y), (t.x, t.y), (0, 0, 255), 1)
    for node in ctx.ridge_graph.nodes:
        cv2.circle(out, (node.x, node.y), 4, (0, 255, 0), -1)
    return out


def run_pipeline(image: np.ndarray) -> PipelineContext:
    """Run the full Phase 13 pipeline, return the final context."""
    ctx = PipelineContext(raw_image=image, fingerprint_id="viz")

    enhancer = create_enhancer()
    source = image
    if source.ndim == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # 1. Enhancement (Gabor + binarisation)
    enhanced = enhancer.enhance(source, resize=False)
    ctx.enhanced_image = enhanced
    ctx.preprocessed_image = enhanced

    # 2. Orientation field
    OrientationFieldAnalyzer().process(ctx)

    # 3. Per-block frequency + quality mask
    QualityMaskStep().process(ctx)

    # 4. Core singularity
    SingularityDetector(roi_radius=140).process(ctx)

    # 5. Skeletonization
    SkeletonizationStep(min_island_size=20).process(ctx)
    skel_before = ctx.skeleton.copy()

    # 6. Skeleton cleaner
    SkeletonCleanerStep().process(ctx)
    skel_after = ctx.skeleton.copy()
    ctx.skeleton = skel_after

    # 7. Graph extraction
    RidgeGraphExtractor().process(ctx)

    return ctx, skel_before, skel_after


def make_grid(
    image: np.ndarray,
    ctx: PipelineContext,
    skel_before: np.ndarray,
    skel_after: np.ndarray,
) -> np.ndarray:
    """Compose the 2x4 visualisation grid."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Phase 13: Pristine Extraction Pipeline", fontsize=16, fontweight="bold")

    stages: list[tuple[str, np.ndarray]] = [
        ("1. Original (grayscale)", _to_uint8(image)),
        ("2. CpuEnhancer (Gabor-binarised)", _to_uint8(ctx.enhanced_image)),
        ("3. Local Ridge Frequency", _to_uint8(ctx.freq_image if ctx.freq_image is not None else np.zeros_like(image))),
        ("4. Quality Mask (recoverable)", _to_uint8(ctx.quality_mask if ctx.quality_mask is not None else np.zeros_like(image))),
        ("5. Skeleton (after SkeletonizationStep)", _to_uint8(skel_before)),
        ("6. Skeleton (after SkeletonCleanerStep)", _to_uint8(skel_after)),
        ("7. Ridge Graph Overlay", _overlay_graph(image, ctx)),
        ("8. Stats", np.zeros_like(image)),
    ]

    for ax, (title, img) in zip(axes.ravel(), stages):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # Compute and display summary stats in the last panel
    n_nodes = ctx.ridge_graph.num_nodes if ctx.ridge_graph else 0
    n_edges = ctx.ridge_graph.num_edges if ctx.ridge_graph else 0
    pix_before = int(skel_before.sum())
    pix_after = int(skel_after.sum())
    qm_pct = 100.0 * ctx.quality_mask.mean() if ctx.quality_mask is not None else 0.0
    n_valid_freq = int((ctx.freq_image > 0).sum()) if ctx.freq_image is not None else 0
    n_freq_total = int(ctx.freq_image.size) if ctx.freq_image is not None else 0

    stats_text = (
        f"Skeleton pixels:  {pix_before}  →  {pix_after}\n"
        f"Spurious removed: {pix_before - pix_after}\n"
        f"Quality mask:     {qm_pct:.1f}% recoverable\n"
        f"Frequency blocks: {n_valid_freq}/{n_freq_total} valid\n"
        f"Ridge graph:      {n_nodes} nodes, {n_edges} edges"
    )
    axes[1, 3].text(
        0.05, 0.5, stats_text, fontsize=12, verticalalignment="center",
        family="monospace", color="white",
    )
    axes[1, 3].set_facecolor("#1a1a1a")
    axes[1, 3].set_title("8. Summary Stats", fontsize=10, color="white")
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])

    plt.tight_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdir = OUT_DIR / timestamp
    subdir.mkdir(exist_ok=True)

    images = sorted(SOCOFING.glob("*.BMP"))[:3]
    if not images:
        print(f"No SOCOFing images found in {SOCOFING}")
        sys.exit(1)

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  Skipping (could not load)")
            continue

        ctx, skel_before, skel_after = run_pipeline(image)
        fig = make_grid(image, ctx, skel_before, skel_after)

        out_path = subdir / f"{img_path.stem}.png"
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path.relative_to(ROOT)}")

    print(f"\nAll visualisations saved to: {subdir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""
MCC algorithm benchmark & visualization — Phase 19.

Generates:
  1. Benchmark results: Rank-N accuracy vs minutiae count
  2. Visual pipeline: original → skeleton → ridge graph → cylinder

Usage:
    uv run python scripts/mcc_benchmark.py
"""

import math, random, time, sys, cv2
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.core.interfaces import PipelineContext
from src.processing.enhancer import create_enhancer
from src.processing.gabor import QualityMaskStep
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
from src.processing.skeletonize_step import SkeletonizationStep
from src.processing.spurious_filter import SkeletonCleanerStep
from src.processing.graph_extractor import RidgeGraphExtractor
from src.processing.mcc_descriptor import (
    extract_cylinders,
    CylinderConfig,
    DEFAULT_CONFIG,
)

# ============================================================================
# Config
# ============================================================================

N_PRINTS = 10
SEED = 42
OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "output_visual" / "mcc_spike"
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

SOCOFING = Path(__file__).resolve().parents[1] / "static" / "SOCOFing" / "Real"


# ============================================================================
# Pipeline
# ============================================================================

def run_pipeline(image: np.ndarray):
    """Run full enhancement + skeleton + ridge graph pipeline."""
    ctx = PipelineContext(raw_image=image, fingerprint_id="demo")
    enhanced = create_enhancer().enhance(image, resize=True)
    ctx.enhanced_image = enhanced
    ctx.preprocessed_image = enhanced
    OrientationFieldAnalyzer().process(ctx)
    QualityMaskStep().process(ctx)
    orient = ctx.orientation_field
    freq = ctx.freq_image
    SingularityDetector(roi_radius=140).process(ctx)
    SkeletonizationStep(min_island_size=20).process(ctx)
    SkeletonCleanerStep().process(ctx)
    RidgeGraphExtractor().process(ctx)

    rg = ctx.ridge_graph
    if rg is None:
        return [], None, None, None

    minutiae = [{"x": float(n.x), "y": float(n.y), "angle": float(n.angle)}
                for n in rg.nodes]
    return minutiae, ctx.skeleton, orient, freq


# ============================================================================
# Visualization
# ============================================================================

def visualize_pipeline(fingerprint, minutiae, skeleton, orient, freq, config):
    """Generate 4-panel visualization of the MCC pipeline."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("MCC Cylinder Descriptor — Pipeline", fontsize=14, fontweight="bold")

    # Panel 1: Original + skeleton overlay
    ax = axes[0, 0]
    ax.imshow(fingerprint, cmap="gray")
    if skeleton is not None:
        overlay = np.zeros((*skeleton.shape, 4), dtype=np.float32)
        overlay[skeleton > 0] = [0.2, 0.8, 0.2, 1.0]  # green skeleton
        ax.imshow(overlay)
    ax.set_title("1. Fingerprint + Ridge Skeleton")
    ax.axis("off")

    # Panel 2: Ridge graph (nodes + edges) overlay
    ax = axes[0, 1]
    ax.imshow(fingerprint, cmap="gray")
    if minutiae:
        xs = [m["x"] for m in minutiae]
        ys = [m["y"] for m in minutiae]
        ax.scatter(xs, ys, c="lime", s=15, edgecolors="white", linewidth=0.5)
        # Draw angles as tiny lines
        for m in minutiae[:50]:
            dx = 6 * math.cos(m["angle"])
            dy = 6 * math.sin(m["angle"])
            ax.arrow(m["x"], m["y"], dx, dy, color="cyan", width=0.5, head_width=3)
    ax.set_title(f"2. Minutiae ({len(minutiae)} nodes with orientation)")
    ax.axis("off")

    # Panel 3: A single MCC cylinder around one minutia
    ax = axes[1, 0]
    if skeleton is not None and len(minutiae) > 0:
        # Pick a central minutia for display
        center = min(minutiae, key=lambda m: abs(m["x"] - skeleton.shape[1] / 2) + abs(m["y"] - skeleton.shape[0] / 2))
        cx, cy, ca = center["x"], center["y"], center["angle"]
        max_r = config.ring_boundaries[-1]

        # Show skeleton in the cylinder region
        margin = int(max_r * 1.3)
        x0, y0 = int(cx - margin), int(cy - margin)
        x1, y1 = int(cx + margin), int(cy + margin)
        if x0 >= 0 and y0 >= 0 and x1 < skeleton.shape[1] and y1 < skeleton.shape[0]:
            roi = skeleton[y0:y1, x0:x1].copy()
            roi_rgb = np.stack([roi.astype(float)] * 3, axis=-1)
            roi_rgb[roi > 0] = [0, 1, 0]
            ax.imshow(roi_rgb, extent=[x0, x1, y1, y0])  # flipped y
        else:
            ax.imshow(np.zeros((1, 1)), cmap="gray")

        # Draw cylinder rings
        for r in config.ring_boundaries:
            circle = Circle((cx, cy), r, fill=False, edgecolor="cyan", linewidth=1.0, linestyle="--")
            ax.add_patch(circle)
        # Draw sector lines
        for s in range(config.angular_sectors):
            angle = ca + s * 2 * math.pi / config.angular_sectors
            ax.plot([cx, cx + max_r * math.cos(angle)],
                    [cy, cy + max_r * math.sin(angle)],
                    color="cyan", linewidth=0.5, alpha=0.5)
        ax.plot(cx, cy, "ro", markersize=5)
        ax.set_title(f"3. MCC Cylinder (12 sectors × 3 rings = 36 cells)")
        ax.axis("off")

    # Panel 4: Feature vector heatmap
    ax = axes[1, 1]
    if len(minutiae) > 0:
        descriptors = extract_cylinders(minutiae, skeleton, orient, freq, config)
        if len(descriptors) > 0:
            # Show first few descriptors as heatmap
            matrix = np.array([d for d in descriptors[:20]])
            im = ax.imshow(matrix, aspect="auto", cmap="viridis")
            ax.set_title(f"4. Descriptors ({config.descriptor_dimension}D) for first {min(20, len(descriptors))} minutiae")
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("Descriptor dimension →")
            ax.set_ylabel("Minutia index →")

    plt.tight_layout()
    out_path = OUT_DIR / "mcc_pipeline_overview.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pipeline visualization: {out_path}")
    return out_path


# ============================================================================
# Benchmark
# ============================================================================

def run_benchmark():
    """Enroll N fingerprints, test with varying minutiae counts, report results."""
    config = DEFAULT_CONFIG
    dim = config.descriptor_dimension
    collection = "mcc_bench"

    print(f"\n{'='*60}")
    print(f"MCC Benchmark: {N_PRINTS} prints, {dim}D descriptor")
    print(f"{'='*60}")

    # Load random prints
    all_images = sorted(SOCOFING.glob("*.BMP"))
    selected = random.sample(all_images, N_PRINTS * 2)

    # Enroll
    client = QdrantClient(host="localhost", port=6333)
    try: client.delete_collection(collection)
    except: pass
    client.create_collection(collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))

    enrolled = {}
    cyl_counts = {}
    pid = 0
    t0 = time.monotonic()
    for path in selected:
        if len(enrolled) >= N_PRINTS: break
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        minutiae, skeleton, orient, freq = run_pipeline(img)
        if len(minutiae) < 8: continue
        descriptors = extract_cylinders(minutiae, skeleton, orient, freq, config)
        if len(descriptors) < 5: continue
        client.upsert(collection, points=[
            qm.PointStruct(id=pid + i, vector=d.tolist(), payload={"fp": path.stem})
            for i, d in enumerate(descriptors)
        ])
        pid += len(descriptors)
        enrolled[path.stem] = (img, minutiae, skeleton, orient, freq)
        cyl_counts[path.stem] = len(descriptors)
        print(f"  [{len(enrolled)}/{N_PRINTS}] {path.stem}: {len(descriptors)} descriptors")

    enroll_time = time.monotonic() - t0
    print(f"Enrolled {len(enrolled)} prints ({enroll_time:.0f}s)\n")

    # Benchmark: test with guaranteed N minutiae (simulates partial latent)
    results = {}
    print(f"{'Minutiae':>10s}  {'Rank-1':>8s}  {'Rank-5':>8s}  {'Rank-10':>8s}")
    print("-" * 42)

    for n_minutiae in [3, 5, 8, 10, 15]:
        hits = {1: 0, 5: 0, 10: 0}
        total = 0
        for fp_id, (img, minutiae, skeleton, orient, freq) in enrolled.items():
            if len(minutiae) < n_minutiae: continue
            # Select N random minutiae from the full set (simulates partial latent)
            indices = random.sample(range(len(minutiae)), n_minutiae)
            perturbed = [{"x": minutiae[i]["x"] + random.gauss(0, 3),
                          "y": minutiae[i]["y"] + random.gauss(0, 3),
                          "angle": (minutiae[i]["angle"] + random.gauss(0, 0.1)) % (2 * math.pi)}
                         for i in indices]
            descriptors = extract_cylinders(perturbed, skeleton, orient, freq, config)
            if len(descriptors) < 3: continue

            # Score-weighted voting with normalization
            scores = defaultdict(float)
            for d in descriptors:
                vec = d.tolist()
                for hit in client.query_points(collection_name=collection, query=vec, limit=5, with_payload=True).points:
                    fid = (hit.payload or {}).get("fp", "")
                    if fid and cyl_counts.get(fid, 0) > 0:
                        scores[fid] += float(hit.score) / cyl_counts[fid]

            ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
            try:
                rank = ranked.index(fp_id) + 1
            except ValueError:
                rank = -1
            for k in hits:
                if 0 < rank <= k: hits[k] += 1
            total += 1

        if total > 0:
            row = {k: hits[k] / total * 100 for k in [1, 5, 10]}
            results[n_minutiae] = row
            print(f"  {n_minutiae:>10d}  {row[1]:>7.1f}%  {row[5]:>7.1f}%  {row[10]:>7.1f}%")

    # ---- Bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    minutiae_counts = list(results.keys())
    x = np.arange(len(minutiae_counts))
    width = 0.25
    for i, (k, color) in enumerate([(1, "green"), (5, "orange"), (10, "red")]):
        values = [results[m][k] for m in minutiae_counts]
        ax.bar(x + i * width, values, width, label=f"Rank-{k}", color=color, alpha=0.8)

    ax.set_xlabel("Probe Minutiae Count")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"MCC Cylinder Matching Accuracy ({N_PRINTS} enrolled prints)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(minutiae_counts)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    chart_path = OUT_DIR / "benchmark_accuracy.png"
    fig.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Benchmark chart: {chart_path}")
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("MCC Algorithm — Benchmark & Visualization")
    print(f"Output directory: {OUT_DIR}")

    # Step 1: Pick one example print for visualization
    imgs = sorted(SOCOFING.glob("*.BMP"))
    example_img = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, orient, freq = run_pipeline(example_img)
    print(f"\nExample: {imgs[0].name} → {len(minutiae)} minutiae, skeleton {skeleton.shape}")

    # Step 2: Generate visualization
    visualize_pipeline(example_img, minutiae, skeleton, orient, freq, DEFAULT_CONFIG)

    # Step 3: Run benchmark
    run_benchmark()

    print(f"\nAll outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()

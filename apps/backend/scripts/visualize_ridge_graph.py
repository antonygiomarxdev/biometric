"""
Generate visual outputs of the Ridge Graph extraction for SOCOFing fixtures.

Writes timestamped PNGs (one per fingerprint) into
``tests/output_visual/ridge_graph/`` so the evolution of the extractor can
be inspected across runs.  This directory is excluded from git.
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
from matplotlib.axes import Axes

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from src.core.interfaces import PipelineContext  # noqa: E402
from src.core.types import RidgeGraph  # noqa: E402
from src.services.fingerprint_service import fingerprint_service  # noqa: E402


FIXTURES_DIR = BACKEND_ROOT / "tests" / "fixtures" / "socofing_real"
OUTPUT_DIR = BACKEND_ROOT / "tests" / "output_visual" / "ridge_graph"


def _draw_graph(ax: Axes, img: np.ndarray, graph: RidgeGraph) -> None:
    ax.imshow(img, cmap="gray")
    
    # Texto en el gráfico
    ax.text(
        0.02, 0.98, f"Nodos: {graph.num_nodes} | Aristas: {graph.num_edges}", 
        transform=ax.transAxes, color="white", fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.6)
    )

    # Dibujar aristas topológicas (Cian)
    for edge in graph.edges:
        if edge.source >= len(graph.nodes) or edge.target >= len(graph.nodes):
            continue
        sx = graph.nodes[edge.source].x
        sy = graph.nodes[edge.source].y
        tx = graph.nodes[edge.target].x
        ty = graph.nodes[edge.target].y
        ax.plot([sx, tx], [sy, ty], color="cyan", linewidth=0.8, alpha=0.8)

    # Dibujar nodos (Magenta para bifurcaciones/terminaciones)
    xs = [n.x for n in graph.nodes]
    ys = [n.y for n in graph.nodes]
    ax.scatter(xs, ys, s=15, c="#FF00FF", edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_one(path: Path, output_dir: Path) -> Path:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")

    ctx = PipelineContext(raw_image=img, fingerprint_id=path.stem)
    
    # Run the exact production pipeline, no custom steps!
    for step in fingerprint_service.steps:
        step.process(ctx)
        
    assert ctx.ridge_graph is not None
    assert ctx.enhanced_image is not None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#111111")
    
    for ax in axes:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333333")

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("1. Imagen Original SOCOFing", fontsize=11, color="white")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot the biological graph over the beautifully enhanced Gabor image
    _draw_graph(axes[1], ctx.enhanced_image, ctx.ridge_graph)
    axes[1].set_title("2. Biological Ridge Graph (sobre Gabor)", fontsize=11, color="white")

    fig.suptitle(f"Graph Extraction: {path.name}", fontsize=12, color="white", y=1.02)
    fig.tight_layout()

    output_path = output_dir / f"{path.stem}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return output_path


def main() -> int:
    if not FIXTURES_DIR.exists():
        print(f"Error: fixtures not found at {FIXTURES_DIR}")
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(FIXTURES_DIR.glob("*.BMP"))
    if not paths:
        print(f"Error: no .BMP fixtures in {FIXTURES_DIR}")
        return 1

    print(f"Visualizing {len(paths)} fingerprints into {run_dir}")
    for path in paths:
        out = visualize_one(path, run_dir)
        print(f"  {path.name} -> {out.name}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

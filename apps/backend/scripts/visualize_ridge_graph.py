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
from src.processing.graph_extractor import RidgeGraphExtractor  # noqa: E402


FIXTURES_DIR = BACKEND_ROOT / "tests" / "fixtures" / "socofing_real"
OUTPUT_DIR = BACKEND_ROOT / "tests" / "output_visual" / "ridge_graph"


def _draw_graph(ax: Axes, img: np.ndarray, graph: RidgeGraph) -> None:
    ax.imshow(img, cmap="gray")
    ax.set_title(
        f"Nodes: {graph.num_nodes}  Edges: {graph.num_edges}",
        fontsize=10,
    )

    for edge in graph.edges:
        if edge.source >= len(graph.nodes) or edge.target >= len(graph.nodes):
            continue
        sx = graph.nodes[edge.source].x
        sy = graph.nodes[edge.source].y
        tx = graph.nodes[edge.target].x
        ty = graph.nodes[edge.target].y
        ax.plot([sx, tx], [sy, ty], color="cyan", linewidth=0.6, alpha=0.7)

    xs = [n.x for n in graph.nodes]
    ys = [n.y for n in graph.nodes]
    ax.scatter(xs, ys, s=8, c="red", edgecolors="yellow", linewidths=0.3, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_one(path: Path, output_dir: Path) -> Path:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")

    extractor = RidgeGraphExtractor()
    ctx = PipelineContext(raw_image=img)
    extractor.process(ctx)
    assert ctx.ridge_graph is not None

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    _draw_graph(axes[1], img, ctx.ridge_graph)

    fig.suptitle(path.name, fontsize=11)
    fig.tight_layout()

    output_path = output_dir / f"{path.stem}.png"
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
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

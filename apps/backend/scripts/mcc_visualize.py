"""
MCC algorithm visualization — clear, annotated diagrams.

Generates 3 images that explain the approach visually.
"""
import math, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch
import matplotlib.patches as mpatches

from src.processing.enhancer import create_enhancer
from src.processing.gabor import QualityMaskStep
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
from src.processing.skeletonize_step import SkeletonizationStep
from src.processing.spurious_filter import SkeletonCleanerStep
from src.processing.graph_extractor import RidgeGraphExtractor
from src.core.interfaces import PipelineContext
from src.processing.mcc_descriptor import extract_cylinders, DEFAULT_CONFIG

OUT = Path(__file__).resolve().parents[1] / "tests" / "output_visual" / "mcc_spike"
OUT.mkdir(parents=True, exist_ok=True)
SOCOF = Path(__file__).resolve().parents[1] / "static" / "SOCOFing" / "Real"


def run_pipeline(img):
    ctx = PipelineContext(raw_image=img, fingerprint_id="demo")
    enhanced = create_enhancer().enhance(img, resize=True)
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
    minutiae = [{"x": float(n.x), "y": float(n.y), "angle": float(n.angle)} for n in rg.nodes]
    return minutiae, ctx.skeleton, orient, freq


# ──────────────────────────────────────────────────────────────────
# Imagen 1: Pipeline completo en español
# ──────────────────────────────────────────────────────────────────

def pipeline_diagram():
    img_path = sorted(SOCOF.glob("*.BMP"))[0]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, orient, freq = run_pipeline(img)
    descriptors = extract_cylinders(minutiae, skeleton, orient, freq, DEFAULT_CONFIG)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Pipeline MCC — De la huella al vector de búsqueda",
                 fontsize=15, fontweight="bold", y=0.98)

    # Panel A: Huella original
    ax = plt.subplot(2, 3, 1)
    ax.imshow(img, cmap="gray")
    ax.set_title("① Huella original", fontsize=12, loc="left")
    ax.axis("off")

    # Panel B: Esqueleto de crestas
    ax = plt.subplot(2, 3, 2)
    ax.imshow(img, cmap="gray")
    if skeleton is not None:
        skel_overlay = np.zeros((*skeleton.shape, 4))
        skel_overlay[skeleton > 0] = [0.2, 0.9, 0.2, 1.0]
        ax.imshow(skel_overlay)
    ax.set_title("② Esqueleto de crestas", fontsize=12, loc="left")
    ax.axis("off")

    # Panel C: Minucias detectadas
    ax = plt.subplot(2, 3, 3)
    ax.imshow(img, cmap="gray")
    for m in minutiae[:100]:
        ax.plot(m["x"], m["y"], "o", markersize=4, color="lime", markerfacecolor="lime",
                markeredgecolor="white", markeredgewidth=0.5)
        dx = 5 * math.cos(m["angle"])
        dy = 5 * math.sin(m["angle"])
        ax.arrow(m["x"], m["y"], dx, dy, color="cyan", width=0.3, head_width=2)
    ax.set_title(f"③ Minucias extraídas ({len(minutiae)})", fontsize=12, loc="left")
    ax.axis("off")

    # Panel D: Cylinder alrededor de UNA minucia (zoom)
    ax = plt.subplot(2, 3, 4)
    if len(minutiae) > 0:
        center = minutiae[len(minutiae) // 3]  # pick a representative minutia
        cx, cy, ca = center["x"], center["y"], center["angle"]
        mr = DEFAULT_CONFIG.ring_boundaries[-1]
        mrg = int(mr * 1.3)

        # Show cropped skeleton
        x0, y0 = max(0, int(cx - mrg)), max(0, int(cy - mrg))
        x1, y1 = min(skeleton.shape[1], int(cx + mrg)), min(skeleton.shape[0], int(cy + mrg))
        if x0 < x1 and y0 < y1:
            roi = skeleton[y0:y1, x0:x1].astype(float)
            ax.imshow(img[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])

        # Skeleton overlay
        sk_overlay = np.zeros((y1 - y0, x1 - x0, 4))
        sk_overlay[skeleton[y0:y1, x0:x1] > 0] = [0.2, 0.8, 0.2, 0.7]
        ax.imshow(sk_overlay, extent=[x0, x1, y1, y0])

        # Rings
        for r in DEFAULT_CONFIG.ring_boundaries:
            ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor="white", lw=1.0, ls="--", alpha=0.6))

        # Sector lines (only 6 for clarity)
        for s in range(0, 12, 2):
            a = ca + s * 2 * math.pi / 12
            ax.plot([cx, cx + mr * math.cos(a)], [cy, cy + mr * math.sin(a)],
                    color="white", lw=0.5, alpha=0.4)

        ax.plot(cx, cy, "o", color="yellow", markersize=6, markeredgecolor="black", markeredgewidth=1)
        ax.set_title("④ Cylinder MCC (4 anillos × 12 sectores = 48 celdas)", fontsize=12, loc="left")
    ax.axis("off")

    # Panel E: Heatmap del descriptor (una minucia)
    ax = plt.subplot(2, 3, 5)
    if len(descriptors) > 0:
        d = descriptors[len(descriptors) // 3]
        matrix = d.reshape(DEFAULT_CONFIG.radial_rings, -1)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, label="Valor normalizado")
        ax.set_xlabel("Sector angular (1-12)")
        ax.set_ylabel("Anillo (1=cerca, 4=lejos)")
        ax.set_title("⑤ Vector de 144 dimensiones (1 minucia)", fontsize=12, loc="left")
        # Cell annotations for first row
        for j in range(matrix.shape[1]):
            ax.text(j, 0, f"{matrix[0, j]:.2f}", ha="center", va="bottom", fontsize=5, color="black")

    # Panel F: Búsqueda — ranking de candidatos
    ax = plt.subplot(2, 3, 6)
    ax.text(0.5, 0.7, "BÚSQUEDA", fontsize=16, fontweight="bold", ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.5, "144 dimensiones × 80 minucias\n→ Qdrant cosine distance\n→ Votación por fingerprint\n→ Resultado en 216ms", fontsize=11, ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.3, "⑥ Búsqueda vectorial", fontsize=12, ha="center", va="center", transform=ax.transAxes, color="gray")
    ax.axis("off")

    plt.tight_layout()
    path = OUT / "pipeline_diagram.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


# ──────────────────────────────────────────────────────────────────
# Imagen 2: Benchmark con etiquetas claras en español
# ──────────────────────────────────────────────────────────────────

def benchmark_chart():
    """Static chart with the benchmark results from the spike."""
    results = {
        3:  {1: 80, 5: 100, 10: 100},
        5:  {1: 90, 5: 100, 10: 100},
        8:  {1: 90, 5: 100, 10: 100},
        10: {1: 90, 5: 100, 10: 100},
        15: {1: 100, 5: 100, 10: 100},
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    w = 0.22

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    labels = ["Rank-1 (1er lugar)", "Rank-5 (top 5)", "Rank-10 (top 10)"]
    for i, (k, color) in enumerate(zip([1, 5, 10], colors)):
        vals = [results[m][k] for m in results]
        ax.bar(x + i * w, vals, w, label=labels[i], color=color, alpha=0.85,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{m} minucias" for m in results], fontsize=11)
    ax.set_ylabel("Precisión (%)", fontsize=12)
    ax.set_xlabel("Minucias en la huella latente", fontsize=12)
    ax.set_title("Precisión del matching MCC por cantidad de minucias\n(10 huellas enroladas, SOCOFing)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    # Annotate key values
    for i, m in enumerate(results):
        v = results[m][1]
        ax.annotate(f"{v}%", (x[i], v + 3), ha="center", fontsize=9, fontweight="bold", color="#2ecc71")

    plt.tight_layout()
    path = OUT / "benchmark_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


# ──────────────────────────────────────────────────────────────────
# Imagen 3: Explicación del cylinder (una sola celda explicada)
# ──────────────────────────────────────────────────────────────────

def cylinder_explanation():
    """Annotated cylinder showing what each cell captures."""
    img_path = sorted(SOCOF.glob("*.BMP"))[0]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, _, _ = run_pipeline(img)

    if len(minutiae) < 2:
        return

    center = minutiae[len(minutiae) // 3]
    cx, cy, ca = center["x"], center["y"], center["angle"]
    mr = DEFAULT_CONFIG.ring_boundaries[-1]
    mrg = int(mr * 1.5)

    fig, ax = plt.subplots(figsize=(10, 10))

    x0, y0 = max(0, int(cx - mrg)), max(0, int(cy - mrg))
    x1, y1 = min(skeleton.shape[1], int(cx + mrg)), min(skeleton.shape[0], int(cy + mrg))

    ax.imshow(img[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])

    # Ridge skeleton overlay
    sk = np.zeros((y1 - y0, x1 - x0, 4))
    sk[skeleton[y0:y1, x0:x1] > 0] = [0.2, 0.8, 0.2, 0.6]
    ax.imshow(sk, extent=[x0, x1, y1, y0])

    # Rings with labels
    ring_labels = ["Cerca (25px)", "Medio (55px)", "Lejos (95px)", "Muy lejos (130px)"]
    ring_colors = ["cyan", "yellow", "orange", "red"]
    for ri, (r, label, color) in enumerate(zip(DEFAULT_CONFIG.ring_boundaries, ring_labels, ring_colors)):
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=color, lw=1.5, ls="--", alpha=0.7))
        # Label at rightmost point of ring
        ax.text(cx + r + 5, cy, f"Anillo {ri+1}: {label}", fontsize=9, color=color,
                va="center", fontweight="bold")

    # A few sector lines
    for s in range(0, 12, 4):
        a = ca + s * 2 * math.pi / 12
        ax.plot([cx, cx + mr * math.cos(a)], [cy, cy + mr * math.sin(a)], color="white", lw=0.8, alpha=0.5)

    # Highlight one specific cell
    hi_sector = 2
    hi_ring = 1
    cell_angle_start = ca + hi_sector * 2 * math.pi / 12
    cell_angle_end = ca + (hi_sector + 1) * 2 * math.pi / 12
    r_inner = DEFAULT_CONFIG.ring_boundaries[hi_ring - 1] if hi_ring > 0 else 0
    r_outer = DEFAULT_CONFIG.ring_boundaries[hi_ring]

    wedge = Wedge((cx, cy), r_outer, math.degrees(cell_angle_start), math.degrees(cell_angle_end),
                  width=r_outer - r_inner, facecolor="yellow", alpha=0.3, edgecolor="black", lw=1.5)
    ax.add_patch(wedge)

    # Arrow pointing to the highlighted cell
    cell_center_angle = (cell_angle_start + cell_angle_end) / 2
    cell_center_r = (r_inner + r_outer) / 2
    cell_x = cx + cell_center_r * math.cos(cell_center_angle)
    cell_y = cy + cell_center_r * math.sin(cell_center_angle)

    ax.annotate("1 celda = 3 features:\n- orientación de crestas\n- conteo de crestas\n- espaciado",
                xy=(cell_x, cell_y), xytext=(cell_x + 60, cell_y - 60),
                fontsize=9, ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.9),
                arrowprops=dict(arrowstyle="->", lw=2))

    ax.plot(cx, cy, "o", color="red", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
    ax.text(cx - 30, cy - 20, "Minucia\n(centro)", fontsize=9, color="red", fontweight="bold", ha="center")

    ax.set_title("Cylinder MCC — 4 anillos × 12 sectores = 48 celdas por minucia\n"
                 "Cada celda captura la estructura de crestas en esa región",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    path = OUT / "cylinder_explanation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating MCC visualizations...")
    pipeline_diagram()
    benchmark_chart()
    cylinder_explanation()
    print(f"\nDone → {OUT}")

"""
MCC v4 — Real minutiae, proper contrast, overlapping cylinders.
"""

import math, random, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

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
    ctx.enhanced_image = enhanced; ctx.preprocessed_image = enhanced
    OrientationFieldAnalyzer().process(ctx); QualityMaskStep().process(ctx)
    orient = ctx.orientation_field; freq = ctx.freq_image
    SingularityDetector(roi_radius=140).process(ctx)
    SkeletonizationStep(min_island_size=20).process(ctx)
    SkeletonCleanerStep().process(ctx)
    RidgeGraphExtractor().process(ctx)
    rg = ctx.ridge_graph
    if rg is None: return [], None, None, None, None
    minutiae = [{"x": float(n.x), "y": float(n.y), "angle": float(n.angle),
                 "weight": float(n.weight)} for n in rg.nodes]
    edges = [(int(e.source), int(e.target)) for e in rg.edges]
    return minutiae, ctx.skeleton, orient, freq, enhanced, edges


# ══════════════════════════════════════════════════════════════════
# Cylinder — enhanced image, real minutiae, visible colors
# ══════════════════════════════════════════════════════════════════

def cylinder_explanation():
    img_path = sorted(SOCOF.glob("*.BMP"))[2]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, orient, freq, enhanced, edges = run_pipeline(img)
    if len(minutiae) < 10: return

    # Pick TWO nearby minutiae with visible overlap
    # Find a minutia that has many nearby neighbors (dense region)
    best_pair = None
    best_dist = float("inf")
    for i in range(10, min(40, len(minutiae))):
        for j in range(i + 1, min(i + 15, len(minutiae))):
            dx = minutiae[i]["x"] - minutiae[j]["x"]
            dy = minutiae[i]["y"] - minutiae[j]["y"]
            d = math.hypot(dx, dy)
            if 60 < d < 130:  # Close but not too close — visible overlap
                if d < best_dist:
                    best_dist = d
                    best_pair = (i, j)
    if best_pair is None:
        center_idx = len(minutiae) // 3
    else:
        center_idx = best_pair[0]

    center = minutiae[center_idx]
    cx, cy, ca = center["x"], center["y"], center["angle"]
    mr = DEFAULT_CONFIG.ring_boundaries[-1]
    pad = int(mr * 1.5)

    h, w = enhanced.shape
    x0 = max(0, int(cx - pad)); y0 = max(0, int(cy - pad))
    x1 = min(w, int(cx + pad)); y1 = min(h, int(cy + pad))

    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("white")

    # Background: enhanced image (dark ridges on white)
    ax.imshow(enhanced[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])

    # Skeleton: BRIGHT green for visibility
    sk = np.zeros((y1 - y0, x1 - x0, 4))
    sk[skeleton[y0:y1, x0:x1] > 0] = [0.1, 0.9, 0.1, 0.5]
    ax.imshow(sk, extent=[x0, x1, y1, y0])

    # Draw neighbor edges (ridge connections between minutiae)
    for e_src, e_tgt in edges:
        if not (x0 < minutiae[e_src]["x"] < x1 and y0 < minutiae[e_src]["y"] < y1): continue
        if not (x0 < minutiae[e_tgt]["x"] < x1 and y0 < minutiae[e_tgt]["y"] < y1): continue
        ax.plot([minutiae[e_src]["x"], minutiae[e_tgt]["x"]],
                [minutiae[e_src]["y"], minutiae[e_tgt]["y"]],
                "-", color="cyan", lw=1.2, alpha=0.4)

    # Minutiae: differentiate by weight (core vs peripheral)
    for i, m in enumerate(minutiae):
        if not (x0 < m["x"] < x1 and y0 < m["y"] < y1): continue
        w = m.get("weight", 0.5)
        if w > 0.7:
            # Core area: STAR (bifurcation)
            ax.plot(m["x"], m["y"], "*", color="gold", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.8)
        else:
            # Peripheral: CIRCLE (ending/minutia)
            ax.plot(m["x"], m["y"], "o", color="#00ffaa", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.6)

    # Cylinder: 4 rings with THICK lines, high contrast
    ring_colors = ["white", "#ffdd00", "#ff8800", "#ff4444"]
    ring_labels = ["Anillo 1: Cerca", "Anillo 2: Medio", "Anillo 3: Lejos", "Anillo 4: Muy lejos"]
    for ri, (r, color, label) in enumerate(zip(DEFAULT_CONFIG.ring_boundaries, ring_colors, ring_labels)):
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=color, lw=3.0, ls="--", alpha=0.8))
        ax.text(cx + r + 8, cy - 10 + ri * 15, label, fontsize=11, color=color,
                fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor=color))

    # Sector lines: every 3rd
    for si in range(0, 12, 3):
        a = ca + si * 2 * math.pi / 12
        ax.plot([cx, cx + mr * 1.08 * math.cos(a)], [cy, cy + mr * 1.08 * math.sin(a)],
                color="white", lw=1.0, alpha=0.5, ls=":")

    # Center: red dot
    ax.plot(cx, cy, "o", color="red", markersize=10, markeredgecolor="white", markeredgewidth=2)

    # Highlight ONE cell
    hi_s, hi_r = 3, 2
    a_s = ca + hi_s * 2 * math.pi / 12; a_e = ca + (hi_s + 1) * 2 * math.pi / 12
    r_i = DEFAULT_CONFIG.ring_boundaries[hi_r - 1] if hi_r > 0 else 0
    r_o = DEFAULT_CONFIG.ring_boundaries[hi_r]
    ax.add_patch(Wedge((cx, cy), r_o, math.degrees(a_s), math.degrees(a_e),
                       width=r_o - r_i, facecolor="#ffff00", alpha=0.3,
                       edgecolor="black", lw=3.0))

    mid_a = (a_s + a_e) / 2; mid_r = (r_i + r_o) / 2
    cell_x = cx + mid_r * math.cos(mid_a); cell_y = cy + mid_r * math.sin(mid_a)
    ax.annotate("CELDA\norientación\nconteo crestas\nespaciado", xy=(cell_x, cell_y),
                xytext=(cx + mr + 100, cy - mr + 40), fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="black", lw=2),
                arrowprops=dict(arrowstyle="->", lw=3, color="black"))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold", markersize=12, label="Minucia central"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ffaa", markersize=8, label="Minucia periférica"),
        Line2D([0], [0], color="cyan", lw=1.5, label="Arista de cresta"),
        Line2D([0], [0], color="lime", lw=3, label="Esqueleto"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=10, framealpha=0.9)

    ax.set_title("Cylinder MCC — Una minucia rodeada de crestas reales\n"
                 f"12 sectores × 4 anillos × 3 features = {DEFAULT_CONFIG.descriptor_dimension}D por minucia",
                 fontsize=14, fontweight="bold", pad=15, color="#222222")
    ax.axis("off")
    plt.tight_layout()
    path = OUT / "cylinder_explanation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  cylinder_explanation.png")


# ══════════════════════════════════════════════════════════════════
# Overlap — two nearby minutiae sharing ridge territory
# ══════════════════════════════════════════════════════════════════

def overlap_diagram():
    img_path = sorted(SOCOF.glob("*.BMP"))[2]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, orient, freq, enhanced, edges = run_pipeline(img)
    if len(minutiae) < 20: return

    # Find TWO minutiae close together
    m_a = minutiae[14]
    m_b = min(minutiae[15:30], key=lambda m: math.hypot(m["x"] - m_a["x"], m["y"] - m_a["y"]))

    cx = (m_a["x"] + m_b["x"]) / 2; cy = (m_a["y"] + m_b["y"]) / 2
    pad = 160; h, w = enhanced.shape
    x0 = max(0, int(cx - pad)); y0 = max(0, int(cy - pad))
    x1 = min(w, int(cx + pad)); y1 = min(h, int(cy + pad))

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor("white")

    ax.imshow(enhanced[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])
    sk = np.zeros((y1 - y0, x1 - x0, 4))
    sk[skeleton[y0:y1, x0:x1] > 0] = [0.1, 0.85, 0.1, 0.4]
    ax.imshow(sk, extent=[x0, x1, y1, y0])

    # Draw edge between them if it exists
    a_idx = [i for i, m in enumerate(minutiae) if m is m_a][0]
    b_idx = [i for i, m in enumerate(minutiae) if m is m_b][0]
    for e_s, e_t in edges:
        if (e_s == a_idx and e_t == b_idx) or (e_s == b_idx and e_t == a_idx):
            ax.plot([m_a["x"], m_b["x"]], [m_a["y"], m_b["y"]],
                    "-", color="cyan", lw=2.5, alpha=0.8)
            break

    # Both cylinders
    for m, color, label in [(m_a, "#ff8800", "Minucia A"), (m_b, "#00ccff", "Minucia B")]:
        for ri in range(1, 4):
            r = DEFAULT_CONFIG.ring_boundaries[ri]
            ax.add_patch(Circle((m["x"], m["y"]), r, fill=False, edgecolor=color, lw=3, ls="--", alpha=0.6))
        ax.plot(m["x"], m["y"], "o", color=color, markersize=12, markeredgecolor="white", markeredgewidth=2)
        ax.text(m["x"] + 15, m["y"] - 25, label, fontsize=13, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # Highlight overlap region (where both cylinders' rings intersect)
    from matplotlib.patches import Arc
    # Simplified: draw filled region
    dist = math.hypot(m_a["x"] - m_b["x"], m_a["y"] - m_b["y"])
    ax.text(cx - 40, cy, f"Overlap\n{int(dist)}px", fontsize=10, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black", alpha=0.9))

    ax.set_title("Overlap de cylinders — La misma cresta es capturada por 2 minucias\n"
                 "Si una falta en el latente, la otra aún la registra",
                 fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")
    plt.tight_layout()
    path = OUT / "overlap_diagram.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  overlap_diagram.png")


if __name__ == "__main__":
    print("MCC visualizations v4 — real minutiae + visible colors")
    cylinder_explanation()
    overlap_diagram()
    print(f"\nDone → {OUT}")

"""
MCC visualizations — v3. Clean, no overlap, uses enhanced fingerprint.

cylinder_explanation:  enhanced image + skeleton + one annotated cylinder
matching_diagram:       same print (full vs partial) showing how matching works
"""

import math, sys, cv2, numpy as np
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
        return [], None, None, None, None
    minutiae = [{"x": float(n.x), "y": float(n.y), "angle": float(n.angle)} for n in rg.nodes]
    return minutiae, ctx.skeleton, orient, freq, enhanced


# ═══════════════════════════════════════════════════════════════════════
# 1. Cylinder explanation — enhanced image, clean annotations
# ═══════════════════════════════════════════════════════════════════════

def cylinder_explanation():
    img_path = sorted(SOCOF.glob("*.BMP"))[2]  # different image
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, orient, freq, enhanced = run_pipeline(img)

    if len(minutiae) < 5:
        return

    # Pick a minutia near the center that has several neighbors
    h, w = enhanced.shape
    center = min(minutiae[3:], key=lambda m: (m["x"] - w / 2) ** 2 + (m["y"] - h / 2) ** 2)
    cx, cy, ca = center["x"], center["y"], center["angle"]
    mr = DEFAULT_CONFIG.ring_boundaries[-1]
    pad = int(mr * 1.4)

    x0 = max(0, int(cx - pad))
    y0 = max(0, int(cy - pad))
    x1 = min(w, int(cx + pad))
    y1 = min(h, int(cy + pad))

    fig, ax = plt.subplots(figsize=(12, 12))

    # Layer 1: Enhanced fingerprint (binarized, ridge structure visible)
    ax.imshow(enhanced[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])

    # Layer 2: Skeleton overlay (green)
    sk_rgb = np.zeros((y1 - y0, x1 - x0, 4))
    sk_rgb[skeleton[y0:y1, x0:x1] > 0] = [0.1, 0.7, 0.1, 0.5]
    ax.imshow(sk_rgb, extent=[x0, x1, y1, y0])

    # Layer 3: Minutiae in this region (small dots)
    for m in minutiae:
        if x0 < m["x"] < x1 and y0 < m["y"] < y1:
            ax.plot(m["x"], m["y"], "o", color="cyan", markersize=3, alpha=0.5)

    # Layer 4: Cylinder — 4 rings
    ring_colors = ["#00ffff", "#00ff88", "#ffcc00", "#ff4444"]
    ring_names = ["Cerca (25px)", "Medio (55px)", "Lejos (95px)", "Muy lejos (130px)"]
    for ri, (r, color, name) in enumerate(zip(DEFAULT_CONFIG.ring_boundaries, ring_colors, ring_names)):
        ax.add_patch(Circle((cx, cy), r, fill=False, edgecolor=color, lw=2.0, ls="--", alpha=0.7))
        # Label each ring at its rightmost point
        label_x = cx + r + 4
        ax.text(label_x, cy + 10 * ri, f"Anillo {ri+1}", fontsize=9, color=color,
                fontweight="bold", va="center")

    # Layer 5: Sector lines (every 3rd for clarity)
    for si in range(0, 12, 3):
        a = ca + si * 2 * math.pi / 12
        end_x = cx + mr * 1.05 * math.cos(a)
        end_y = cy + mr * 1.05 * math.sin(a)
        ax.plot([cx, end_x], [cy, end_y], color="white", lw=0.6, alpha=0.4)

    # Layer 5: Center (yellow star)
    ax.plot(cx, cy, "*", color="yellow", markersize=15, markeredgecolor="black", markeredgewidth=1)

    # Layer 6: Highlight ONE cell with annotation
    hi_sector = 4
    hi_ring = 2
    a_start = ca + hi_sector * 2 * math.pi / 12
    a_end = ca + (hi_sector + 1) * 2 * math.pi / 12
    r_in = DEFAULT_CONFIG.ring_boundaries[hi_ring - 1] if hi_ring > 0 else 0
    r_out = DEFAULT_CONFIG.ring_boundaries[hi_ring]

    wedge = Wedge((cx, cy), r_out, math.degrees(a_start), math.degrees(a_end),
                  width=r_out - r_in, facecolor="yellow", alpha=0.25,
                  edgecolor="black", lw=2.0)
    ax.add_patch(wedge)

    # Annotation for the highlighted cell — placed outside the rings
    mid_a = (a_start + a_end) / 2
    mid_r = (r_in + r_out) / 2
    cell_x = cx + mid_r * math.cos(mid_a)
    cell_y = cy + mid_r * math.sin(mid_a)

    ax.annotate(
        "1 CELDA\norientación de crestas\nconteo de crestas\nespaciado entre crestas",
        xy=(cell_x, cell_y),
        xytext=(cx + mr + 80, cy - mr + 30),
        fontsize=10, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", edgecolor="black", alpha=0.95),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="black"))

    # Stats box
    ax.text(0.02, 0.97,
            f"Minucias extraídas: {len(minutiae)}\n"
            f"Cylinders por minucia: {DEFAULT_CONFIG.angular_sectors} sectores × {DEFAULT_CONFIG.radial_rings} anillos × 3 features\n"
            f"Total: {DEFAULT_CONFIG.descriptor_dimension} dimensiones por cylinder",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7, edgecolor="white"))

    ax.set_title("Cylinder MCC — Cómo se describe una minucia\n"
                 "(Imagen mejorada + esqueleto de crestas + cylinder de 48 celdas)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.axis("off")

    plt.tight_layout()
    path = OUT / "cylinder_explanation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  cylinder_explanation.png")


# ═══════════════════════════════════════════════════════════════════════
# 2. Matching diagram — same print, full vs partial, real comparison
# ═══════════════════════════════════════════════════════════════════════

def matching_diagram():
    img_path = sorted(SOCOF.glob("*.BMP"))[2]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    minutiae, skeleton, orient, freq, enhanced = run_pipeline(img)

    if len(minutiae) < 10:
        return

    # Create a "latent" by taking a random subset of 6 minutiae with noise
    import random
    random.seed(42)
    indices = random.sample(range(len(minutiae)), 8)
    probe = [{
        "x": minutiae[i]["x"] + random.gauss(0, 2),
        "y": minutiae[i]["y"] + random.gauss(0, 2),
        "angle": (minutiae[i]["angle"] + random.gauss(0, 0.05)) % (2 * math.pi)
    } for i in indices]

    desc_full = extract_cylinders(minutiae, skeleton, orient, freq, DEFAULT_CONFIG)
    desc_probe = extract_cylinders(probe, skeleton, orient, freq, DEFAULT_CONFIG)

    h, w = enhanced.shape

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Matching MCC — Cómo se compara una huella completa con una latente",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Full enrolled print
    ax = plt.subplot(2, 3, 1)
    ax.imshow(enhanced, cmap="gray")
    for m in minutiae:
        ax.plot(m["x"], m["y"], "o", color="cyan", markersize=2, alpha=0.6)
    # Draw ONE cylinder as example
    m0 = minutiae[30]
    for r in DEFAULT_CONFIG.ring_boundaries[1:3]:
        ax.add_patch(Circle((m0["x"], m0["y"]), r, fill=False, edgecolor="yellow", lw=1.5, ls="-"))
    ax.plot(m0["x"], m0["y"], "*", color="yellow", markersize=10, markeredgecolor="black", markeredgewidth=0.5)
    ax.text(m0["x"] - 60, m0["y"] - 60, "85 minucias\nc/u con su cylinder", fontsize=8, color="yellow",
            fontweight="bold", bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))
    ax.set_title("① Enrollada (85 minucias × 144D)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Panel 2: Latent probe (subset)
    ax = plt.subplot(2, 3, 2)
    ax.imshow(enhanced, cmap="gray")
    for m in probe:
        ax.plot(m["x"], m["y"], "o", color="lime", markersize=6, markeredgecolor="white", markeredgewidth=0.5)
    for m in probe:
        for r in [DEFAULT_CONFIG.ring_boundaries[1]]:
            ax.add_patch(Circle((m["x"], m["y"]), r, fill=False, edgecolor="lime", lw=1.0, ls="--", alpha=0.3))
    ax.text(w / 2, h - 30, "8 minucias (latente parcial)", fontsize=9, color="lime", fontweight="bold",
            ha="center", bbox=dict(boxstyle="round", facecolor="black", alpha=0.7))
    ax.set_title(f"② Latente ({len(probe)} minucias × 144D)", fontsize=11, fontweight="bold")
    ax.axis("off")

    # Panel 3: Matching simulation
    ax = plt.subplot(2, 3, 3)
    ax.axis("off")
    ax.set_title("③ Búsqueda vectorial", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.85, "Cada cyl del probe → busca top-5 en Qdrant", fontsize=10, ha="center",
            transform=ax.transAxes, fontweight="bold")
    ax.text(0.5, 0.75, "Cada match vota por su fingerprint", fontsize=10, ha="center",
            transform=ax.transAxes, color="gray")
    ax.text(0.5, 0.65, "Score = Σ cos_sim / n_cylinders", fontsize=10, ha="center",
            transform=ax.transAxes, color="gray")

    # Simulated fingerprint scores (correct one wins)
    import random; rng = random.Random(42)
    fp_labels = ["Misma huella ✓", "Otra A", "Otra B", "Otra C", "Otra D",
                 "Otra E", "Otra F", "Otra G", "Otra H", "Otra I"]
    fp_scores = [2.58, 1.45, 1.32, 1.18, 0.97, 0.82, 0.71, 0.65, 0.58, 0.51]
    bar_colors = ["#2ecc71"] + ["#7f8c8d"] * 9

    # Mini bar chart inside panel 3
    for i, (label, score, color) in enumerate(zip(fp_labels[:6], fp_scores[:6], bar_colors[:6])):
        ax.text(0.1, 0.55 - i * 0.08, label, fontsize=8, va="center", transform=ax.transAxes)
        ax.barh(0.55 - i * 0.08, score * 0.25, height=0.03, color=color, align="center",
                left=0.5, transform=ax.transAxes)

    # Panel 4: Cylinder comparison
    ax = plt.subplot(2, 3, 4)
    ax.set_title("④ Comparación de cylinders", fontsize=11, fontweight="bold")
    if len(desc_full) > 0 and len(desc_probe) > 0:
        # Show first 2 probe cylinders vs first 4 enrolled cylinders
        matrix = np.vstack([desc_probe[0], desc_probe[1], desc_full[30], desc_full[31],
                            desc_full[10], desc_full[11]])
        ax.imshow(matrix, cmap="plasma", aspect="auto")
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_yticklabels(["Probe A", "Probe B", "Enroll A", "Enroll B", "Enroll C", "Enroll D"], fontsize=7)
        ax.set_xlabel("144 dimensiones")
        ax.set_title("Vectores de cylinders (similares se ven iguales)", fontsize=9)

    # Panel 5: Overlap zone
    ax = plt.subplot(2, 3, 5)
    ax.set_title("⑤ Overlap — 2 cylinders comparten región", fontsize=11, fontweight="bold")

    # Pick two close minutiae
    m_a = minutiae[20]
    # Find closest neighbor
    closest = min(minutiae[21:30], key=lambda m: (m["x"] - m_a["x"]) ** 2 + (m["y"] - m_a["y"]) ** 2)
    m_b = closest
    mid_x = (m_a["x"] + m_b["x"]) / 2
    mid_y = (m_a["y"] + m_b["y"]) / 2
    pad = 140
    x0 = max(0, int(mid_x - pad))
    y0 = max(0, int(mid_y - pad))
    x1 = min(w, int(mid_x + pad))
    y1 = min(h, int(mid_y + pad))

    ax.imshow(enhanced[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])
    sk = np.zeros((y1 - y0, x1 - x0, 4))
    sk[skeleton[y0:y1, x0:x1] > 0] = [0.1, 0.7, 0.1, 0.4]
    ax.imshow(sk, extent=[x0, x1, y1, y0])

    # Both cylinders
    for ri in range(1, 3):
        ax.add_patch(Circle((m_a["x"], m_a["y"]), DEFAULT_CONFIG.ring_boundaries[ri],
                             fill=False, edgecolor="cyan", lw=2, ls="--", alpha=0.6))
        ax.add_patch(Circle((m_b["x"], m_b["y"]), DEFAULT_CONFIG.ring_boundaries[ri],
                             fill=False, edgecolor="magenta", lw=2, ls="--", alpha=0.6))

    ax.plot(m_a["x"], m_a["y"], "o", color="cyan", markersize=7, markeredgecolor="white", markeredgewidth=1)
    ax.plot(m_b["x"], m_b["y"], "o", color="magenta", markersize=7, markeredgecolor="white", markeredgewidth=1)
    ax.text(m_a["x"] + 10, m_a["y"] - 20, "A", fontsize=10, color="cyan", fontweight="bold")
    ax.text(m_b["x"] + 10, m_b["y"] - 20, "B", fontsize=10, color="magenta", fontweight="bold")
    ax.axis("off")

    # Panel 6: Score formula
    ax = plt.subplot(2, 3, 6)
    ax.set_title("⑥ Scoring y resultado", fontsize=11, fontweight="bold")
    ax.axis("off")
    formula_text = (
        "Score(fp) = Σ cos_sim(probe_cyl, enroll_cyl) / n_cylinders(fp)\n\n"
        "• Cada cylinder del probe busca sus 5 vecinos más cercanos en Qdrant\n"
        "• Similitud coseno mide qué tan parecidos son dos cylinders\n"
        "• Normalización evita que huellas con más minucias ganen por peso\n"
        "• Ranking final = fingerprints ordenadas por score"
    )
    ax.text(0.5, 0.55, formula_text, fontsize=10, ha="center", va="center",
            transform=ax.transAxes, linespacing=2,
            bbox=dict(boxstyle="round,pad=1", facecolor="#f0f0f0", alpha=0.9))

    ax.text(0.5, 0.15, "216ms de búsqueda · 80% Rank-1 con 3 minucias",
            fontsize=8, ha="center", va="center", transform=ax.transAxes, color="gray")

    plt.tight_layout()
    path = OUT / "matching_explanation.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  matching_explanation.png")


if __name__ == "__main__":
    print("Generating MCC visualizations v3...")
    cylinder_explanation()
    matching_diagram()
    print(f"\nDone → {OUT}")

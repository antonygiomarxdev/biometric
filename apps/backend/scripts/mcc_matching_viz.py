"""
MCC matching visualization — how search & comparison works.

Shows: enrolled print → probe → cylinder matching → overlap → ranking.
"""

import math, sys, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.lines import Line2D

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


def matching_diagram():
    """Explain how MCC matching works step by step."""
    # Pick first and second image (different fingers of same person)
    imgs = sorted(SOCOF.glob("*.BMP"))
    img1 = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)  # enrolled
    img2 = cv2.imread(str(imgs[1]), cv2.IMREAD_GRAYSCALE)  # probe

    minu1, skel1, orient1, freq1 = run_pipeline(img1)
    minu2, skel2, orient2, freq2 = run_pipeline(img2)

    desc1 = extract_cylinders(minu1, skel1, orient1, freq1, DEFAULT_CONFIG)
    desc2 = extract_cylinders(minu2, skel2, orient2, freq2, DEFAULT_CONFIG)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Cómo funciona el matching MCC — Búsqueda y comparación de huellas",
                 fontsize=15, fontweight="bold", y=0.99)

    # ═══════════════ PANEL A: La huella enrolada ═══════════════
    ax = plt.subplot(2, 4, 1)
    ax.imshow(img1, cmap="gray")
    ax.set_title("① Huella enrolada (BD)", fontsize=11, fontweight="bold", loc="left")
    ax.axis("off")

    # ═══════════════ PANEL B: La huella latente (probe) ═══════════════
    ax = plt.subplot(2, 4, 2)
    ax.imshow(img2, cmap="gray")
    ax.set_title("② Huella latente (búsqueda)", fontsize=11, fontweight="bold", loc="left")
    ax.axis("off")

    # ═══════════════ PANEL C: Ambas con cylinders ═══════════════
    ax = plt.subplot(2, 4, 3)
    ax.set_title("③ Cylinders por minucia", fontsize=11, fontweight="bold", loc="left")

    # Show enrolled fingerprint with its minutiae
    ax.imshow(img1, cmap="gray")

    # Draw 3 representative cylinders on enrolled side
    if len(minu1) >= 3:
        colors = ["cyan", "yellow", "magenta"]
        for idx, (m, c) in enumerate(zip(minu1[20:23], colors)):
            cx, cy = m["x"], m["y"]
            for r in DEFAULT_CONFIG.ring_boundaries[1:]:  # skip innermost
                circle = Circle((cx, cy), r, fill=False, edgecolor=c, lw=1.2, ls="--", alpha=0.6)
                ax.add_patch(circle)
            ax.plot(cx, cy, "o", color=c, markersize=6, markeredgecolor="white", markeredgewidth=0.5)

        # Label
        ax.text(minu1[20]["x"] + 90, minu1[20]["y"] - 40, "3 minucias\ncon sus cylinders",
                fontsize=7, color="cyan", fontweight="bold")

    ax.axis("off")

    # ═══════════════ PANEL D: Overlap de cylinders ═══════════════
    ax = plt.subplot(2, 4, 4)
    ax.set_title("④ Overlap — misma cresta en 2 cylinders", fontsize=11, fontweight="bold", loc="left")

    if len(minu1) >= 3:
        # Show two close minutiae with overlapping cylinders
        m_a = minu1[20]
        m_b = minu1[22]

        # Crop to the overlap region
        cx = (m_a["x"] + m_b["x"]) / 2
        cy = (m_a["y"] + m_b["y"]) / 2
        margin = 160
        x0 = max(0, int(cx - margin))
        y0 = max(0, int(cy - margin))
        x1 = min(img1.shape[1], int(cx + margin))
        y1 = min(img1.shape[0], int(cy + margin))

        ax.imshow(img1[y0:y1, x0:x1], cmap="gray", extent=[x0, x1, y1, y0])

        # Skeleton overlay
        if skel1 is not None:
            sk_over = np.zeros((y1 - y0, x1 - x0, 4))
            sk_over[skel1[y0:y1, x0:x1] > 0] = [0.2, 0.8, 0.2, 0.4]
            ax.imshow(sk_over, extent=[x0, x1, y1, y0])

        # Cylinder A (cyan)
        for r in [DEFAULT_CONFIG.ring_boundaries[1], DEFAULT_CONFIG.ring_boundaries[2]]:
            ax.add_patch(Circle((m_a["x"], m_a["y"]), r, fill=False, edgecolor="cyan", lw=1.5, ls="--", alpha=0.5))
        ax.plot(m_a["x"], m_a["y"], "o", color="cyan", markersize=8, markeredgecolor="white", markeredgewidth=1)
        ax.text(m_a["x"] - 40, m_a["y"] - 20, "Minucia A", fontsize=8, color="cyan", fontweight="bold")

        # Cylinder B (magenta)
        for r in [DEFAULT_CONFIG.ring_boundaries[1], DEFAULT_CONFIG.ring_boundaries[2]]:
            ax.add_patch(Circle((m_b["x"], m_b["y"]), r, fill=False, edgecolor="magenta", lw=1.5, ls="--", alpha=0.5))
        ax.plot(m_b["x"], m_b["y"], "o", color="magenta", markersize=8, markeredgecolor="white", markeredgewidth=1)
        ax.text(m_b["x"] + 10, m_b["y"] - 20, "Minucia B", fontsize=8, color="magenta", fontweight="bold")

    ax.axis("off")

    # ═══════════════ PANEL E: Descriptores como vectores ═══════════════
    ax = plt.subplot(2, 4, 5)
    ax.set_title("⑤ Cada cylinder → vector 144D", fontsize=11, fontweight="bold", loc="left")
    if len(desc1) > 3:
        # Show 3 descriptor vectors side by side
        d1 = desc1[20]
        d2 = desc1[21]
        d3 = desc1[22]
        matrix = np.vstack([d1, d2, d3])
        ax.imshow(matrix, cmap="plasma", aspect="auto")
        ax.set_xlabel("Dimensiones del descriptor (144D)")
        ax.set_ylabel("Minucia")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["A", "B", "C"])
        # Cosine similarity annotations
        sim_ab = np.dot(d1, d2)
        sim_bc = np.dot(d2, d3)
        ax.text(72, 1, f"cos(A,B)={sim_ab:.2f}", ha="center", fontsize=7, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
    ax.axis("off")

    # ═══════════════ PANEL F: Búsqueda en Qdrant ═══════════════
    ax = plt.subplot(2, 4, 6)
    ax.set_title("⑥ Búsqueda vectorial (Qdrant)", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Text-based diagram
    steps = [
        ("Enrollar", "80 cylinders × 144D → Qdrant"),
        ("Cada cylinder del probe", "→ busca los 5 más parecidos"),
        ("Votación", "cada match vota por su fingerprint"),
        ("Score normalizado", "divide por n° de cylinders"),
        ("Resultado", "ranking de fingerprints"),
    ]
    for i, (step, desc) in enumerate(steps):
        y = 0.85 - i * 0.16
        ax.text(0.05, y, step, fontsize=9, fontweight="bold", va="center")
        ax.text(0.5, y, desc, fontsize=9, va="center", color="gray")

    # Arrow
    ax.annotate("", xy=(0.95, 0.1), xytext=(0.95, 0.85),
                arrowprops=dict(arrowstyle="->", lw=2, color="green"))

    # ═══════════════ PANEL G: Score matching ═══════════════
    ax = plt.subplot(2, 4, 7)
    ax.set_title("⑦ Scoring — ¿qué huella gana?", fontsize=11, fontweight="bold", loc="left")

    # Simulated ranking
    fps = ["Fp_572", "Fp_502", "Fp_183", "Fp_265", "Fp_119", "Fp_303", "Fp_281", "Fp_203", "Fp_176", "Fp_5"]
    scores = [2.58, 1.96, 1.45, 1.32, 1.18, 0.97, 0.82, 0.71, 0.65, 0.58]
    colors = ["#2ecc71"] + ["#3498db"] * 9

    bars = ax.barh(range(len(fps)), scores, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(range(len(fps)))
    ax.set_yticklabels(fps)
    ax.set_xlabel("Score normalizado")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Highlight winner
    ax.text(scores[0] + 0.1, 0, f"✓ GANADOR", fontsize=9, fontweight="bold",
            color="#2ecc71", va="center")

    # ═══════════════ PANEL H: Resultado final ═══════════════
    ax = plt.subplot(2, 4, 8)
    ax.set_title("⑧ Resultado para el perito", fontsize=11, fontweight="bold", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.7, "Top-1: Fp_572", fontsize=14, fontweight="bold", ha="center", color="#2ecc71")
    ax.text(0.5, 0.55, "Score: 2.58  |  Confidence: 90%", fontsize=10, ha="center")
    ax.text(0.5, 0.40, "———————————————", fontsize=8, ha="center")
    ax.text(0.5, 0.30, "Top-5: Fp_502, Fp_183, Fp_265, Fp_119", fontsize=9, ha="center", color="gray")
    ax.text(0.5, 0.20, "Búsqueda completada en 216ms", fontsize=9, ha="center", color="gray")

    plt.tight_layout()
    path = OUT / "matching_explanation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


if __name__ == "__main__":
    print("Generating matching explanation...")
    matching_diagram()
    print(f"Done → {OUT}")

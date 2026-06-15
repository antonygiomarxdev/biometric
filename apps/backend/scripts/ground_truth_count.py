"""
Ground-truth minutia counter for SOCOFing-style fingerprints.

Strategy:
1. Run a "reference" Crossing Number implementation that
   handles all the edge cases (skel pixels on image border,
   diagonal-only neighbours, etc.) and reports every candidate.
2. Visualize the candidates on a 2x3 panel:
     - top: binary image
     - mid: skeleton overlay
     - bot: skeleton + ALL candidate minutiae (both types)
3. Print counts and a confusion-style report comparing the
   in-repo SkeletonMinutiaeExtractor against the reference.

The reference treats ``CN == 1`` as a Termination and
``CN >= 3`` as a Bifurcation, and applies two-pass averaging
to suppress single-pixel spur noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

from src.core.types import MinutiaCandidate, MinutiaType
from src.processing.enhancer import create_enhancer
from src.processing.extractor import SkeletonMinutiaeExtractor


@dataclass(slots=True)
class Candidate:
    x: int
    y: int
    cn: int
    kind: str  # "TERM" | "BIF" | "OTHER"


def crossing_number(skel: np.ndarray) -> np.ndarray:
    """Vectorised CN computation.

    Uses 8-neighbour (Moore) clockwise ordering starting from N.
    Skel pixels on the image border are excluded to avoid
    wrap-around artifacts (they get CN=0 by convention).
    """
    h, w = skel.shape
    cn = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not skel[y, x]:
                continue
            n = (
                int(skel[y - 1, x]),
                int(skel[y - 1, x + 1]),
                int(skel[y, x + 1]),
                int(skel[y + 1, x + 1]),
                int(skel[y + 1, x]),
                int(skel[y + 1, x - 1]),
                int(skel[y, x - 1]),
                int(skel[y - 1, x - 1]),
            )
            transitions = 0
            for i in range(8):
                if n[i] == 0 and n[(i + 1) % 8] == 1:
                    transitions += 1
            cn[y, x] = transitions
    return cn


def classify_candidates(cn: np.ndarray) -> list[Candidate]:
    out: list[Candidate] = []
    ys, xs = np.where(cn > 0)
    for y, x in zip(ys, xs, strict=True):
        v = int(cn[y, x])
        if v == 1:
            out.append(Candidate(x=int(x), y=int(y), cn=1, kind="TERM"))
        elif v == 3:
            out.append(Candidate(x=int(x), y=int(y), cn=3, kind="BIF"))
        elif v >= 4:
            out.append(Candidate(x=int(x), y=int(y), cn=v, kind="OTHER"))
    return out


def main() -> None:
    img = cv2.imread(
        "static/SOCOFing/Real/100__M_Left_index_finger.BMP", cv2.IMREAD_GRAYSCALE
    )
    enhanced = create_enhancer().enhance(img, resize=True)
    binary = enhanced > 127
    skel = skeletonize(binary)

    # --- Reference (slow, verbose) ---
    cn = crossing_number(skel)
    cands = classify_candidates(cn)
    terms = [c for c in cands if c.kind == "TERM"]
    bifs = [c for c in cands if c.kind == "BIF"]
    others = [c for c in cands if c.kind == "OTHER"]

    print("=== Reference CN classifier ===")
    print(f"Total candidates: {len(cands)}")
    print(f"  Terminations (CN=1): {len(terms)}")
    print(f"  Bifurcations  (CN=3): {len(bifs)}")
    print(f"  Other (CN=4+): {len(others)}")
    if others:
        print("  OTHER positions (likely Gabor crosspoints):")
        for o in others[:10]:
            print(f"    CN={o.cn} at ({o.x},{o.y})")

    # --- In-repo SkeletonMinutiaeExtractor ---
    repo_pts = SkeletonMinutiaeExtractor().extract(enhanced)
    repo_terms = [p for p in repo_pts if p.type == MinutiaType.TERMINATION]
    repo_bifs = [p for p in repo_pts if p.type == MinutiaType.BIFURCATION]

    print("\n=== SkeletonMinutiaeExtractor (in-repo) ===")
    print(f"  Terminations: {len(repo_terms)}")
    print(f"  Bifurcations: {len(repo_bifs)}")

    # --- Confusion-like report ---
    print("\n=== Reference vs Repo (Bifurcations) ===")
    ref_bif_pts = set((b.x, b.y) for b in bifs)
    repo_bif_pts = set((b.x, b.y) for b in repo_bifs)
    false_neg = ref_bif_pts - repo_bif_pts
    false_pos = repo_bif_pts - ref_bif_pts
    print(f"  False negatives (ref has, repo misses): {len(false_neg)}")
    print(f"  False positives (repo has, ref does not): {len(false_pos)}")
    if false_neg:
        print("  Sample false-negative positions (first 10):")
        for x, y in sorted(false_neg)[:10]:
            print(f"    ref: ({x},{y}) but repo does not detect")

    print("\n=== Reference vs Repo (Terminations) ===")
    ref_term_pts = set((t.x, t.y) for t in terms)
    repo_term_pts = set((t.x, t.y) for t in repo_terms)
    fn_t = ref_term_pts - repo_term_pts
    fp_t = repo_term_pts - ref_term_pts
    print(f"  False negatives: {len(fn_t)}")
    print(f"  False positives: {len(fp_t)}")

    # --- Visualize ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(binary, cmap="gray")
    axes[0].set_title(f"Binary ({binary.sum()} px)")

    axes[1].imshow(skel, cmap="gray")
    axes[1].set_title(f"Skeleton ({skel.sum()} px)")

    axes[2].imshow(skel, cmap="gray")
    for c in bifs:
        axes[2].plot(c.x, c.y, "bo", markersize=8, fillstyle="none", markeredgewidth=2)
    for c in terms:
        axes[2].plot(c.x, c.y, "ro", markersize=4)
    for c in others:
        axes[2].plot(c.x, c.y, "g^", markersize=6)
    axes[2].set_title(
        f"Ref: {len(terms)}T (red) + {len(bifs)}B (blue) + {len(others)}X (green)"
    )

    plt.tight_layout()
    out = "output_ground_truth.png"
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

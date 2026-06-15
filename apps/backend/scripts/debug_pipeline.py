"""
End-to-end pipeline debugger for :class:`FingerprintService`.

Runs the production pipeline (OrientationFieldAnalyzer → CpuEnhancer →
SkeletonExtractor → PreFusion QualityFilter → SpurRemover →
BrokenRidgeHealer → BorderMaskCleaner (core ROI) → OrientationRefiner →
LowConfidenceFilter) on a single SOCOFing image, captures
intermediate state at every stage, and writes the results into a
structured directory:

    test_results/pipeline_debug/<image_slug>/
        meta.json                — stage counts, timing, settings
        00_input.png             — raw 96x103 input
        01_orientation.png       — OrientationFieldAnalyzer + mask
        02_singularity.png       — SingularityDetector ROI
        03_enhanced.png          — CpuEnhancer (Gabor) + ROI overlay
        04_skeleton_raw.png      — SkeletonExtractor raw candidates
        05_pre_fusion_quality    — pre-fusion ROI filter
        06_fused.png             — EnsembleFusion (clustered)
        07_spur_remover.png
        08_broken_ridge.png
        09_border_mask.png       — BorderMaskCleaner (core ROI)
        10_orientation_refiner.png
        11_low_confidence.png
        99_overview.png          — contact sheet of all stages

Usage::

    python3 scripts/debug_pipeline.py [image_path]

    python3 scripts/debug_pipeline.py
        # defaults to the CR-altered fingerprint
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize

from src.core.interfaces import PipelineContext
from src.core.types import MinutiaCandidate, MinutiaType
from src.processing.enhancer import create_enhancer
from src.processing.extractor import SkeletonMinutiaeExtractor
from src.processing.post_hooks import (
    BorderMaskCleaner,
    BrokenRidgeHealer,
    EnsembleFusionFilter,
    LowConfidenceFilter,
    OrientationRefiner,
    QualityFilter,
    SpurRemover,
)
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Slug helper: convert any image path to a short filesystem-friendly name.
# ---------------------------------------------------------------------------

def slugify(path: Path) -> str:
    """``static/SOCOFing/Altered-Easy/100__M_Left_index_finger_CR.BMP``
    → ``altered_easy__100__M_Left_index_finger_CR``."""
    stem = path.stem
    parent = path.parent.name
    parent_slug = re.sub(r"[^A-Za-z0-9]+", "_", parent).strip("_").lower()
    return f"{parent_slug}__{stem}"


# ---------------------------------------------------------------------------
# Stage dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Stage:
    """Snapshot of one pipeline stage."""

    name: str
    slug: str
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    points: list[MinutiaCandidate] = field(default_factory=list)
    note: str = ""


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _plot_points(ax, base: np.ndarray, pts: list[MinutiaCandidate], title: str, limit: int = 400) -> None:
    ax.imshow(base, cmap="gray")
    for m in pts[:limit]:
        c = "red" if m.type == MinutiaType.TERMINATION else "blue"
        ax.plot(m.x, m.y, "o", color=c, markersize=4, fillstyle="none", markeredgewidth=1.5)
    if len(pts) > limit:
        ax.text(
            0.02, 0.95, f"+{len(pts) - limit} more",
            transform=ax.transAxes, color="yellow", fontsize=8,
        )
    ax.set_title(f"{title} ({len(pts)} pts)", fontsize=10)


def _plot_mask_overlay(ax, base: np.ndarray, mask: np.ndarray, title: str) -> None:
    color = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    if mask is not None and mask.shape == base.shape:
        red_overlay = color.copy()
        red_overlay[~mask] = [255, 0, 0]
        cv2.addWeighted(red_overlay, 0.35, color, 0.65, 0, color)
    ax.imshow(color)
    pct = 100.0 * mask.mean() if mask is not None else 0
    ax.set_title(f"{title} ({pct:.0f}% valid)", fontsize=10)


def _plot_bif_term_summary(
    ax, base: np.ndarray, pts: list[MinutiaCandidate], title: str
) -> None:
    """Highlight a single panel with a stacked bar of bif vs term counts."""
    ax.imshow(base, cmap="gray")
    for m in pts:
        c = "red" if m.type == MinutiaType.TERMINATION else "blue"
        ax.plot(m.x, m.y, "o", color=c, markersize=4, fillstyle="none", markeredgewidth=1.5)
    bifs = sum(1 for m in pts if m.type == MinutiaType.BIFURCATION)
    terms = sum(1 for m in pts if m.type == MinutiaType.TERMINATION)
    ax.set_title(f"{title}  B:{bifs}  T:{terms}  ({len(pts)} pts)", fontsize=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        img_path = Path(
            "static/SOCOFing/Altered/Altered-Easy/100__M_Left_index_finger_CR.BMP"
        )
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    slug = slugify(img_path)
    out_dir = Path("test_results/pipeline_debug") / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing pipeline artefacts to: {out_dir}")

    print(f"Processing: {img_path}")
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"  cannot read {img_path}")
        sys.exit(1)
    print(f"  raw shape: {raw.shape}")

    stages: list[Stage] = []
    t0 = time.time()

    stages.append(Stage("0. INPUT", "00_input", image=raw, note=f"shape={raw.shape}"))

    # 1. OrientationFieldAnalyzer (on raw)
    pre = OrientationFieldAnalyzer(block_size=16, coherence_threshold=0.35)
    ctx_raw = PipelineContext(raw_image=raw)
    pre.process(ctx_raw)
    stages.append(
        Stage(
            "1. OrientationFieldAnalyzer", "01_orientation",
            image=ctx_raw.preprocessed_image, mask=ctx_raw.quality_mask,
            note=f"ori={pre.orientation_field.shape} meanCoh="
            f"{float(pre.coherence_field[pre.coherence_field > 0].mean()):.2f}"
            if (pre.coherence_field > 0).any() else "no valid blocks",
        )
    )

    # 1b. SingularityDetector (on raw)
    sing = SingularityDetector(window=5, poi_threshold=0.5, roi_radius=100)
    sing_ctx = PipelineContext(
        raw_image=raw,
        orientation_field=ctx_raw.orientation_field,
        coherence_field=ctx_raw.coherence_field,
        quality_mask=ctx_raw.quality_mask.copy() if ctx_raw.quality_mask is not None else None,
    )
    sing.process(sing_ctx)
    core_str = f"core=({sing.core[0]},{sing.core[1]})" if sing.core else "no core"
    delta_str = f"delta=({sing.delta[0]},{sing.delta[1]})" if sing.delta else "no delta"
    stages.append(
        Stage(
            "1b. SingularityDetector (raw)", "02_singularity",
            image=sing_ctx.preprocessed_image, mask=sing_ctx.quality_mask,
            note=f"{core_str} {delta_str} ROI={100*sing_ctx.quality_mask.mean():.0f}%",
        )
    )

    # 2. Enhancer (CPU via create_enhancer legacy path)
    from src.core.interfaces import PipelineContext as Ctx
    from src.processing.enhancer import create_enhancer as ce
    from src.processing.enhancers.base import EnhancerConfig
    from src.processing.enhancers.cpu import CpuEnhancer
    enhancer = CpuEnhancer(EnhancerConfig())
    enhanced = enhancer.enhance(raw, resize=True)
    pre_enh = OrientationFieldAnalyzer(block_size=16, coherence_threshold=0.35)
    ctx_enh = PipelineContext(raw_image=enhanced)
    pre_enh.process(ctx_enh)
    sing_enh = SingularityDetector(window=5, poi_threshold=0.5, roi_radius=100)
    sing_enh_ctx = PipelineContext(
        raw_image=enhanced,
        orientation_field=ctx_enh.orientation_field,
        coherence_field=ctx_enh.coherence_field,
        quality_mask=ctx_enh.quality_mask.copy() if ctx_enh.quality_mask is not None else None,
    )
    sing_enh.process(sing_enh_ctx)
    if sing_enh_ctx.quality_mask is not None and ctx_enh.quality_mask is not None:
        mask_resized = sing_enh_ctx.quality_mask & ctx_enh.quality_mask
    elif sing_enh_ctx.quality_mask is not None:
        mask_resized = sing_enh_ctx.quality_mask
    else:
        mask_resized = ctx_enh.quality_mask
    core_enh = f"core=({sing_enh.core[0]},{sing_enh.core[1]})" if sing_enh.core else "no core"
    stages.append(
        Stage(
            "2. CpuEnhancer (Gabor)", "03_enhanced",
            image=enhanced, mask=mask_resized,
            note=f"shape={enhanced.shape} {core_enh} ROI={100*mask_resized.mean():.0f}%",
        )
    )

    pipe_ctx = PipelineContext(
        raw_image=raw,
        preprocessed_image=enhanced,
        enhanced_image=enhanced,
        quality_mask=mask_resized,
        roi_mask=sing_enh_ctx.quality_mask,
        orientation_field=ctx_enh.orientation_field,
        coherence_field=ctx_enh.coherence_field,
    )

    # 3-4. Raw extractors
    skel_raw = SkeletonMinutiaeExtractor().extract(enhanced)
    stages.append(Stage("3. SkeletonExtractor (raw)", "04_skeleton_raw",
                        image=enhanced, mask=mask_resized, points=skel_raw))

    pipe_ctx.candidate_groups = [skel_raw]
    chain = [
        ("5. EnsembleFusion (avg)", "06_fused", EnsembleFusionFilter(radius=8.0, min_votes=2)),
        ("6. QualityFilter", "07_quality_filter", QualityFilter()),
        ("7. SpurRemover", "08_spur_remover", SpurRemover(max_distance=10.0)),
        ("8. BrokenRidgeHealer", "09_broken_ridge", BrokenRidgeHealer(max_distance=8.0)),
        ("9. BorderMaskCleaner (core ROI, border_px=0)", "10_border_mask", BorderMaskCleaner(border_px=0, roi_mode="core")),
        ("10. OrientationRefiner", "11_orientation_refiner", OrientationRefiner(window=16, coherence_threshold=0.65)),
        ("11. LowConfidenceFilter (FINAL)", "12_low_confidence", LowConfidenceFilter(threshold=0.75)),
    ]
    for name, slug, hook in chain:
        hook.process(pipe_ctx)
        stages.append(Stage(name, slug, image=enhanced, mask=pipe_ctx.quality_mask, points=list(pipe_ctx.candidates)))
    after_lcf = pipe_ctx.candidates
    elapsed = time.time() - t0

    # ---- Save meta.json ----
    meta = {
        "input": str(img_path),
        "slug": slug,
        "elapsed_s": elapsed,
        "raw_shape": list(raw.shape),
        "enhanced_shape": list(enhanced.shape),
        "stages": [
            {
                "name": st.name,
                "slug": st.slug,
                "n_points": len(st.points),
                "bifurcations": sum(1 for p in st.points if p.type == MinutiaType.BIFURCATION),
                "terminations": sum(1 for p in st.points if p.type == MinutiaType.TERMINATION),
                "mask_valid_pct": 100.0 * st.mask.mean() if st.mask is not None else None,
                "note": st.note,
            }
            for st in stages
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- Save per-stage PNGs ----
    for st in stages:
        if st.image is None:
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        if st.mask is not None and st.slug in {
            "01_orientation", "02_singularity", "03_enhanced"
        }:
            _plot_mask_overlay(ax, st.image, st.mask, st.name)
        elif st.points:
            _plot_bif_term_summary(ax, st.image, st.points, st.name)
        else:
            ax.imshow(st.image, cmap="gray")
            ax.set_title(st.name, fontsize=11)
        if st.note:
            ax.text(
                0.02, 0.02, st.note,
                transform=ax.transAxes, color="yellow", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=2),
            )
        plt.tight_layout()
        plt.savefig(out_dir / f"{st.slug}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ---- Contact sheet (overview) ----
    n_stages = len(stages)
    cols = 3
    rows = (n_stages + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5))
    axes = np.array(axes).reshape(-1)
    mask_stage_indices = {1, 2, 3}
    for i, st in enumerate(stages):
        ax = axes[i]
        if st.image is None:
            ax.axis("off")
            continue
        if i == 0:
            ax.imshow(st.image, cmap="gray")
            ax.set_title(st.name, fontsize=11)
        elif st.mask is not None and i in mask_stage_indices:
            _plot_mask_overlay(ax, st.image, st.mask, st.name)
        else:
            _plot_bif_term_summary(ax, st.image, st.points, st.name)
        if st.note:
            ax.text(
                0.02, 0.02, st.note,
                transform=ax.transAxes, color="yellow", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=2),
            )
    for j in range(n_stages, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "99_overview.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote 13 stage PNGs + 99_overview.png + meta.json into {out_dir}")


if __name__ == "__main__":
    main()

"""Visualizador extendido para la caja negra (Spike 02).

Runs the new ``detect()`` black box and renders a 2x3 panel that
shows the new metadata: zone, ridge trace length, overlap flags,
all singularities, and the pattern area mask.

This is the diagnostic the perito uses to decide if the new
validation actually improves the output.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "apps" / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.core.types import MinutiaType
from validation_spike import detect
from types_spike import SingularityKind, Zone

SOCOFING_REAL = ROOT / "static" / "SOCOFing" / "Real"
SOCOFING_ALTERED = ROOT / "static" / "SOCOFing" / "Altered" / "Altered-Easy"
OUT_DIR = Path(__file__).resolve().parent / "sample_outputs"


ZONE_COLOURS: dict[Zone, tuple[int, int, int]] = {
    Zone.BORDER: (255, 200, 0),
    Zone.INTERIOR: (0, 200, 0),
    Zone.NEAR_CORE: (255, 0, 255),
    Zone.NEAR_DELTA: (0, 200, 255),
}


def _default_samples() -> list[Path]:
    samples: list[Path] = []
    samples.extend(sorted(SOCOFING_REAL.glob("100__*.BMP"))[:1])
    if SOCOFING_ALTERED.exists():
        samples.extend(sorted(SOCOFING_ALTERED.glob("100__*_CR.BMP"))[:1])
        samples.extend(sorted(SOCOFING_ALTERED.glob("100__*_Zcut.BMP"))[:1])
        samples.extend(sorted(SOCOFING_REAL.glob("101__*.BMP"))[:1])
        samples.extend(sorted(SOCOFING_ALTERED.glob("101__*_Zcut.BMP"))[:1])
    return [p for p in samples if p.exists()]


def _draw_minutia(
    canvas: np.ndarray,
    x: int,
    y: int,
    angle: float,
    zone: Zone,
    confidence: float,
    is_termination: bool,
    ridge_trace_length: int,
    is_overlap: bool,
) -> None:
    """Render a single minutia on a BGR canvas with forensic metadata."""
    color = ZONE_COLOURS[zone]
    radius = 3 + int(4 * confidence)
    if is_termination:
        cv2.circle(canvas, (x, y), radius, color, 1, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(
            canvas, (x - radius, y - radius), (x + radius, y + radius),
            color, 1, lineType=cv2.LINE_AA,
        )
    line_length = 6 + int(8 * confidence)
    dx = int(round(line_length * np.cos(angle)))
    dy = int(round(line_length * np.sin(angle)))
    cv2.line(canvas, (x, y), (x + dx, y + dy), color, 1, lineType=cv2.LINE_AA)
    if is_overlap:
        cv2.drawMarker(
            canvas, (x, y), (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=10, thickness=1, line_type=cv2.LINE_AA,
        )


def _draw_singularity(
    canvas: np.ndarray,
    x: int, y: int, kind: SingularityKind,
) -> None:
    if kind == SingularityKind.CORE:
        cv2.drawMarker(
            canvas, (x, y), (0, 255, 0),
            markerType=cv2.MARKER_STAR, markerSize=18, thickness=2,
            line_type=cv2.LINE_AA,
        )
    else:
        cv2.drawMarker(
            canvas, (x, y), (0, 0, 255),
            markerType=cv2.MARKER_TRIANGLE_DOWN, markerSize=18, thickness=2,
            line_type=cv2.LINE_AA,
        )


def _overlay_pattern_mask(
    image: np.ndarray, mask: np.ndarray | None,
) -> np.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if mask is None:
        return out
    overlay = out.copy()
    overlay[mask] = (180, 180, 100)
    return cv2.addWeighted(out, 0.6, overlay, 0.4, 0)


def _make_grid(image: np.ndarray, result: Any) -> tuple[plt.Figure, dict[str, Any]]:
    enhanced = result.enhanced_image
    skeleton = result.skeleton
    minutiae = result.minutiae
    cores = result.cores
    deltas = result.deltas
    pattern_mask = result.pattern_area_mask

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Spike 02: Black-Box Detector — {result.num_minutiae} validated "
        f"({result.num_cores} cores, {result.num_deltas} deltas)",
        fontsize=14, fontweight="bold",
    )

    # Panel 1: enhanced + validated minutiae (colour by zone)
    ax = axes[0, 0]
    canvas = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    for m in minutiae:
        _draw_minutia(
            canvas, m.x, m.y, m.angle, m.zone, m.confidence,
            m.type == MinutiaType.TERMINATION, m.ridge_trace_length, m.is_overlap,
        )
    for c in cores:
        _draw_singularity(canvas, c.x, c.y, c.kind)
    for d in deltas:
        _draw_singularity(canvas, d.x, d.y, d.kind)
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    zone_counts: dict[str, int] = {}
    for m in minutiae:
        zone_counts[m.zone.name] = zone_counts.get(m.zone.name, 0) + 1
    ax.set_title(
        f"1. Enhanced + validated minutiae\n"
        f"   zones: {zone_counts}",
        fontsize=10,
    )
    ax.axis("off")

    # Panel 2: skeleton
    ax = axes[0, 1]
    skel_viz = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    for m in minutiae:
        _draw_minutia(
            skel_viz, m.x, m.y, m.angle, m.zone, m.confidence,
            m.type == MinutiaType.TERMINATION, m.ridge_trace_length, m.is_overlap,
        )
    ax.imshow(cv2.cvtColor(skel_viz, cv2.COLOR_BGR2RGB))
    ax.set_title("2. Skeleton + minutiae", fontsize=10)
    ax.axis("off")

    # Panel 3: pattern area mask overlay
    ax = axes[0, 2]
    overlay = _overlay_pattern_mask(enhanced, pattern_mask)
    for m in minutiae:
        if m.in_pattern_area:
            cv2.circle(overlay, (m.x, m.y), 4, (0, 255, 0), 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(overlay, (m.x, m.y), 4, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    in_pa = sum(1 for m in minutiae if m.in_pattern_area)
    ax.set_title(
        f"3. Pattern area (green=inside, red=outside)\n"
        f"   {in_pa}/{len(minutiae)} inside",
        fontsize=10,
    )
    ax.axis("off")

    # Panel 4: confidence heatmap
    ax = axes[1, 0]
    h, w = enhanced.shape
    qimg = np.zeros((h, w, 3), dtype=np.uint8)
    for m in minutiae:
        q = m.confidence
        b, g, r = 0, int(255 * q), int(255 * (1.0 - q))
        cv2.circle(qimg, (m.x, m.y), 5, (b, g, r), -1, lineType=cv2.LINE_AA)
    ax.imshow(cv2.cvtColor(qimg, cv2.COLOR_BGR2RGB))
    confidences = [m.confidence for m in minutiae]
    mean_q = float(np.mean(confidences)) if confidences else 0.0
    ax.set_title(f"4. Confidence (mean={mean_q:.2f})", fontsize=10)
    ax.axis("off")

    # Panel 5: overlap flagging
    ax = axes[1, 1]
    ov_canvas = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    n_overlap = sum(1 for m in minutiae if m.is_overlap)
    for m in minutiae:
        if m.is_overlap:
            cv2.drawMarker(
                ov_canvas, (m.x, m.y), (0, 0, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=14, thickness=2, line_type=cv2.LINE_AA,
            )
        else:
            cv2.circle(ov_canvas, (m.x, m.y), 3, (0, 200, 0), 1, lineType=cv2.LINE_AA)
    ax.imshow(cv2.cvtColor(ov_canvas, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"5. Overlap detection (red=crossing, green=real)\n"
        f"   {n_overlap} overlaps flagged",
        fontsize=10,
    )
    ax.axis("off")

    # Panel 6: summary
    ax = axes[1, 2]
    n_border = sum(1 for m in minutiae if m.zone == Zone.BORDER)
    n_interior = sum(1 for m in minutiae if m.zone == Zone.INTERIOR)
    n_near_core = sum(1 for m in minutiae if m.zone == Zone.NEAR_CORE)
    n_near_delta = sum(1 for m in minutiae if m.zone == Zone.NEAR_DELTA)
    high_conf = sum(1 for m in minutiae if m.confidence >= 0.7)
    low_conf = sum(1 for m in minutiae if m.confidence < 0.4)
    avg_trace = float(np.mean([m.ridge_trace_length for m in minutiae])) if minutiae else 0.0
    term_count = sum(1 for m in minutiae if m.type == MinutiaType.TERMINATION)
    bif_count = sum(1 for m in minutiae if m.type == MinutiaType.BIFURCATION)
    timings = result.metadata.get("timings", {})
    summary = (
        f"raw candidates:  {result.metadata.get('n_raw_candidates', 0)}\n"
        f"validated:       {result.num_minutiae}\n"
        f"  border:        {n_border}\n"
        f"  interior:      {n_interior}\n"
        f"  near core:     {n_near_core}\n"
        f"  near delta:    {n_near_delta}\n"
        f"  high conf:     {high_conf}\n"
        f"  low conf:      {low_conf}\n"
        f"  overlap:       {n_overlap}\n"
        f"  avg trace:     {avg_trace:.1f} px\n"
        f"\n"
        f"type: T={term_count}  B={bif_count}\n"
        f"cores: {result.num_cores}  deltas: {result.num_deltas}\n"
        f"\n"
        f"enhance:  {timings.get('enhance_ms', 0)} ms\n"
        f"norm:     {timings.get('norm_ms', 0)} ms\n"
        f"thin:     {timings.get('thin_ms', 0)} ms\n"
        f"cn:       {timings.get('cn_ms', 0)} ms"
    )
    ax.text(
        0.05, 0.5, summary, fontsize=10, verticalalignment="center",
        family="monospace", color="white",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#222", alpha=0.85),
    )
    ax.set_title("6. Summary", fontsize=10)
    ax.axis("off")
    ax.set_facecolor("#111")

    plt.tight_layout()

    summary_dict = {
        "n_raw_candidates": result.metadata.get("n_raw_candidates", 0),
        "n_validated": result.num_minutiae,
        "n_border": n_border,
        "n_interior": n_interior,
        "n_near_core": n_near_core,
        "n_near_delta": n_near_delta,
        "n_high_conf": high_conf,
        "n_low_conf": low_conf,
        "n_overlap": n_overlap,
        "n_cores": result.num_cores,
        "n_deltas": result.num_deltas,
        "mean_confidence": mean_q,
        "avg_trace_px": avg_trace,
    }
    return fig, summary_dict


def main(argv: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    subdir = OUT_DIR / timestamp
    subdir.mkdir(exist_ok=True)

    if len(argv) > 1:
        samples = [Path(p) for p in argv[1:]]
    else:
        samples = _default_samples()

    if not samples:
        print(f"No samples found. Checked: {SOCOFING_REAL}, {SOCOFING_ALTERED}")
        sys.exit(1)

    print(f"Visualising {len(samples)} image(s) with the spike black-box detector.")
    print(f"Output: {subdir.relative_to(ROOT.parent.parent)}")
    print()

    aggregate: list[dict[str, Any]] = []
    for img_path in samples:
        print(f"  -> {img_path.name}")
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("     skipped: could not load")
            continue
        result = detect(image)
        fig, summary = _make_grid(image, result)
        out_path = subdir / f"{img_path.stem}.png"
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(
            f"     raw={summary['n_raw_candidates']:3d}  "
            f"validated={summary['n_validated']:3d}  "
            f"overlap={summary['n_overlap']:2d}  "
            f"cores={summary['n_cores']}  "
            f"deltas={summary['n_deltas']}  "
            f"mean_q={summary['mean_confidence']:.2f}"
        )
        summary["image"] = img_path.name
        aggregate.append(summary)

    summary_path = subdir / "summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2))
    print()
    print(f"Summary: {summary_path.relative_to(ROOT.parent.parent)}")
    print(f"Visualisations: {subdir.relative_to(ROOT.parent.parent)}")


if __name__ == "__main__":
    main(sys.argv)

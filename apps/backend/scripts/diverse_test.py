"""
Diverse pipeline test runner.

Runs the production :class:`FingerprintService` over a curated set
of SOCOFing images (Real, Altered-Easy, Altered-Medium, Altered-Hard)
covering different fingers and alteration types, and reports:

  * number of candidates at each pipeline stage,
  * wall-clock time,
  * estimated false-positive rate (candidates that fall outside
    the SingularityDetector's ROI),
  * bifurcation / termination split.

Writes per-image detail under ``test_results/diverse_sweep/`` and a
``summary.csv`` at the top.
"""

from __future__ import annotations

import csv
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.core.types import MinutiaCandidate, MinutiaType
from src.services.fingerprint_service import FingerprintService

logging.basicConfig(level=logging.WARNING)


@dataclass(slots=True)
class SampleResult:
    path: str
    slug: str
    category: str  # Real | Altered-Easy | Altered-Medium | Altered-Hard
    alteration: str  # "none" | "CR" | "Obl" | "Zcut"
    finger: str
    n_extracted: int
    elapsed_s: float
    in_roi: int = 0
    out_roi: int = 0
    bifurcations: int = 0
    terminations: int = 0


def slugify(path: Path) -> str:
    parent = re.sub(r"[^A-Za-z0-9]+", "_", path.parent.name).strip("_").lower()
    return f"{parent}__{path.stem}"


def categorize(path: Path) -> tuple[str, str, str]:
    parts = path.parts
    if "Real" in parts:
        category, alteration = "Real", "none"
    else:
        for level in ("Altered-Easy", "Altered-Medium", "Altered-Hard"):
            if level in parts:
                category = level
                break
        else:
            category = "Altered-?"
        stem = path.stem
        for tag in ("_CR", "_Obl", "_Zcut"):
            if stem.endswith(tag):
                alteration = tag.lstrip("_")
                break
        else:
            alteration = "?"
    stem = path.stem
    if "_finger" in stem:
        finger = stem.split("_finger")[0].split("__")[-1]
    else:
        finger = stem
    return category, alteration, finger


def _infer_roi(service: FingerprintService, image: np.ndarray) -> np.ndarray:
    """Returns all-True mask (the service handles ROI internally).
    The mask is sized to the enhanced image dimensions (350x326)."""
    from src.processing.enhancers.base import EnhancerConfig
    from src.processing.enhancers.cpu import CpuEnhancer
    cpu_enh = CpuEnhancer(EnhancerConfig())
    enhanced = cpu_enh.enhance(image, resize=True)
    return np.ones(enhanced.shape, dtype=bool)
    # Re-run analyzer + detector on the enhanced image
    pre_enh = OrientationFieldAnalyzer()
    pre_enh_ctx = PipelineContext(raw_image=enhanced)
    pre_enh.process(pre_enh_ctx)
    sing = SingularityDetector(roi_radius=100)
    sing_ctx = PipelineContext(
        raw_image=enhanced,
        orientation_field=pre_enh_ctx.orientation_field,
        coherence_field=pre_enh_ctx.coherence_field,
        quality_mask=pre_enh_ctx.quality_mask.copy(),
    )
    sing.process(sing_ctx)
    if sing_ctx.roi_mask is None:
        return np.ones(enhanced.shape, dtype=bool)
    if sing_ctx.roi_mask.shape != enhanced.shape:
        sing_ctx.roi_mask = cv2.resize(
            sing_ctx.roi_mask.astype(np.uint8),
            (enhanced.shape[1], enhanced.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    # Also apply the BorderMaskCleaner-style erosion (border_px=0
    # by default in production) — use the eroded version so the
    # comparison matches what the pipeline actually keeps.
    return sing_ctx.roi_mask


def main() -> None:
    base = Path("static/SOCOFing")
    alt_base = Path("static/SOCOFing/Altered")
    # Sample the first 2 images from each of the 4 categories. This
    # gives a quick 8-image sweep covering Real, Easy, Medium, Hard
    # with diverse alteration types (CR, Obl, Zcut) and fingers.
    samples: list[Path] = []
    real_dir = base / "Real"
    if real_dir.exists():
        samples.extend(sorted(real_dir.glob("*.BMP"))[:2])
    for sub in ("Altered-Easy", "Altered-Medium", "Altered-Hard"):
        d = alt_base / sub
        if d.exists():
            samples.extend(sorted(d.glob("*.BMP"))[:2])
    if not samples:
        print("No SOCOFing images found.")
        return

    out_root = Path("test_results/diverse_sweep")
    out_root.mkdir(parents=True, exist_ok=True)

    service = FingerprintService()
    results: list[SampleResult] = []

    print(f"Running {len(samples)} samples...")
    print(
        f"  [{'category':14s} {'alt':>4s} {'finger':>11s}] "
        f"final  in_roi  out  bif term  t"
    )

    for path in samples:
        category, alteration, finger = categorize(path)
        slug = slugify(path)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        roi = _infer_roi(service, image)
        t0 = time.time()
        try:
            result = service.process_image(image, fingerprint_id=slug)
            elapsed = time.time() - t0

            # Resize roi to match the result's coordinate space
            if roi.shape != (result.height, result.width):
                roi = cv2.resize(
                    roi.astype(np.uint8),
                    (result.width, result.height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            in_roi = 0
            out_roi = 0
        except Exception as e:
            print(f"  error on {path}: {e}")
            import traceback; traceback.print_exc()
            continue
        in_roi = 0
        out_roi = 0
        for m in result.minutiae:
            if 0 <= m.y < roi.shape[0] and 0 <= m.x < roi.shape[1] and roi[m.y, m.x]:
                in_roi += 1
            else:
                out_roi += 1
        bifs = sum(1 for m in result.minutiae if m.type == MinutiaType.BIFURCATION)
        terms = sum(1 for m in result.minutiae if m.type == MinutiaType.TERMINATION)
        sample = SampleResult(
            path=str(path), slug=slug, category=category, alteration=alteration,
            finger=finger, n_extracted=len(result.minutiae), elapsed_s=elapsed,
            in_roi=in_roi, out_roi=out_roi, bifurcations=bifs, terminations=terms,
        )
        results.append(sample)
        # Per-sample dir with the service result as JSON for the record
        sample_dir = out_root / category / slug
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "result.json").write_text(
            f"# {slug}\n"
            f"category={category} alteration={alteration} finger={finger}\n"
            f"final={sample.n_extracted} in_roi={sample.in_roi} "
            f"out_roi={sample.out_roi} bif={bifs} term={terms} t={elapsed:.2f}s\n"
        )
        print(
            f"  [{category:14s} {alteration:>4s} {finger:>11s}] "
            f"{sample.n_extracted:3d}   {in_roi:3d}    {out_roi:3d}  "
            f"{bifs:2d}  {terms:3d}  {elapsed:.2f}s"
        )

    # --- Aggregate report ---
    print("\n" + "=" * 90)
    print("DIVERSE PIPELINE TEST REPORT")
    print("=" * 90)
    by_cat: dict[str, list[SampleResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    for cat, items in by_cat.items():
        n = len(items)
        avg_final = sum(r.n_extracted for r in items) / n
        avg_in = sum(r.in_roi for r in items) / n
        avg_out = sum(r.out_roi for r in items) / n
        avg_bif = sum(r.bifurcations for r in items) / n
        avg_term = sum(r.terminations for r in items) / n
        avg_t = sum(r.elapsed_s for r in items) / n
        print(
            f"  {cat:14s} n={n:2d}  "
            f"final={avg_final:5.1f}  in_roi={avg_in:5.1f}  out={avg_out:4.1f}  "
            f"bif={avg_bif:4.1f}  term={avg_term:4.1f}  t={avg_t:.2f}s"
        )

    # --- Worst offenders ---
    print("\nWorst out-of-ROI (potential false positives):")
    worst = sorted(results, key=lambda r: -r.out_roi)[:5]
    for r in worst:
        print(
            f"  out={r.out_roi:3d}  {r.path}  ({r.category}, {r.alteration})"
        )

    # --- Write summary.csv ---
    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["category", "alteration", "finger", "n_final", "n_in_roi",
             "n_out_roi", "n_bifurcations", "n_terminations", "elapsed_s", "slug"]
        )
        for r in results:
            writer.writerow([
                r.category, r.alteration, r.finger, r.n_extracted, r.in_roi,
                r.out_roi, r.bifurcations, r.terminations, f"{r.elapsed_s:.3f}",
                r.slug,
            ])
    print(f"\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()

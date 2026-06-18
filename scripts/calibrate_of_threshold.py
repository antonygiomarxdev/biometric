"""Calibration: compute OF similarity matrix on N SOCOFing persons.

For each pair of enrolled fingerprints, compute the OF similarity
score. Produces:
- NxN similarity matrix (same-person diagonal vs cross-person off-diag)
- Score distribution stats
- Recommended threshold (95th percentile of cross-person scores)

Usage (from apps/backend):
    uv run python ../../scripts/calibrate_of_threshold.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from src.core.config import config
from src.core.interfaces import PipelineContext
from src.processing.enhancer import create_enhancer
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
from src.processing.scale_normalization import normalize_to_256

SOCOFING_REAL = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "backend"
    / "static"
    / "SOCOFing"
    / "Real"
)

N_PERSONS = 20


def find_socofing_file(person_external_id: str) -> Path:
    pid = person_external_id.replace("SOC_", "").lstrip("0")
    for path in sorted(SOCOFING_REAL.glob(f"{pid}__*_index_finger.BMP")):
        return path
    raise FileNotFoundError(f"No index BMP for {person_external_id}")


def compute_of(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Compute orientation field + coherence for one image.

    Returns (ori_field, coh_field) of shape (rows, cols), normalised
    to 256x256 pixels.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    enhancer = create_enhancer()
    enhanced = enhancer.enhance(img, resize=True)
    normalized = normalize_to_256(enhanced)
    ctx = PipelineContext(raw_image=normalized, fingerprint_id="calibration")
    ctx.preprocessed_image = normalized
    of_analyzer = OrientationFieldAnalyzer(block_size=16)
    of_analyzer.process(ctx)
    return of_analyzer.orientation_field, of_analyzer.coherence_field


def of_complex_vector(ori: np.ndarray, coh: np.ndarray) -> np.ndarray:
    """Build complex OF vector V = cos(2θ) + i·sin(2θ), masked by coherence."""
    valid = coh >= 0.35
    masked_ori = np.where(valid, ori, 0.0)
    v = np.cos(2 * masked_ori) + 1j * np.sin(2 * masked_ori)
    v *= valid
    return v


def of_similarity_rms(ori_a: np.ndarray, coh_a: np.ndarray,
                      ori_b: np.ndarray, coh_b: np.ndarray) -> float:
    """RMS difference of complex OF vectors, masked by coherence.

    Returns 0.0 (identical) to ~1.0 (perpendicular/orthogonal).
    """
    v_a = of_complex_vector(ori_a, coh_a)
    v_b = of_complex_vector(ori_b, coh_b)
    valid = (coh_a >= 0.35) & (coh_b >= 0.35)
    if valid.sum() == 0:
        return 1.0
    diff = np.abs(v_a[valid] - v_b[valid])
    return float(np.sqrt(np.mean(diff ** 2)))


def main() -> int:
    ext_ids = [f"SOC_{i:04d}" for i in range(100, 100 + N_PERSONS)]

    print(f"Loading {N_PERSONS} SOCOFing persons...")
    ofs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ext_id in ext_ids:
        try:
            img = find_socofing_file(ext_id).read_bytes()
        except FileNotFoundError:
            print(f"  {ext_id}: file not found, skipping")
            continue
        ori, coh = compute_of(img)
        ofs[ext_id] = (ori, coh)
        print(f"  {ext_id}: OF shape={ori.shape}, valid_blocks={int((coh >= 0.35).sum())}")

    if len(ofs) < 2:
        print("Need at least 2 persons")
        return 1

    persons = sorted(ofs.keys())
    n = len(persons)
    sim_matrix = np.zeros((n, n), dtype=np.float64)

    print(f"\nComputing {n}x{n} similarity matrix...")
    for i, pid_a in enumerate(persons):
        ori_a, coh_a = ofs[pid_a]
        for j, pid_b in enumerate(persons):
            if j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
            if i == j:
                sim_matrix[i, j] = 0.0
                continue
            ori_b, coh_b = ofs[pid_b]
            sim_matrix[i, j] = of_similarity_rms(ori_a, coh_a, ori_b, coh_b)

    # Stats: same-person is diagonal (all 0 by construction); cross-person
    # is the upper triangle.
    iu = np.triu_indices(n, k=1)
    cross_scores = sim_matrix[iu]
    same_scores = np.diag(sim_matrix)

    print("\n=== Cross-person OF similarity (RMS, lower = more similar) ===")
    print(f"  N pairs:        {len(cross_scores)}")
    print(f"  min:            {cross_scores.min():.4f}")
    print(f"  max:            {cross_scores.max():.4f}")
    print(f"  mean:           {cross_scores.mean():.4f}")
    print(f"  median:         {np.median(cross_scores):.4f}")
    print(f"  5th percentile: {np.percentile(cross_scores, 5):.4f}  (most similar pairs)")
    print(f"  95th percentile:{np.percentile(cross_scores, 95):.4f}  (least similar pairs)")

    # Show top-5 most-similar cross pairs
    flat_idx = np.argsort(cross_scores)
    print("\nTop-5 most-similar cross-person pairs (potential false negatives):")
    for k in flat_idx[:5]:
        i, j = iu[0][k], iu[1][k]
        print(f"  {persons[i]} <-> {persons[j]}: {cross_scores[k]:.4f}")

    print("\nTop-5 least-similar cross-person pairs (easiest to reject):")
    for k in flat_idx[-5:]:
        i, j = iu[0][k], iu[1][k]
        print(f"  {persons[i]} <-> {persons[j]}: {cross_scores[k]:.4f}")

    # Recommend threshold: 5th percentile of cross scores (most similar
    # cross-pair sets a lower bound; reject any pair above this).
    threshold_5pct = float(np.percentile(cross_scores, 5))
    threshold_1pct = float(np.percentile(cross_scores, 1))
    threshold_mean = float(cross_scores.mean())
    print("\n=== Recommended thresholds ===")
    print(f"  Aggressive (1st pct): {threshold_1pct:.4f} — high recall, some false positives")
    print(f"  Balanced (5th pct):   {threshold_5pct:.4f} — recommended for v1")
    print(f"  Conservative (mean):   {threshold_mean:.4f} — high precision")

    np.save("/tmp/of_similarity_matrix.npy", sim_matrix)
    print("\nSaved matrix to /tmp/of_similarity_matrix.npy")
    return 0


if __name__ == "__main__":
    sys.exit(main())

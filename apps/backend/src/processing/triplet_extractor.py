"""Triplet extraction from minutiae lists.

Each triplet of minutiae (mi, mj, mk) produces a 6-D feature vector
(3 distance ratios + 2 sin/cos angle deltas + 1 angle cosine) that is
fully invariant to translation, rotation, and scale.

The triplet descriptor is far more discriminative than the 5-D pair
descriptor because it captures the full geometric relationship among
three points.

Reference: NIST NBIS Bozorth3, Jain et al. (1997).
"""

from __future__ import annotations

import math

import numpy as np

from .minutia_quality import score_minutia

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
DEFAULT_QUALITY_THRESHOLD = 0.3
DEFAULT_MAX_RADIUS = 0.25  # fraction of normalised image size
DEFAULT_MAX_TRIPLETS = 200


def extract_triplets(
    minutiae: list[dict],
    skeleton: np.ndarray,
    normalized_shape: tuple[int, int],
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    max_radius: float = DEFAULT_MAX_RADIUS,
    max_triplets: int = DEFAULT_MAX_TRIPLETS,
) -> list[dict]:
    """Extract local triplets from quality-filtered minutiae.

    Each returned triplet has:
      - ``mi_idx``, ``mj_idx``, ``mk_idx``: indices into original minutiae
      - ``mi_x``, ``mi_y``, ``mi_angle`` (normalised)
      - ``mj_x``, ``mj_y``, ``mj_angle``
      - ``mk_x``, ``mk_y``, ``mk_angle``
      - ``d_ij``, ``d_ik``, ``d_jk``: pairwise Euclidean distances
      - ``type_triple``: encoded type int (0-26)
      - ``quality_min``, ``quality_avg``: quality stats of the three minutiae
      - ``vector``: 6-D descriptor (not stored separately; use *triplet_to_vector*)

    Triplets are capped to *max_triplets*, preferring higher-quality ones.
    """
    if len(minutiae) < 3:
        return []

    # 1. Score and filter by quality
    scored: list[tuple[int, dict, float]] = []
    for idx, m in enumerate(minutiae):
        q = score_minutia(m, skeleton, normalized_shape)
        if q >= quality_threshold:
            scored.append((idx, m, q))

    if len(scored) < 3:
        return []

    # 2. Build neighbour lists within radius
    r = max_radius
    indices = [(idx, m, q) for idx, m, q in scored]
    n = len(indices)

    candidates: list[tuple[float, dict]] = []

    for i in range(n):
        i_idx, im, iq = indices[i]
        ix, iy = float(im["x"]), float(im["y"])
        ia = float(im["angle"])

        i_neighbours: list[tuple[int, dict, float, float, float]] = []
        for j in range(i + 1, n):
            j_idx, jm, jq = indices[j]
            dx = float(jm["x"]) - ix
            dy = float(jm["y"]) - iy
            d = math.sqrt(dx * dx + dy * dy)
            if d <= r:
                i_neighbours.append((j_idx, jm, jq, d, math.atan2(dy, dx)))

        if len(i_neighbours) < 2:
            continue

        # For each pair (j, k) among i's neighbours
        for a in range(len(i_neighbours)):
            j_idx, jm, jq, d_ij, _ = i_neighbours[a]
            jx, jy = float(jm["x"]), float(jm["y"])
            ja = float(jm["angle"])

            for b in range(a + 1, len(i_neighbours)):
                k_idx, km, kq, d_ik, _ = i_neighbours[b]
                kx, ky = float(km["x"]), float(km["y"])
                ka = float(km["angle"])

                # Check d(j,k) constraint
                dx_jk = kx - jx
                dy_jk = ky - jy
                d_jk = math.sqrt(dx_jk * dx_jk + dy_jk * dy_jk)
                if d_jk > r:
                    continue

                # Skip degenerate triangles (any side near zero)
                if d_ij < 1e-8 or d_ik < 1e-8 or d_jk < 1e-8:
                    continue

                quality_min = min(iq, jq, kq)
                quality_avg = (iq + jq + kq) / 3.0

                triplet: dict = {
                    "mi_idx": i_idx,
                    "mj_idx": j_idx,
                    "mk_idx": k_idx,
                    "mi_x": ix,
                    "mi_y": iy,
                    "mi_angle": ia,
                    "mj_x": jx,
                    "mj_y": jy,
                    "mj_angle": ja,
                    "mk_x": kx,
                    "mk_y": ky,
                    "mk_angle": ka,
                    "d_ij": d_ij,
                    "d_ik": d_ik,
                    "d_jk": d_jk,
                    "type_triple": _encode_type_triple(
                        int(im.get("type", 2)),
                        int(jm.get("type", 2)),
                        int(km.get("type", 2)),
                    ),
                    "quality_min": quality_min,
                    "quality_avg": quality_avg,
                }
                candidates.append((quality_avg, triplet))

    # 3. Sort by quality (descending) and cap
    def _sort_key(item: tuple[float, dict]) -> float:
        return item[0]

    candidates.sort(key=_sort_key, reverse=True)
    selected = [t for _, t in candidates[:max_triplets]]

    return selected


def triplet_to_vector(t: dict) -> list[float]:
    """Convert a triplet dict to a 6-D feature vector for Qdrant.

    The vector components (all scale/rotation/translation invariant):

      r1 = d(i,j) / d(j,k)
      r2 = d(i,k) / d(j,k)
      r3 = d(i,j) / d(i,k)
      c1 = cos(theta_j - theta_i)
      s1 = sin(theta_j - theta_i)
      c2 = cos(theta_k - theta_j)

    The vector is L2-normalised so cosine similarity in Qdrant works.
    """
    d_ij = t["d_ij"]
    d_ik = t["d_ik"]
    d_jk = t["d_jk"]
    theta_i = t["mi_angle"]
    theta_j = t["mj_angle"]
    theta_k = t["mk_angle"]

    r1 = d_ij / d_jk
    r2 = d_ik / d_jk
    r3 = d_ij / d_ik
    c1 = math.cos(theta_j - theta_i)
    s1 = math.sin(theta_j - theta_i)
    c2 = math.cos(theta_k - theta_j)

    v = [r1, r2, r3, c1, s1, c2]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        return [0.0] * 6
    return [x / norm for x in v]


def _encode_type_triple(t1: int, t2: int, t3: int) -> int:
    """Encode three minutia types as a deterministic int (0-26).

    Each type maps: 1→0 (termination), 3→1 (bifurcation), else→2 (unknown).
    Encoded as base-3: t1*9 + t2*3 + t3.
    """
    def _norm(t: int) -> int:
        if t == 1:
            return 0
        if t == 3:
            return 1
        return 2

    return _norm(t1) * 9 + _norm(t2) * 3 + _norm(t3)

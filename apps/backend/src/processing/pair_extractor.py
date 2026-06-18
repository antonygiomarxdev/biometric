"""Pair extraction from minutiae list.

Each pair of minutiae (mi, mj) produces a 5-D feature vector
(dx_norm, dy_norm, sin(dtheta), cos(dtheta), distance_norm)
that is translation-invariant and approximately rotation/scale invariant.

Pairs are capped to avoid O(M²) combinatorial blowup for large M.
"""

from __future__ import annotations

import math


def extract_pairs(
    minutiae: list[dict],
    max_pairs: int = 500,
) -> list[dict]:
    """Enumerate all (mi, mj) pairs from *minutiae*.

    Each returned dict has:
      - ``i``, ``j``: indices into the original minutiae list
      - ``mi_x``, ``mi_y``, ``mi_angle``: first minutia (normalised)
      - ``mj_x``, ``mj_y``, ``mj_angle``: second minutia (normalised)
      - ``dx``: mj.x - mi.x
      - ``dy``: mj.y - mi.y
      - ``dtheta``: signed angle difference in [-pi, pi]
      - ``distance``: sqrt(dx² + dy²)
      - ``type_pair``: encodes (mi.type, mj.type) as a small int

    The output is capped to *max_pairs* by uniform random sampling
    when M*(M-1)/2 exceeds the limit.

    Coordinates are expected to be already normalised to [0, 1]
    (e.g., x / 256, y / 256).
    """
    m = len(minutiae)
    if m < 2:
        return []

    all_pairs: list[dict] = []
    for i in range(m):
        mi = minutiae[i]
        mix = float(mi["x"])
        miy = float(mi["y"])
        mi_angle = float(mi["angle"])
        mi_type = int(mi.get("type", 2))
        for j in range(i + 1, m):
            mj = minutiae[j]
            dx = float(mj["x"]) - mix
            dy = float(mj["y"]) - miy
            raw_dtheta = float(mj["angle"]) - mi_angle
            dtheta = _normalise_angle(raw_dtheta)
            distance = math.sqrt(dx * dx + dy * dy)
            all_pairs.append(
                {
                    "i": i,
                    "j": j,
                    "mi_x": mix,
                    "mi_y": miy,
                    "mi_angle": mi_angle,
                    "mj_x": float(mj["x"]),
                    "mj_y": float(mj["y"]),
                    "mj_angle": float(mj["angle"]),
                    "dx": dx,
                    "dy": dy,
                    "dtheta": dtheta,
                    "distance": distance,
                    "type_pair": _encode_type_pair(mi_type, int(mj.get("type", 2))),
                },
            )

    total = len(all_pairs)
    if total > max_pairs:
        # Uniform sampling — keep evenly spaced pairs across the enumeration
        step = total / max_pairs
        sampled = [all_pairs[int(round(i * step)) % total] for i in range(max_pairs)]
        return sampled

    return all_pairs


def _normalise_angle(theta: float) -> float:
    """Bring angle into [-pi, pi]."""
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def _encode_type_pair(t1: int, t2: int) -> int:
    """Deterministic small int from (type1, type2)."""
    if t1 > t2:
        t1, t2 = t2, t1
    return t1 * 10 + t2


def pair_to_vector(p: dict) -> list[float]:
    """Convert a pair dict to a 5-D feature vector for Qdrant.

    (dx, dy, sin(dtheta), cos(dtheta), distance)

    The vector is L2-normalised so cosine similarity in Qdrant works.
    """
    dx = p["dx"]
    dy = p["dy"]
    dtheta = p["dtheta"]
    dist = p["distance"]

    v = [dx, dy, math.sin(dtheta), math.cos(dtheta), dist]
    norm = math.sqrt(sum(x * x for x in v))
    if norm < 1e-10:
        return [0.0] * 5
    return [x / norm for x in v]

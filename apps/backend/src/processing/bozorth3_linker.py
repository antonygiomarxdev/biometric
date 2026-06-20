"""NBIS Bozorth3-style pair linking algorithm.

Bozorth3 is the NIST-standard minutiae pair matching algorithm,
originally developed by Alan Bozorth (FBI, 1993-95) and distributed
as part of NIST Biometric Image Software (NBIS).

References
----------
[1] Watson, C.I. et al. (2004). Studies of fingerprint matching using
    the NIST Verification Test Bed (VTB). NISTIR 7020. National
    Institute of Standards and Technology.
    https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7020.pdf

[2] Garris, M.D. et al. (2004). User's Guide to NIST Biometric Image
    Software (NBIS). NISTIR 7392. National Institute of Standards and
    Technology.
    https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7392.pdf

[3] Chapnick, P. et al. (2018). Forensic Latent Fingerprint
    Preprocessing Assessment. NIST IR 8215. National Institute of
    Standards and Technology.
    https://nvlpubs.nist.gov/nistpubs/ir/2018/NIST.IR.8215.pdf

[4] Cappelli, R., Ferrara, M., & Maltoni, D. (2010). Minutia
    Cylinder-Code: A New Representation and Matching Technique for
    Fingerprint Recognition. IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 32(12), 2128-2141.
    doi:10.1109/TPAMI.2010.52

Algorithm
---------
Given a set of probe minutiae pairs and candidate pair hits from
KNN search, Bozorth3:

1. Computes a rotation-only transformation (0, 0, dtheta) for each
   matched pair — invariant to translation and crop.

   ** CRITICAL — rotation-only transform for latent matching **
   The original Bozorth3 achieves rotation invariance by building an
   angular compatibility graph [1]. Translation invariance emerges
   from checking angular consistency across ALL pairs simultaneously,
   not by computing absolute spatial offsets. For latent fingerprints
   (partial impressions, random placement, distortion), using absolute
   (mi_x, mi_y) coordinates as the transformation reference is
   incorrect: a minutia at position 0.10 in a cropped latent vs 0.50
   in the enrolled image produces dx=0.40, which exceeds any
   reasonable dx_tol and prevents genuine pairs from grouping.

   This implementation therefore uses (0, 0, dtheta) as the
   transformation — only the angular component is tested for
   compatibility. Two pairs are compatible if they agree on the global
   rotation, regardless of where in the image they lie.

2. Links matched pairs whose transformations are compatible (within
   dtheta_tol), building a compatibility graph.

3. Finds the largest connected component via Union-Find. This
   represents the largest set of pairs that all agree on the same
   global rotation — equivalent to the best global alignment.

4. Applies a minimum component guard: the largest component must have
   at least ``min_component_size`` pairs (default 3) AND represent
   at least ``min_component_fraction`` of all available hits (default
   0.25). This prevents single-rotation-bin false-positives — groups
   of pairs from different fingers that coincidentally share a similar
   orientation offset — from generating misleading scores.

5. Scores the candidate by component size / saturation.

Tolerance calibration
---------------------
dtheta_tol = 0.20 rad (≈ 11.5°).

This value is deliberately tighter than the 0.35 rad used when the
transform included spatial components. Rationale:

- With rotation-only transforms ALL pairs share dx=dy=0, so the
  angular tolerance is the sole discriminator. A wide tolerance
  (0.35 rad) at rotation-only mode groups pairs from different fingers
  that share a similar rotation offset into spurious large components,
  producing false matches.

- NIST IR 8215 [3] reports latent orientation-field estimation error
  ±8–15°. The Crossing Number detector adds ±5–10° in low-quality
  zones. The value 0.20 rad (11.5°) covers the typical ±8° case with
  a small safety margin. Worst-case ±15° latents may need 0.28 rad,
  but that should only be unlocked after evaluating on your dataset.

- For rolled/slap impressions from controlled scanners, 0.10–0.15 rad
  (6–9°) is appropriate [1][2].

Score formula
-------------
The rank-based score uses a margin-normalised formula:

    margin = (n - best_fp) / max(n, 1)      ∈ [-1, 1]
    score  = (margin + 1) / 2               ∈ [ 0, 1]

This is more stable than n/(n+best_fp) when n ≈ best_fp, a common
situation for latents with few extractable minutiae. When n >> best_fp
(clear genuine match), both formulas converge to ~1.0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


class Bozorth3Linker:
    """NBIS-style Bozorth3 pair linker with latent-optimised parameters.

    Parameters
    ----------
    dx_tol:
        Translation tolerance in X (normalised coords). Retained for
        API compatibility; has no effect when ``_compute_transform``
        returns dx=0 (rotation-only mode). Default 0.02.
    dy_tol:
        Translation tolerance in Y. Same note as dx_tol. Default 0.02.
    dtheta_tol:
        Rotation tolerance in radians. Default 0.20 (≈ 11.5°), tightened
        from 0.35 because in rotation-only mode the angular tolerance is
        the sole discriminator — a wide value groups pairs from different
        fingers into spurious large components.
        For rolled/slap from controlled scanners: 0.10–0.15 rad.
        For worst-case latents (very low quality): up to 0.28 rad.
    saturation:
        Component size at which the absolute score reaches 1.0 (used
        only for single-candidate results). Default 30.
    min_component_size:
        Minimum number of pairs in the largest component for a result
        to be considered valid. Prevents single-pair components from
        generating misleading scores. Default 3.
    min_component_fraction:
        Minimum fraction of total available hits that the largest
        component must represent. Guards against trivial wins when very
        few hits are returned. Default 0.25.

    References
    ----------
    [1] Watson et al. NISTIR 7020 (2004).
    [3] Chapnick et al. NIST IR 8215 (2018).
    """

    def __init__(
        self,
        dx_tol: float = 0.02,
        dy_tol: float = 0.02,
        dtheta_tol: float = 0.20,
        saturation: int = 30,
        min_component_size: int = 3,
        min_component_fraction: float = 0.25,
    ) -> None:
        self._dx_tol = dx_tol
        self._dy_tol = dy_tol
        self._dtheta_tol = dtheta_tol
        self._saturation = saturation
        self._min_component_size = min_component_size
        self._min_component_fraction = min_component_fraction

    def link(
        self,
        probe_pairs: list[dict[str, Any]],
        candidate_hits: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Run Bozorth3 linking and return ranked candidates.

        Parameters
        ----------
        probe_pairs:
            List of probe pair dicts from :func:`extract_pairs`.
        candidate_hits:
            Raw KNN hits from ``knn_search_pairs``. Each hit has
            ``query_pair_index``, ``person_id``, and the candidate's
            pair geometry (``mi_x``, ``mi_y``, ``mi_angle``, etc.).
        top_k:
            Maximum number of candidates to return.

        Returns
        -------
        List of candidate dicts sorted by score descending, each with:
            - ``person_id``: str
            - ``score``: float (0-1)
            - ``validated_count``: int (size of largest component)
            - ``supporting_pairs``: list of hit dicts in the component
        """
        if not probe_pairs or not candidate_hits:
            return []

        per_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for hit in candidate_hits:
            per_person[hit["person_id"]].append(hit)

        results: list[dict[str, Any]] = []
        for person_id, hits in per_person.items():
            result = self._link_person(probe_pairs, hits)
            if result is not None:
                result["person_id"] = person_id
                results.append(result)

        if len(results) > 1:
            for r in results:
                n = r["validated_count"]
                best_fp = max(
                    (or_["validated_count"] for or_ in results if or_ is not r),
                    default=0,
                )
                margin = (n - best_fp) / max(n, 1)
                r["score"] = round((margin + 1) / 2, 4)
        elif len(results) == 1:
            n = results[0]["validated_count"]
            results[0]["score"] = round(min(1.0, n / max(self._saturation, 1)), 4)

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        return results[:top_k]

    def _link_person(
        self,
        probe_pairs: list[dict[str, Any]],
        person_hits: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Link hits for a single candidate person."""
        best_per_query: dict[int, dict[str, Any]] = {}
        for hit in person_hits:
            q_idx = hit["query_pair_index"]
            if q_idx not in best_per_query or hit["similarity"] > best_per_query[q_idx]["similarity"]:
                best_per_query[q_idx] = hit

        if not best_per_query:
            return None

        transforms: dict[int, tuple[float, float, float]] = {}
        for q_idx, hit in best_per_query.items():
            if q_idx >= len(probe_pairs):
                continue
            pp = probe_pairs[q_idx]
            transforms[q_idx] = _compute_transform(pp, hit)

        if not transforms:
            return None

        uf = _UnionFind()
        keys = sorted(transforms.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if _are_compatible(
                    transforms[ki], transforms[kj],
                    self._dx_tol, self._dy_tol, self._dtheta_tol,
                ):
                    uf.union(ki, kj)

        components: dict[int, list[int]] = defaultdict(list)
        for k in keys:
            root = uf.find(k)
            components[root].append(k)

        if not components:
            return None

        largest_key = max(components, key=lambda r: len(components[r]))
        largest = components[largest_key]
        n = len(largest)
        total_hits = len(best_per_query)

        # Guard: reject trivially small or fractionally insignificant components.
        # Eliminates false-positive components from pairs that coincidentally
        # share a rotation offset but belong to different fingers.
        if n < self._min_component_size:
            return None
        if total_hits > 0 and (n / total_hits) < self._min_component_fraction:
            return None

        score = min(1.0, n / max(self._saturation, 1))

        component_hits = [best_per_query[q_idx] for q_idx in largest]
        component_hits.sort(key=lambda h: float(h["similarity"]), reverse=True)

        return {
            "person_id": "",
            "score": round(score, 4),
            "validated_count": n,
            "supporting_pairs": component_hits,
        }


def _compute_transform(
    probe_pair: dict[str, Any],
    hit: dict[str, Any],
) -> tuple[float, float, float]:
    """Compute rigid transformation for a matched pair.

    Returns (0, 0, dtheta) — rotation-only form.
    See module docstring for full rationale.
    """
    dtheta = _normalise_angle(float(hit["mi_angle"]) - float(probe_pair["mi_angle"]))
    return (0.0, 0.0, dtheta)


def _are_compatible(
    t1: tuple[float, float, float],
    t2: tuple[float, float, float],
    dx_tol: float,
    dy_tol: float,
    dtheta_tol: float,
) -> bool:
    """Check if two transformations are compatible (within tolerance)."""
    return (
        abs(t1[0] - t2[0]) <= dx_tol
        and abs(t1[1] - t2[1]) <= dy_tol
        and abs(t1[2] - t2[2]) <= dtheta_tol
    )


def _normalise_angle(theta: float) -> float:
    """Bring angle into [-pi, pi]."""
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


class _UnionFind:
    """Simplified Union-Find for connected-component graph building."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            self._parent[rx] = ry
        elif self._rank[rx] > self._rank[ry]:
            self._parent[ry] = rx
        else:
            self._parent[ry] = rx
            self._rank[rx] += 1

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

1. Computes a rigid transformation (dx, dy, dtheta) for each matched
   pair — the spatial offset and rotation that maps the probe pair
   onto the gallery pair.

   The transformation is computed as:
     dx    = gallery_mi_x - probe_mi_x   (translation in X)
     dy    = gallery_mi_y - probe_mi_y   (translation in Y)
     dtheta = gallery_mi_angle - probe_mi_angle  (rotation)

   Two pairs are compatible if they agree on (dx, dy, dtheta) within
   tolerance — meaning they both support the SAME rigid alignment of
   probe onto gallery. This is the correct geometric interpretation:
   if two pairs from the latent map to two pairs in the enrolled image
   under the same transform, they are evidence of a genuine match.

   NOTE on translation invariance for latent matching
   ---------------------------------------------------
   Previous versions used a rotation-only transform (dx=0, dy=0) to
   handle the unknown placement of the latent on the finger. However,
   this discards all spatial information and allows pairs from entirely
   different spatial zones — including different people — to be grouped
   as compatible as long as their angle delta matches. The result is a
   high rate of false positives.

   The correct approach is to keep (dx, dy): two genuinely matching
   pairs WILL agree on the same translation because they both come from
   the same physical zone of the same finger. Pairs from different
   fingers or different zones will disagree on the translation even if
   they accidentally share a similar rotation.

   The dx_tol / dy_tol parameters (default 0.10 in normalised coords)
   provide enough slack for the elastic distortion typical of latent
   impressions without being so loose as to merge pairs from different
   fingers.

2. Links matched pairs whose transformations are compatible (within
   dx_tol, dy_tol, dtheta_tol), building a compatibility graph.

3. Finds the largest connected component via Union-Find. This
   represents the largest set of pairs that all agree on the same
   rigid alignment — equivalent to the best global alignment.

4. Applies a minimum component guard: the largest component must have
   at least ``min_component_size`` pairs (default 3) AND represent
   at least ``min_component_fraction`` of all available hits (default
   0.25). This prevents small coincidental clusters from generating
   misleading scores.

5. Scores the candidate by component size / saturation.

Tolerance calibration
---------------------
dx_tol = dy_tol = 0.10 (normalised coords, i.e. 10% of image width/height).
  Covers the typical elastic distortion in latent impressions while
  remaining tight enough to reject pairs from different spatial zones.

dtheta_tol = 0.20 rad (≈ 11.5°).
  Covers ±8° orientation-field estimation error (NIST IR 8215 [3])
  plus ±5° from the Crossing Number detector in low-quality zones.
  For rolled/slap impressions from controlled scanners: 0.10–0.15 rad.

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
    """NBIS-style Bozorth3 pair linker.

    Parameters
    ----------
    dx_tol:
        Translation tolerance in X (normalised coords). Two pairs are
        compatible only if their implied X-translations agree within this
        margin. Default 0.10 (10% of image width), covering elastic
        distortion in latent impressions.
    dy_tol:
        Translation tolerance in Y. Default 0.10.
    dtheta_tol:
        Rotation tolerance in radians. Default 0.20 (≈ 11.5°).
        For rolled/slap from controlled scanners: 0.10–0.15 rad.
        For worst-case latents (very low quality): up to 0.28 rad.
    saturation:
        Component size at which the absolute score reaches 1.0 (used
        only for single-candidate results). Default 30.
    min_component_size:
        Minimum number of pairs in the largest component for a result
        to be considered valid. Default 3.
    min_component_fraction:
        Minimum fraction of total available hits that the largest
        component must represent. Default 0.25.

    References
    ----------
    [1] Watson et al. NISTIR 7020 (2004).
    [3] Chapnick et al. NIST IR 8215 (2018).
    """

    def __init__(
        self,
        dx_tol: float = 0.10,
        dy_tol: float = 0.10,
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
    """Compute the rigid transformation (dx, dy, dtheta) for a matched pair.

    Given a probe pair (mi_probe, mj_probe) and a gallery hit (mi_gallery,
    mj_gallery), the transformation that maps the probe anchor minutia onto
    the gallery anchor minutia is:

        dx     = gallery_mi_x - probe_mi_x
        dy     = gallery_mi_y - probe_mi_y
        dtheta = gallery_mi_angle - probe_mi_angle

    Two pairs are "compatible" (support the same alignment) if they agree
    on (dx, dy, dtheta) within tolerance. This is the correct geometric
    test: genuinely matching pairs from the same finger zone will all imply
    the same rigid transform; pairs from different zones or different people
    will disagree on dx/dy even if they accidentally share a similar dtheta.
    """
    dx = float(hit.get("mi_x", 0.0)) - float(probe_pair["mi_x"])
    dy = float(hit.get("mi_y", 0.0)) - float(probe_pair["mi_y"])
    dtheta = _normalise_angle(float(hit["mi_angle"]) - float(probe_pair["mi_angle"]))
    return (dx, dy, dtheta)


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

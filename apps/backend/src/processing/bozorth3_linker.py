"""NBIS Bozorth3-style pair linking algorithm.

Bozorth3 is the NIST-standard minutiae pair matching algorithm.
Given a set of probe minutiae pairs and candidate pair hits from
KNN search, it:

1. Computes a rigid transformation (dx, dy, dtheta) for each
   matched pair (how the probe pair aligns to the candidate pair).
2. Links matched pairs whose transformations are compatible
   (within tolerance), building a graph.
3. Finds the largest connected component — this represents the
   best global alignment.
4. Scores the candidate by component size / saturation.

The NIST reference implementation (C) uses an edge-table approach.
This Python implementation uses Union-Find for cleaner readability
with equivalent semantics.
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
        Translation tolerance in X (normalised coords, default 0.02
        ≈ 5px at 256×256). Calibrated on SOCOFing Altered-Easy CR
        (5 subjects, 100% top-1 accuracy).
    dy_tol:
        Translation tolerance in Y (default 0.02).
    dtheta_tol:
        Rotation tolerance in radians (default 0.15 ≈ 8.6°).
    saturation:
        Component size at which score reaches 1.0. Default 30
        (calibrated; score = min(1.0, votes/30) gives 0.7+ for
        genuine matches on SOCOFing Altered-Easy CR).
    """

    def __init__(
        self,
        dx_tol: float = 0.02,
        dy_tol: float = 0.02,
        dtheta_tol: float = 0.15,
        saturation: int = 30,
    ) -> None:
        self._dx_tol = dx_tol
        self._dy_tol = dy_tol
        self._dtheta_tol = dtheta_tol
        self._saturation = saturation

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
            pair geometry (``mi_x``, ``mi_y``, etc.).
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

        results.sort(key=lambda r: float(r["score"]), reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Internal: per-person linking
    # ------------------------------------------------------------------

    def _link_person(
        self,
        probe_pairs: list[dict[str, Any]],
        person_hits: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Link hits for a single candidate person.

        Returns a result dict or ``None`` if no valid component found.
        """
        # Per-probe-pair best hit (highest similarity)
        best_per_query: dict[int, dict[str, Any]] = {}
        for hit in person_hits:
            q_idx = hit["query_pair_index"]
            if q_idx not in best_per_query or hit["similarity"] > best_per_query[q_idx]["similarity"]:
                best_per_query[q_idx] = hit

        if not best_per_query:
            return None

        # Compute transformation for each hit using the FIRST minutia (mi)
        transforms: dict[int, tuple[float, float, float]] = {}
        for q_idx, hit in best_per_query.items():
            if q_idx >= len(probe_pairs):
                continue
            pp = probe_pairs[q_idx]
            transforms[q_idx] = _compute_transform(pp, hit)

        if not transforms:
            return None

        # Build compatibility graph via Union-Find
        uf = _UnionFind()
        keys = sorted(transforms.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                ki, kj = keys[i], keys[j]
                if _are_compatible(transforms[ki], transforms[kj], self._dx_tol, self._dy_tol, self._dtheta_tol):
                    uf.union(ki, kj)

        # Find largest component
        components: dict[int, list[int]] = defaultdict(list)
        for k in keys:
            root = uf.find(k)
            components[root].append(k)

        if not components:
            return None

        largest_key = max(components, key=lambda r: len(components[r]))
        largest = components[largest_key]
        n = len(largest)

        score = min(1.0, n / max(self._saturation, 1))

        # Collect hits in the largest component, sorted by similarity desc
        component_hits = [best_per_query[q_idx] for q_idx in largest]
        component_hits.sort(key=lambda h: float(h["similarity"]), reverse=True)

        return {
            "person_id": "",  # filled by caller
            "score": round(score, 4),
            "validated_count": n,
            "supporting_pairs": component_hits,
        }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _compute_transform(
    probe_pair: dict[str, Any],
    hit: dict[str, Any],
) -> tuple[float, float, float]:
    """Compute rigid transformation (dx, dy, dtheta) for a matched pair.

    Uses the FIRST minutia (mi) of each pair as the reference point.
    All values are in normalised coordinates (0-1 range).
    """
    dx = float(hit["mi_x"]) - float(probe_pair["mi_x"])
    dy = float(hit["mi_y"]) - float(probe_pair["mi_y"])
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

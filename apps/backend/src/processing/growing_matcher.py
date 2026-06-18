"""Growing algorithm for triplet-based fingerprint matching.

Replaces Hough voting with the standard AFIS approach:
1. Pick the best triplet match (highest KNN similarity)
2. Compute 3-point alignment transformation
3. Validate by checking geometric consistency of other triplet matches
4. Grow: iteratively add consistent matches and refine the transform

Reference: NIST NBIS Bozorth3.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .triplet_alignment import AlignmentTransform
from .triplet_validator import TripletCorrespondence, TripletValidator, extract_correspondence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default validation tolerance (5% of normalised image = ~12px at 256x256)
VALIDATION_TOLERANCE: float = 0.05

# Minimum number of confirming triplet matches to accept a candidate
MIN_CONFIRMING_TRIPLETS: int = 3

# Maximum growing iterations before convergence
MAX_GROWING_ITERATIONS: int = 20

# Smoothing factor for score calculation (prevents division-by-one or
# spurious high scores from a single triplet match).  The formula
# ``score * (confirmed / (confirmed + SMOOTHING_OFFSET))`` ensures that
# a single confirmed triplet yields at most score * 0.5.
SMOOTHING_OFFSET: int = 1


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class GrowthResult:
    """Result of growing algorithm for one candidate person."""

    person_id: str
    transform: AlignmentTransform
    validated_count: int
    confirming_triplets: int
    total_probe_triplets: int
    score: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def grow_matches(
    probe_triplets: list[dict],
    knn_hits: list[dict],
    tolerance: float = VALIDATION_TOLERANCE,
    min_confirming: int = MIN_CONFIRMING_TRIPLETS,
    max_iterations: int = MAX_GROWING_ITERATIONS,
) -> list[GrowthResult]:
    """Run the growing algorithm for all candidate persons.

    Parameters
    ----------
    probe_triplets:
        List of probe triplet dicts from :func:`extract_triplets`.
    knn_hits:
        Raw hits from :meth:`knn_search_triplets`.  Each hit contains
        ``query_triplet_index``, ``similarity``, ``person_id``, and the
        candidate's triplet minutia positions (``mi_x``, ``mi_y``, etc.).
    tolerance:
        Fraction of image size for point alignment validation.
    min_confirming:
        Minimum confirming triplets to produce a result.
    max_iterations:
        Maximum refinement iterations.

    Returns
    -------
    List of ``GrowthResult`` sorted by score descending.
    """
    if not probe_triplets or not knn_hits:
        return []

    per_person: dict[str, list[dict]] = defaultdict(list)
    for hit in knn_hits:
        per_person[hit["person_id"]].append(hit)

    results: list[GrowthResult] = []
    num_probe = len(probe_triplets)

    for person_id, person_hits in per_person.items():
        result = _grow_person(
            person_id, person_hits, probe_triplets, num_probe,
            tolerance, min_confirming, max_iterations,
        )
        if result is not None:
            results.append(result)

    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Internal: single-person growing
# ---------------------------------------------------------------------------


def _grow_person(
    person_id: str,
    person_hits: list[dict],
    probe_triplets: list[dict],
    num_probe_triplets: int,
    tolerance: float,
    min_confirming: int,
    max_iterations: int,
) -> GrowthResult | None:
    """Run the growing algorithm for a single candidate person."""
    # Build per-query best-hit lookup
    def _sort_key(h: dict) -> float:
        return float(h["similarity"])

    person_hits.sort(key=_sort_key, reverse=True)

    best_per_query: dict[int, dict] = {}
    for hit in person_hits:
        q_idx = hit["query_triplet_index"]
        if q_idx not in best_per_query or hit["similarity"] > best_per_query[q_idx]["similarity"]:
            best_per_query[q_idx] = hit

    if not best_per_query:
        return None

    seed_hit = person_hits[0]
    seed_idx = seed_hit["query_triplet_index"]
    if seed_idx >= len(probe_triplets):
        return None

    # Build initial correspondence from the best seed hit
    seed_triplet = probe_triplets[seed_idx]
    seed_corr = extract_correspondence(seed_triplet, seed_hit, probe_triplet_index=seed_idx)

    current_transform = TripletValidator.compute_transform([seed_corr])
    confirmed_correspondences: list[TripletCorrespondence] = [seed_corr]

    # Build all candidate correspondences for this person
    all_correspondences: list[TripletCorrespondence] = []
    for q_idx, hit in best_per_query.items():
        if q_idx >= len(probe_triplets):
            continue
        pt = probe_triplets[q_idx]
        all_correspondences.append(extract_correspondence(pt, hit, probe_triplet_index=q_idx))

    # Growing loop: find new consistent matches, refine, repeat
    for _iteration in range(max_iterations):
        consistent = TripletValidator.filter_consistent(
            [c for c in all_correspondences if c not in confirmed_correspondences],
            current_transform, tolerance,
        )
        if not consistent:
            break

        confirmed_correspondences.extend(consistent)
        current_transform = TripletValidator.compute_transform(confirmed_correspondences)

    n_confirmed = len(confirmed_correspondences)
    if n_confirmed < min_confirming:
        return None

    similarity_mean = _similarity_mean(confirmed_correspondences)
    score = _compute_score(n_confirmed, num_probe_triplets, similarity_mean)

    return GrowthResult(
        person_id=person_id,
        transform=current_transform,
        validated_count=n_confirmed,
        confirming_triplets=n_confirmed,
        total_probe_triplets=num_probe_triplets,
        score=round(score, 4),
    )


def _similarity_mean(correspondences: list[TripletCorrespondence]) -> float:
    """Mean cosine similarity across all confirmed triplet correspondences.

    Each correspondence stores a ``candidate_hit`` dict with the KNN
    similarity score for that hit.  A match with low mean similarity
    (e.g. 0.4) is more likely a false positive even if many triplets
    are geometrically consistent.
    """
    if not correspondences:
        return 0.0
    sims = [float(c.candidate_hit.get("similarity", 0.0)) for c in correspondences]
    return sum(sims) / len(sims)


def _compute_score(
    n_confirmed: int,
    num_probe_triplets: int,
    similarity_mean: float,
) -> float:
    """Compute the confirmation score combining geometric ratio and similarity.

    The final score is::

        ratio × smooth × similarity_mean

    where ``ratio`` is the fraction of probe triplets confirmed and
    ``smooth`` is the linear dampener ``n / (n + SMOOTHING_OFFSET)``.

    Multiplying by ``similarity_mean`` ensures that confirmations built
    from low-similarity KNN hits cannot yield a high score — they may
    be geometrically consistent with the seed transform, but they are
    not confident matches at the descriptor level.
    """
    ratio = n_confirmed / max(num_probe_triplets, 1)
    smooth = n_confirmed / (n_confirmed + SMOOTHING_OFFSET)
    return min(ratio * smooth * similarity_mean, 1.0)

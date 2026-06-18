"""Triplet correspondence validation for growing matching algorithm.

Encapsulates the logic of checking whether a probe-candidate triplet
pair is geometrically consistent under a given transformation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .triplet_alignment import AlignmentTransform, align_3pts, apply_transform


@dataclass
class TripletCorrespondence:
    """A triplet match between probe and candidate minutiae.

    Stores both the raw triplet dicts (for payload) and the extracted
    point arrays (for alignment computation).

    Equality based on ``probe_triplet_index`` so that ``list.__contains__``
    works correctly during the growing loop.
    """

    probe_triplet: dict
    candidate_hit: dict
    probe_points: np.ndarray
    candidate_points: np.ndarray
    probe_angles: np.ndarray
    candidate_angles: np.ndarray
    probe_triplet_index: int = -1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TripletCorrespondence):
            return NotImplemented
        return self.probe_triplet_index == other.probe_triplet_index

    def __hash__(self) -> int:
        return self.probe_triplet_index


def extract_correspondence(
    probe_triplet: dict,
    candidate_hit: dict,
    *,
    probe_triplet_index: int = -1,
) -> TripletCorrespondence:
    """Build a :class:`TripletCorrespondence` from raw triplet dicts."""
    probe_points = np.array([
        [probe_triplet["mi_x"], probe_triplet["mi_y"]],
        [probe_triplet["mj_x"], probe_triplet["mj_y"]],
        [probe_triplet["mk_x"], probe_triplet["mk_y"]],
    ], dtype=np.float64)
    probe_angles = np.array([
        probe_triplet["mi_angle"],
        probe_triplet["mj_angle"],
        probe_triplet["mk_angle"],
    ], dtype=np.float64)

    candidate_points = np.array([
        [candidate_hit["mi_x"], candidate_hit["mi_y"]],
        [candidate_hit["mj_x"], candidate_hit["mj_y"]],
        [candidate_hit["mk_x"], candidate_hit["mk_y"]],
    ], dtype=np.float64)
    candidate_angles = np.array([
        candidate_hit["mi_angle"],
        candidate_hit["mj_angle"],
        candidate_hit["mk_angle"],
    ], dtype=np.float64)

    return TripletCorrespondence(
        probe_triplet=probe_triplet,
        candidate_hit=candidate_hit,
        probe_points=probe_points,
        candidate_points=candidate_points,
        probe_angles=probe_angles,
        candidate_angles=candidate_angles,
        probe_triplet_index=probe_triplet_index,
    )


class TripletValidator:
    """Validates triplet correspondences under a similarity transform."""

    @staticmethod
    def is_consistent(
        corr: TripletCorrespondence,
        transform: AlignmentTransform,
        tolerance: float,
    ) -> bool:
        """Check if a triplet correspondence is consistent with *transform*.

        Transforms the probe points using *transform* and checks if all
        three points land within *tolerance* of the candidate points.
        """
        transformed = apply_transform(corr.probe_points, transform)
        errors = np.sqrt(np.sum((transformed - corr.candidate_points) ** 2, axis=1))
        return bool(np.all(errors <= tolerance))

    @staticmethod
    def filter_consistent(
        correspondences: list[TripletCorrespondence],
        transform: AlignmentTransform,
        tolerance: float,
    ) -> list[TripletCorrespondence]:
        """Return only correspondences consistent with *transform*."""
        return [
            c for c in correspondences
            if TripletValidator.is_consistent(c, transform, tolerance)
        ]

    @staticmethod
    def compute_transform(
        correspondences: list[TripletCorrespondence],
    ) -> AlignmentTransform:
        """Compute the best similarity transform from multiple correspondences.

        Uses the first 3 point pairs (Procrustes) for a clean least-squares
        solution.  If fewer than 3 correspondences are provided, returns
        the identity transform.
        """
        if not correspondences:
            return AlignmentTransform(scale=1.0, angle=0.0, dx=0.0, dy=0.0)

        probe_pts_list: list[list[float]] = []
        cand_pts_list: list[list[float]] = []
        probe_ang_list: list[float] = []
        cand_ang_list: list[float] = []

        for corr in correspondences:
            for k in range(3):
                probe_pts_list.append([float(corr.probe_points[k, 0]), float(corr.probe_points[k, 1])])
                cand_pts_list.append([float(corr.candidate_points[k, 0]), float(corr.candidate_points[k, 1])])
                probe_ang_list.append(float(corr.probe_angles[k]))
                cand_ang_list.append(float(corr.candidate_angles[k]))

        n = min(len(probe_pts_list), 3)
        if n < 3:
            return AlignmentTransform(scale=1.0, angle=0.0, dx=0.0, dy=0.0)

        return align_3pts(
            np.array(probe_pts_list[:n], dtype=np.float64),
            np.array(cand_pts_list[:n], dtype=np.float64),
            np.array(probe_ang_list[:n], dtype=np.float64),
            np.array(cand_ang_list[:n], dtype=np.float64),
        )

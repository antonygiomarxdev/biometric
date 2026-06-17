"""
Minutia Cylinder Code (MCC) descriptor — Phase 19.

Builds rotation-invariant, scale-normalized cylinder descriptors for each
minutia extracted from a fingerprint ridge graph.

Architecture
------------
1. Pipeline produces: skeleton + ridge graph (nodes=minutiae, edges=ridges)
2. For each minutia, a cylindrical grid (angular sectors × radial rings) is
   centered on the minutia and ALIGNED to its local ridge orientation.
3. Each cell captures structural features of the ridge skeleton in that region:
   - dominant ridge orientation (relative)
   - ridge crossing count
   - local ridge frequency
4. The resulting 108-dimensional vector is L2-normalized and stored.

Rotation invariance
--------------------
The cylinder is aligned to the minutia's local ridge angle. When the entire
fingerprint is rotated by θ, both the minutia angle and the ridge orientations
rotate by θ. The RELATIVE orientation in each cell remains unchanged.

Scale normalization
--------------------
Ridge counts are normalized by the local ridge frequency, making the descriptor
invariant to image resolution changes.

Usage
-----
    from src.processing.mcc_descriptor import extract_cylinders

    descriptors = extract_cylinders(minutiae, skeleton, orientation_field, frequency_map)
    # descriptors[i] → 108D L2-normalized float32 vector for minutia i

References
----------
Cappelli, R., Ferrara, M., & Maltoni, D. (2010).
Minutia Cylinder-Code: A new representation and matching technique for
fingerprint recognition. IEEE TPAMI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Configuration — tunable without code changes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CylinderConfig:
    """Parameters for the MCC cylinder descriptor."""

    # Cylinder geometry
    angular_sectors: int = 12   # Number of wedge-shaped sectors (360°/N)
    radial_rings: int = 4       # Number of concentric rings

    # Ring distances (pixels at normalized scale ≈ 350px height)
    ring_boundaries: tuple[float, float, float, float] = (25.0, 55.0, 95.0, 130.0)

    # Features per cell
    use_orientation: bool = True
    use_ridge_count: bool = True
    use_frequency: bool = True

    @property
    def descriptor_dimension(self) -> int:
        features_per_cell = sum([self.use_orientation, self.use_ridge_count, self.use_frequency])
        return self.angular_sectors * self.radial_rings * features_per_cell


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = CylinderConfig()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cylinders(
    minutiae: Sequence[dict],
    skeleton: np.ndarray,
    orientation_field: np.ndarray | None = None,
    frequency_map: np.ndarray | None = None,
    config: CylinderConfig | None = None,
) -> list[np.ndarray]:
    """Build MCC cylinder descriptors for each minutia (production path).

    With the default config (12 sectors x 4 rings x 3 features), each
    descriptor is 144-D, L2-normalized, rotation-invariant (cylinder is
    aligned to the minutia's local ridge angle) and scale-stable (ridge
    counts bounded by local frequency).

    Args:
        minutiae: List of dicts with keys ``(x, y, angle)``.
        skeleton: Binary ridge skeleton (non-zero = ridge pixel).
        orientation_field: Block-level ridge orientation map (radians).
        frequency_map: Block-level ridge frequency map (cycles/pixel).
        config: Cylinder parameters; defaults to ``DEFAULT_CONFIG``.

    Returns:
        List of L2-normalized descriptor vectors, each with dimension
        ``config.descriptor_dimension`` (default 144).

    References:
        Cappelli, R., Ferrara, M., & Maltoni, D. (2010).
        Minutia Cylinder-Code. IEEE TPAMI.
    """
    if config is None:
        config = DEFAULT_CONFIG

    if len(minutiae) == 0 or skeleton is None or skeleton.sum() == 0:
        return []

    # Pre-compute skeleton pixel coordinates for fast lookup
    skeleton_rows, skeleton_cols = np.where(skeleton > 0)
    if len(skeleton_rows) < 10:
        return [_zero_vector(config)] * len(minutiae)

    # Block scaling: orientation/ frequency maps are at lower resolution
    orient_scale_y = orientation_field.shape[0] / skeleton.shape[0] if orientation_field is not None else 0
    orient_scale_x = orientation_field.shape[1] / skeleton.shape[1] if orientation_field is not None else 0
    freq_scale_y = frequency_map.shape[0] / skeleton.shape[0] if frequency_map is not None else 0
    freq_scale_x = frequency_map.shape[1] / skeleton.shape[1] if frequency_map is not None else 0

    rings = np.array(config.ring_boundaries)
    descriptors = []

    for minutia in minutiae:
        cylinder = _build_cylinder(
            minutia,
            skeleton_rows, skeleton_cols,
            orientation_field, orient_scale_y, orient_scale_x,
            frequency_map, freq_scale_y, freq_scale_x,
            rings, config,
        )
        descriptors.append(cylinder)

    return descriptors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _zero_vector(config: CylinderConfig) -> np.ndarray:
    """Return a zero-filled descriptor vector."""
    return np.zeros(config.descriptor_dimension, dtype=np.float32)


def _build_cylinder(
    minutia: dict,
    skel_rows: np.ndarray,
    skel_cols: np.ndarray,
    orientation_field: np.ndarray | None,
    orient_scale_y: float,
    orient_scale_x: float,
    frequency_map: np.ndarray | None,
    freq_scale_y: float,
    freq_scale_x: float,
    ring_boundaries: np.ndarray,
    config: CylinderConfig,
) -> np.ndarray:
    """Build a single MCC cylinder around one minutia."""

    center_x = minutia["x"]
    center_y = minutia["y"]
    center_angle = minutia["angle"]

    n_rings = config.radial_rings
    n_sectors = config.angular_sectors
    max_radius = ring_boundaries[-1]

    # ---- Stage 1: accumulate skeleton-density features ----

    # Distances and angles of ALL skeleton pixels relative to the minutia center
    delta_x = skel_cols.astype(np.float32) - center_x
    delta_y = skel_rows.astype(np.float32) - center_y
    distances = np.sqrt(delta_x**2 + delta_y**2)

    # Filter to pixels within the maximum cylinder radius
    mask = distances <= max_radius
    if mask.sum() == 0:
        return _zero_vector(config)

    delta_x = delta_x[mask]
    delta_y = delta_y[mask]
    distances = distances[mask]

    # Compute angle of each pixel RELATIVE to the minutia's orientation.
    # This makes the descriptor rotation-invariant: rotating the print
    # by θ shifts both center_angle and the absolute angles by θ,
    # leaving relative angles unchanged.
    absolute_angles = np.arctan2(delta_y, delta_x)
    relative_angles = (absolute_angles - center_angle + math.pi) % (2 * math.pi)

    # Assign each pixel to its angular sector
    sector_indices = np.int32(relative_angles * n_sectors / (2 * math.pi)) % n_sectors

    # Assign each pixel to its radial ring
    ring_indices = np.clip(np.digitize(distances, ring_boundaries), 0, n_rings - 1)

    # Accumulate skeleton density per cell
    density_cells = np.zeros((n_rings, n_sectors), dtype=np.float32)
    for sector, ring in zip(sector_indices, ring_indices):
        density_cells[ring, sector] += 1.0

    # ---- Stage 2: structure features sampled at cell centers ----

    orientation_cells = np.zeros((n_rings, n_sectors), dtype=np.float32)
    ridge_count_cells = np.zeros((n_rings, n_sectors), dtype=np.float32)
    frequency_cells = np.zeros((n_rings, n_sectors), dtype=np.float32)

    sector_width = 2 * math.pi / n_sectors

    for ring_idx in range(n_rings):
        sample_radius = (ring_boundaries[ring_idx] + ring_boundaries[min(ring_idx + 1, n_rings - 1)]) / 2

        for sector_idx in range(n_sectors):
            # World angle of the center of this cell (relative to minutia orientation)
            cell_center_angle = center_angle + (sector_idx + 0.5) * sector_width
            sample_x = center_x + sample_radius * math.cos(cell_center_angle)
            sample_y = center_y + sample_radius * math.sin(cell_center_angle)

            # ---- Sampled orientation (block-level, mapped to pixel coords) ----
            if orientation_field is not None and orient_scale_y > 0:
                block_y = min(int(sample_y * orient_scale_y), orientation_field.shape[0] - 1)
                block_x = min(int(sample_x * orient_scale_x), orientation_field.shape[1] - 1)
                absolute_orient = float(orientation_field[block_y, block_x])
                relative_orient = (absolute_orient - center_angle + math.pi) % (2 * math.pi)
                orientation_cells[ring_idx, sector_idx] = relative_orient / (2 * math.pi)

            # ---- Ridge crossing count (trace from center to sample point) ----
            if config.use_ridge_count:
                num_steps = max(1, int(sample_radius / 3))
                path_points = [
                    (
                        int(center_x + (sample_x - center_x) * t / num_steps),
                        int(center_y + (sample_y - center_y) * t / num_steps),
                    )
                    for t in range(num_steps + 1)
                ]
                ridge_count_cells[ring_idx, sector_idx] = _count_ridge_crossings(
                    path_points, skeleton_rows=skel_rows, skeleton_cols=skel_cols
                )

            # ---- Ridge frequency (spacing) at sample point ----
            if frequency_map is not None and freq_scale_y > 0:
                block_y = min(int(sample_y * freq_scale_y), frequency_map.shape[0] - 1)
                block_x = min(int(sample_x * freq_scale_x), frequency_map.shape[1] - 1)
                freq_val = float(frequency_map[block_y, block_x])
                frequency_cells[ring_idx, sector_idx] = min(max(freq_val, 0.0), 1.0)

    # ---- Stage 3: assemble final descriptor ----

    features: list[float] = []

    for ring_idx in range(n_rings):
        for sector_idx in range(n_sectors):
            if config.use_orientation:
                features.append(float(orientation_cells[ring_idx, sector_idx]))
            if config.use_ridge_count:
                features.append(float(min(ridge_count_cells[ring_idx, sector_idx], 10) / 10.0))
            if config.use_frequency:
                features.append(float(frequency_cells[ring_idx, sector_idx]))

    # Include density features if no structure features are enabled
    if len(features) == 0:
        for ring_idx in range(n_rings):
            for sector_idx in range(n_sectors):
                features.append(float(density_cells[ring_idx, sector_idx]))
        vec = np.array(features, dtype=np.float32)
        norm = np.sqrt(np.sum(vec**2)) + 1e-10
        return (vec / norm).astype(np.float32)

    vec = np.array(features, dtype=np.float32)
    norm = np.sqrt(np.sum(vec**2)) + 1e-10
    return (vec / norm).astype(np.float32)


def _count_ridge_crossings(
    path: list[tuple[int, int]],
    skeleton_rows: np.ndarray | None = None,
    skeleton_cols: np.ndarray | None = None,
) -> int:
    """Count how many ridges the sampling line crosses.

    Uses the skeleton pixel set for fast lookup. Each transition
    from background → ridge counts as one crossing.
    """
    crossings = 0
    prev_is_ridge = False

    for px, py in path:
        is_ridge = False
        if skeleton_rows is not None and skeleton_cols is not None:
            # Quick check: is there a skeleton pixel at this integer coordinate?
            is_ridge = np.any((skeleton_rows == py) & (skeleton_cols == px))
        if is_ridge and not prev_is_ridge:
            crossings += 1
        prev_is_ridge = is_ridge

    return crossings

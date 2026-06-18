"""
Tipos estrictos para el dominio biométrico.
Clean Code: Definiciones inmutables y explícitas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, ClassVar

import numpy as np

# Tipos primitivos para claridad
Radians = float
Degrees = float
Confidence = float  # 0.0 a 1.0

class MinutiaType(Enum):
    TERMINATION = 0
    BIFURCATION = 1
    UNKNOWN = 2

class AlgorithmOrigin(Enum):
    SKELETON = auto()
    GABOR = auto()
    ADAPTIVE = auto()
    CONSENSUS = auto()
    DEEP_LEARNING = auto()      # DL-based extraction (CNN / ViT)
    GAN_ENHANCED = auto()       # GAN-enhanced minutiae
    SEGMENTATION_AI = auto()    # AI segmentation (U-Net, etc.)

@dataclass(frozen=True, slots=True)
class MinutiaCandidate:
    """
    Representa una minucia candidata detectada por un algoritmo.
    Inmutable para garantizar thread-safety.
    """
    x: int
    y: int
    angle: float
    type: MinutiaType
    confidence: Confidence
    origin: AlgorithmOrigin

    def to_vector_part(self) -> list[float]:
        """Serializa la minucia para el vector final."""
        return [float(self.type.value), float(self.x), float(self.y), self.angle]

@dataclass(frozen=True)
class NormalizedFingerprint:
    """
    Huella normalizada lista para vectorización.
    Las coordenadas son relativas al centroide o sistema canónico.
    """
    id: str
    minutiae: list[MinutiaCandidate]
    width: int
    height: int
    image: np.ndarray | None = field(default=None, repr=False)
    ridge_graph: RidgeGraph | None = None

    @property
    def vector(self) -> np.ndarray:
        """Genera el vector plano normalizado."""
        # Se asume que self.minutiae ya está ordenado canónicamente
        data = []
        for m in self.minutiae:
            data.extend(m.to_vector_part())
        return np.asarray(data, dtype=np.float32)

# Alias para compatibilidad con código legacy
# En el futuro, migrar todo a NormalizedFingerprint
Fingerprint = NormalizedFingerprint
Minutiae = MinutiaCandidate

@dataclass
class MatchResult:
    """Resultado detallado de la comparación."""
    matched: bool
    person_id: str | None
    score: float  # Combined score
    confidence: float

    # Métricas detalladas
    l2_distance: float
    cosine_distance: float
    combined_score: float

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RidgeNode:
    x: int
    y: int
    weight: float = 1.0          # Forensic importance (1.0 = core, decays outwards)
    is_cutoff: bool = False      # True if it's an artificial boundary truncation
    angle: float = 0.0           # Local ridge orientation in radians [0, π)


@dataclass(frozen=True, slots=True)
class MccCylinder:
    """Minutia Cylinder-Code descriptor for a single minutia.

    A 3D cylinder capturing the spatio-directional relationship between
    a central minutia and its neighbours.  Invariant to rotation and
    translation (not scale — images expected at similar resolution).

    Dimensions:
        spatial: (C, C) grid spanning [-R, R] px (C = 2R//S + 1)
        directional: D bins covering [0, π)
    """
    values: np.ndarray          # shape: (spatial, spatial, dir_bins)

    @property
    def num_cells(self) -> int:
        return int(np.prod(self.values.shape))

    def cosine_similarity(self, other: MccCylinder) -> float:
        a = self.values.ravel()
        b = other.values.ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

@dataclass(frozen=True, slots=True)
class RidgeEdge:
    source: int
    target: int
    path: list[tuple[int, int]]
    length: int

@dataclass(frozen=True, slots=True)
class RidgeGraph:
    nodes: list[RidgeNode]
    edges: list[RidgeEdge]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def is_empty(self) -> bool:
        return self.num_nodes == 0


@dataclass(frozen=True, slots=True)
class GraphEmbedding:
    """
    Macro-topology features of a RidgeGraph, used as a dense vector
    for coarse matching via vector search.

    All features are size-invariant (ratios / normalised) so that two
    graphs with the same topology but different absolute scale produce
    identical embeddings.  The canonical 22-dim layout is::

        degree_0..degree_4plus    (5)   node-degree histogram ratios
        edge_len_p10..p90         (5)   edge length percentiles
        edge_len_mean, std        (2)
        weight_p10, p50, p90      (3)
        weight_mean, std          (2)
        log_nodes, log_edges      (2)
        cutoff_ratio              (1)
        avg_degree                (1)
        density                   (1)
        ----
        total                     (22)
    """
    # Degree histogram ratios (5 bins: 0, 1, 2, 3, 4+)
    degree_0_ratio: float
    degree_1_ratio: float
    degree_2_ratio: float
    degree_3_ratio: float
    degree_4plus_ratio: float
    # Edge length distribution (5 percentiles + mean + std)
    edge_len_p10: float
    edge_len_p25: float
    edge_len_p50: float
    edge_len_p75: float
    edge_len_p90: float
    edge_len_mean: float
    edge_len_std: float
    # Node weight distribution (3 percentiles + mean + std)
    weight_p10: float
    weight_p50: float
    weight_p90: float
    weight_mean: float
    weight_std: float
    # Graph-level stats
    log_num_nodes: float
    log_num_edges: float
    cutoff_ratio: float
    avg_degree: float
    density: float

    EMBEDDING_DIM: ClassVar[int] = 22

    def to_vector(self) -> np.ndarray:
        """Flatten to a fixed-size numpy vector for vector search."""
        return np.array(
            [
                self.degree_0_ratio,
                self.degree_1_ratio,
                self.degree_2_ratio,
                self.degree_3_ratio,
                self.degree_4plus_ratio,
                self.edge_len_p10,
                self.edge_len_p25,
                self.edge_len_p50,
                self.edge_len_p75,
                self.edge_len_p90,
                self.edge_len_mean,
                self.edge_len_std,
                self.weight_p10,
                self.weight_p50,
                self.weight_p90,
                self.weight_mean,
                self.weight_std,
                self.log_num_nodes,
                self.log_num_edges,
                self.cutoff_ratio,
                self.avg_degree,
                self.density,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True, slots=True)
class CoarseMatch:
    """A single candidate returned by the coarse matcher."""
    fingerprint_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MccCylinderHit:
    """A single cylinder-level hit from Qdrant KNN search.

    Returned per matched cylinder; aggregated into ``MccPersonHit``.
    ``query_cylinder_index`` is the probe's cylinder index (loop index
    from ``knn_search``, **not** a Qdrant payload field).
    """
    person_id: str
    fingerprint_id: str
    capture_id: str
    similarity: float  # cosine similarity in [0, 1]
    query_cylinder_index: int = 0  # index into the query_vectors list passed to knn_search
    candidate_x: int = 0
    candidate_y: int = 0
    candidate_angle: float = 0.0


@dataclass(frozen=True, slots=True)
class MccPersonHit:
    """Aggregated per-fingerprint match result.

    ``total_score`` is the sum of cosine similarities across all matching
    cylinders. When ``score_normalization == "fingerprint"`` (default), the
    caller divides by the number of enrolled cylinders to remove population
    bias.
    """
    person_id: str
    total_score: float
    hits: int
    contributing_fingerprints: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MccSearchHit:
    """A single ranked match candidate from MCC search (Phase 21 / Phase 23).

    ``match_trace`` is populated by :meth:`MccMatchingService.search`
    with per-cylinder (probe ↔ candidate) pairs for frontend overlay
    rendering.
    """
    person_id: str
    total_score: float
    hits: int
    contributing_fingerprints: list[str]
    match_trace: list[MatchTraceEntry] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class MinutiaSummary:
    """Lightweight minutia descriptor returned in search responses (Phase 23).

    Mirrors the public fields of :class:`MinutiaCandidate` without
    carrying the full forensic metadata (confidence, origin) — those
    are not needed by the UI for cylinder-level trace rendering.
    """
    x: int
    y: int
    angle: float
    type: int  # 0=termination, 1=bifurcation, 2=unknown (mirrors MinutiaType.value)


@dataclass(frozen=True, slots=True)
class MatchTraceEntry:
    """A single (probe_cylinder, candidate_cylinder) match pair (Phase 23).

    Surfaced in the ``match_trace`` list of each ``MccSearchHit`` so
    the frontend can render connecting lines between the two
    synchronized canvases (D-04/D-05/D-06).
    """
    probe_cylinder_index: int       # index into probe.minutiae list
    probe_x: int
    probe_y: int
    probe_angle: float
    candidate_capture_id: str
    candidate_fingerprint_id: str
    candidate_x: int
    candidate_y: int
    candidate_angle: float
    similarity: float               # cosine similarity in [0, 1]

"""
Tipos estrictos para el dominio biométrico.
Clean Code: Definiciones inmutables y explícitas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, List, NewType, Optional, Tuple
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

    def to_vector_part(self) -> List[float]:
        """Serializa la minucia para el vector final."""
        return [float(self.type.value), float(self.x), float(self.y), self.angle]

@dataclass(frozen=True)
class NormalizedFingerprint:
    """
    Huella normalizada lista para vectorización.
    Las coordenadas son relativas al centroide o sistema canónico.
    """
    id: str
    minutiae: List[MinutiaCandidate]
    width: int
    height: int
    image: Optional[np.ndarray] = field(default=None, repr=False)
    
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
    person_id: Optional[str]
    score: float  # Combined score
    confidence: float

    # Métricas detalladas
    l2_distance: float
    cosine_distance: float
    combined_score: float

    metadata: dict = field(default_factory=dict)


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
class TripletVector:
    """
    A local invariant structure (RAG chunk) representing a Delaunay triangle
    of three minutiae points.

    - `features`: Invariant vector (side lengths + angles + types) for KNN search.
    - `weight`: Forensic importance weight (0.0 to 1.0). Higher = closer to Core.
    """
    features: List[float]
    weight: float


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
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkHit:
    """A single chunk-level hit from Qdrant chunk search."""
    person_id: str
    fingerprint_id: str
    chunk_type: str
    weight: float
    similarity: float
    weighted_score: float


@dataclass(frozen=True, slots=True)
class PersonHit:
    """Aggregated person-level result from chunk search."""
    person_id: str
    total_score: float
    hits: int
    contributing_fingerprints: list[str] = field(default_factory=list)

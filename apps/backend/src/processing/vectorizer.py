"""Vectorización de minutiae mediante Triplets de Delaunay.

Para huellas latentes (escena del crimen), la vectorización tradicional
con coordenadas (x, y) es inútil porque un fragmento tiene coordenadas
distintas a la huella completa. Los Triplets de Delaunay son:

* **Invariantes a traslación**: los lados del triángulo miden igual
  sin importar dónde esté el dedo en la imagen.
* **Invariantes a rotación**: los ángulos interiores y las longitudes
  de los lados no cambian al rotar la huella.
* **Invariantes a datos parciales**: si un fragmento contiene 5
  minucias, formará un subconjunto de los triángulos que aparecen
  en la huella completa almacenada en la DB.

RAG Dactilar (Phase 10):
En lugar de producir un solo vector global, el `RagTripletVectorizer`
devuelve una lista de "chunks" (TripletVector), cada uno con un peso
basado en su distancia al Core de la huella. Esto permite que
`Qdrant` recupere los mejores K triángulos y luego agregamos los
puntajes ponderados por huella.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from src.core.interfaces import IVectorizer, PipelineContext
from src.core.metrics import timed
from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
    TripletVector,
)


class TripletVectorizer:
    """Convierte una lista de minucias en un vector de triplets.

    El vector resultante contiene los lados y ángulos de los triángulos
    de Delaunay ordenados canónicamente, hasta ``max_triangles``
    triángulos (9 valores cada uno). Ideal para Qdrant.
    """

    def __init__(self, max_triangles: int = 50) -> None:
        self.max_triangles = max_triangles

    @property
    def target_dim(self) -> int:
        """Dimensión total del vector (9 valores por triángulo)."""
        return self.max_triangles * 9

    @timed("vectorize_minutiae_triplets")
    def to_vector(self, minutiae: List[MinutiaCandidate]) -> np.ndarray:
        """Convierte minucias a vector de triplets invariantes.

        Args:
            minutiae: Lista de minucias ordenadas (cualquier orden).

        Returns:
            Vector float32 de dimensión ``target_dim``.
        """
        if len(minutiae) < 3:
            return np.zeros(self.target_dim, dtype=np.float32)

        points = np.array([[m.x, m.y] for m in minutiae], dtype=np.float64)

        # Delaunay triangulation
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points)
            simplices = tri.simplices
        except Exception:
            return np.zeros(self.target_dim, dtype=np.float32)

        features: List[float] = []

        for simplex in simplices:
            if len(simplex) < 3:
                continue
            i, j, k = simplex[:3]
            a, b, c = points[i], points[j], points[k]

            d12 = float(np.linalg.norm(a - b))
            d23 = float(np.linalg.norm(b - c))
            d31 = float(np.linalg.norm(c - a))

            if d12 < 1 or d23 < 1 or d31 < 1:
                continue

            sides = sorted([d12, d23, d31])
            angles = sorted(self._angles_from_sides(sides[0], sides[1], sides[2]))

            t_i = 1.0 if minutiae[i].type == MinutiaType.BIFURCATION else 0.0
            t_j = 1.0 if minutiae[j].type == MinutiaType.BIFURCATION else 0.0
            t_k = 1.0 if minutiae[k].type == MinutiaType.BIFURCATION else 0.0

            features.extend([sides[0], sides[1], sides[2],
                             angles[0], angles[1], angles[2],
                             t_i, t_j, t_k])

        if not features:
            return np.zeros(self.target_dim, dtype=np.float32)

        # Normalise side lengths by the average
        # (not by the max, so a small fragment with smaller sides still works)
        avg_side = float(np.mean([abs(features[i]) for i in range(0, len(features), 9)]))
        if avg_side > 0:
            for i in range(0, len(features), 9):
                features[i] /= avg_side
                features[i + 1] /= avg_side
                features[i + 2] /= avg_side

        vec = np.array(features, dtype=np.float32)
        if len(vec) >= self.target_dim:
            return vec[:self.target_dim]
        padded = np.zeros(self.target_dim, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded

    @staticmethod
    def _angles_from_sides(a: float, b: float, c: float) -> Tuple[float, float, float]:
        """Law of cosines."""
        if a + b <= c or b + c <= a or c + a <= b:
            # Degenerate triangle
            return math.pi / 3, math.pi / 3, math.pi / 3
        A = math.acos(max(-1, min(1, (b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))
        B = math.acos(max(-1, min(1, (a ** 2 + c ** 2 - b ** 2) / (2 * a * c))))
        C = math.pi - A - B
        return A, B, C

    @staticmethod
    def from_vector(vector: np.ndarray, expected_minutiae: int = 10) -> List[MinutiaCandidate]:
        """Placeholder: reconstructs a circle of artificial points."""
        pts: List[MinutiaCandidate] = []
        for i in range(expected_minutiae):
            angle = 2 * math.pi * i / expected_minutiae
            pts.append(MinutiaCandidate(
                x=int(50 * math.cos(angle)),
                y=int(50 * math.sin(angle)),
                angle=0.0,
                type=MinutiaType.BIFURCATION if i % 3 == 0 else MinutiaType.TERMINATION,
                confidence=1.0,
                origin=AlgorithmOrigin.SKELETON,
            ))
        return pts


class RagTripletVectorizer(IVectorizer):
    """
    Vectorizador RAG (Phase 10): produce una lista de chunks de
    tripletas, cada uno con su peso forense.

    Aplica el modelo experto: "los triángulos cerca del centro valen
    más que los del borde ruidoso". Usa decaimiento Gaussiano sobre
    la distancia del centroide del triángulo al Core detectado.
    """

    FEATURE_DIM: int = 9
    DEFAULT_SIGMA: float = 80.0

    def __init__(self, sigma: float = DEFAULT_SIGMA) -> None:
        self.sigma = sigma

    def vectorize(self, ctx: PipelineContext) -> List[TripletVector]:
        """Devuelve chunks de invariantes locales (RAG).

        Implementación Clean Code: delega a métodos privados pequeños.
        """
        if len(ctx.candidates) < 3:
            return []

        points, types = self._extract_points_and_types(ctx.candidates)
        triangles = self._delaunay_triangulate(points)
        if not triangles:
            return []

        features_per_triangle: List[List[float]] = []
        centroids_list: List[np.ndarray] = []
        for tri in triangles:
            pts_tri = points[list(tri)]
            types_tri = types[list(tri)]
            features_per_triangle.append(
                self._triangle_invariants(pts_tri, types_tri)
            )
            centroids_list.append(pts_tri.mean(axis=0))
        centroids = np.array(centroids_list, dtype=np.float64)
        weights = self._compute_weights(centroids, ctx.core)

        return [
            TripletVector(features=feat, weight=w)
            for feat, w in zip(features_per_triangle, weights)
        ]

    def chunks_from_minutiae(
        self,
        minutiae: list,
        core: tuple[int, int] | None = None,
    ) -> List[TripletVector]:
        """Public helper for services that already have a minutiae list.

        Used by ``RagMatchingService`` after running the pipeline.
        """
        if len(minutiae) < 3:
            return []
        from src.core.types import MinutiaCandidate, MinutiaType

        ctx = PipelineContext(
            raw_image=np.zeros((1, 1), dtype=np.uint8),
            candidates=list(minutiae),
            core=core,
        )
        return self.vectorize(ctx)

    def _chunks_from_normalized(
        self,
        normalized: NormalizedFingerprint,
    ) -> List[TripletVector]:
        """Build chunks from a fully processed fingerprint."""
        return self.chunks_from_minutiae(
            normalized.minutiae,
            core=None,
        )

    @staticmethod
    def _extract_points_and_types(
        candidates: List[MinutiaCandidate],
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = np.array([[c.x, c.y] for c in candidates], dtype=np.float64)
        types = np.array(
            [1.0 if c.type == MinutiaType.BIFURCATION else 0.0 for c in candidates],
            dtype=np.float64,
        )
        return points, types

    @staticmethod
    def _delaunay_triangulate(points: np.ndarray) -> List[Tuple[int, int, int]]:
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(points)
        except Exception:
            return []
        triangles: List[Tuple[int, int, int]] = []
        for simplex in tri.simplices:
            if len(simplex) != 3:
                continue
            i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
            a, b, c = points[i], points[j], points[k]
            d12 = float(np.linalg.norm(a - b))
            d23 = float(np.linalg.norm(b - c))
            d31 = float(np.linalg.norm(c - a))
            if min(d12, d23, d31) < 1.0:
                continue
            triangles.append((i, j, k))
        return triangles

    @classmethod
    def _triangle_invariants(
        cls,
        pts: np.ndarray,
        types: np.ndarray,
    ) -> List[float]:
        """Calcula features invariantes: 3 lados + 3 ángulos + 3 tipos."""
        d12 = float(np.linalg.norm(pts[0] - pts[1]))
        d23 = float(np.linalg.norm(pts[1] - pts[2]))
        d31 = float(np.linalg.norm(pts[2] - pts[0]))
        sides = sorted([d12, d23, d31])
        angles = sorted(cls._angles_from_sides(sides[0], sides[1], sides[2]))
        return [
            sides[0], sides[1], sides[2],
            angles[0], angles[1], angles[2],
            float(types[0]), float(types[1]), float(types[2]),
        ]

    @staticmethod
    def _angles_from_sides(a: float, b: float, c: float) -> Tuple[float, float, float]:
        if a + b <= c or b + c <= a or c + a <= b:
            return math.pi / 3, math.pi / 3, math.pi / 3
        A = math.acos(max(-1, min(1, (b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))
        B = math.acos(max(-1, min(1, (a ** 2 + c ** 2 - b ** 2) / (2 * a * c))))
        C = math.pi - A - B
        return A, B, C

    def _compute_weights(
        self,
        centroids: np.ndarray,
        core: tuple[int, int] | None,
    ) -> List[float]:
        """Aplica el peso experto. Si no hay core, fallback 1.0 uniforme."""
        if core is None:
            return [1.0] * len(centroids)
        core_pt = np.array(core, dtype=np.float64)
        distances = np.linalg.norm(centroids - core_pt, axis=1)
        weights = np.exp(-(distances ** 2) / (2.0 * self.sigma ** 2))
        return [float(w) for w in weights]

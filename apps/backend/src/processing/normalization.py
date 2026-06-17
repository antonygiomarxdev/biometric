"""
Normalización geométrica de minucias para huellas latentes y controladas.

Clean Architecture: Los normalizadores solo ordenan y deduplican. NO aplican
centrado global ni rotación PCA — ambas transformaciones son inválidas para
huellas parciales (escena del crimen) donde no existe un "centro del dedo" real.

Los downstream vectores en Qdrant se encargarán de la invarianza rotacional
y traslacional mediante triplets geométricos (Delaunay). Este módulo solo
garantiza que los puntos de entrada estén limpios, ordenados y sin duplicados.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.core.types import MinutiaCandidate, NormalizedFingerprint


class MinutiaNormalizer:
    """
    Normalizador para huellas latentes y controladas.

    - Deduplica puntos cercanos (consenso).
    - Ordena puntos canónicamente (distancia radial + ángulo desde el
      centro de la imagen, no el centro de masa).
    - NO centra globalmente. NO rota con PCA.
    """

    def __init__(self, consensus_distance: int = 5) -> None:
        self.consensus_distance = consensus_distance

    def normalize(
        self,
        minutiae: List[MinutiaCandidate],
        img_shape: Tuple[int, int],
    ) -> NormalizedFingerprint:
        if not minutiae:
            return NormalizedFingerprint(
                id="unknown", minutiae=[],
                width=img_shape[1], height=img_shape[0],
            )

        # 1. Eliminar duplicados cercanos (consenso)
        deduped = self._apply_consensus(minutiae)

        # 2. Ordenar canónicamente (determinista)
        sorted_m = self._canonical_sort(deduped, img_shape)

        return NormalizedFingerprint(
            id="unknown",
            minutiae=sorted_m,
            width=img_shape[1],
            height=img_shape[0],
        )

    def _apply_consensus(
        self,
        candidates: List[MinutiaCandidate],
    ) -> List[MinutiaCandidate]:
        """Fusiona candidatos muy cercanos (distancia euclidiana)."""
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda m: m.confidence, reverse=True)
        kept: List[MinutiaCandidate] = []

        for cand in ordered:
            is_dup = False
            for existing in kept:
                dx = cand.x - existing.x
                dy = cand.y - existing.y
                if dx * dx + dy * dy < self.consensus_distance ** 2:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)

        return kept

    def _canonical_sort(
        self,
        minutiae: List[MinutiaCandidate],
        img_shape: Tuple[int, int],
    ) -> List[MinutiaCandidate]:
        """Ordena por (radio desde el centro de la imagen, ángulo polar)."""
        cx = img_shape[1] / 2.0
        cy = img_shape[0] / 2.0

        def sort_key(m: MinutiaCandidate) -> tuple:
            r = (m.x - cx) ** 2 + (m.y - cy) ** 2
            return (r, m.y, m.x, m.angle)

        return sorted(minutiae, key=sort_key)

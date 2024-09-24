import math
from typing import List

import numpy as np

from src.fingerprint.domain.entities.minutiae import Minutiae


class FingerprintMinutiaeExtractorImpl:
    """Implementación del servicio de extracción de minucias."""

    def __init__(self, spurious_threshold: int = 10):
        self.spurious_threshold = (
            spurious_threshold  # Umbral para filtrar minucias espurias
        )

    def extract_minutiae(self, skeletonized_img: np.ndarray) -> List[Minutiae]:
        """Extraer las minucias de la imagen esqueletizada.

        Args:
            skeletonized_img (np.ndarray): Imagen esqueletizada de huella dactilar.

        Returns:
            List[Minutiae]: Lista de minucias detectadas.
        """
        minutiae = self._get_terminations_and_bifurcations(skeletonized_img)
        minutiae = self._filter_minutiae_by_distance(minutiae, skeletonized_img.shape)
        minutiae = self._remove_spurious_minutiae(minutiae)
        return minutiae

    def _get_terminations_and_bifurcations(self, img: np.ndarray) -> List[Minutiae]:
        """Detecta terminaciones y bifurcaciones en la imagen esqueletizada."""
        minutiae_list: List[Minutiae] = []
        rows, cols = img.shape

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if img[i, j] == 255:
                    block = img[i - 1 : i + 2, j - 1 : j + 2]
                    block_val = np.sum(block)

                    if block_val == 2 * 255:  # Terminación (1 vecino blanco)
                        orientation = self._calculate_orientation(block, "termination")
                        minutiae_list.append(
                            Minutiae("termination", (j, i), orientation)
                        )
                    elif block_val == 4 * 255:  # Bifurcación (3 vecinos blancos)
                        orientation = self._calculate_orientation(block, "bifurcation")
                        minutiae_list.append(
                            Minutiae("bifurcation", (j, i), orientation)
                        )

        return minutiae_list

    def _calculate_orientation(self, block: np.ndarray, minutiae_type: str) -> float:
        """Calcula la orientación de una minucia con base en los píxeles vecinos.

        Args:
            block (np.ndarray): Bloque 3x3 de la imagen.
            minutiae_type (str): Tipo de minucia ('termination' o 'bifurcation').

        Returns:
            float: Orientación de la minucia en grados.
        """
        rows, cols = block.shape
        center_x, center_y = (rows - 1) / 2, (cols - 1) / 2

        angles = []
        for i in range(rows):
            for j in range(cols):
                if (i == 0 or i == rows - 1 or j == 0 or j == cols - 1) and block[
                    i, j
                ] != 0:
                    angle = -math.degrees(math.atan2(i - center_y, j - center_x))
                    angles.append(angle)

        # Si es una terminación, debe haber solo un vecino; para bifurcación, debe haber tres vecinos.
        if minutiae_type == "termination" and len(angles) == 1:
            return angles[0]
        elif minutiae_type == "bifurcation" and len(angles) == 3:
            return sum(angles) / 3  # Promedio de los 3 ángulos
        return float("nan")  # Si no se puede calcular una orientación válida

    def _filter_minutiae_by_distance(
        self, minutiae: List[Minutiae], img_shape: tuple, border_margin: int = 10
    ) -> List[Minutiae]:
        """Filtrar las minucias que estén cerca de los bordes de la imagen.

        Args:
            minutiae (List[Minutiae]): Lista de minucias detectadas.
            img_shape (tuple): Tamaño de la imagen (alto, ancho).
            border_margin (int, optional): Margen desde el borde de la imagen. Defaults to 10.

        Returns:
            List[Minutiae]: Lista de minucias filtradas.
        """
        height, width = img_shape
        return [
            m
            for m in minutiae
            if border_margin <= m.position[0] <= width - border_margin
            and border_margin <= m.position[1] <= height - border_margin
        ]

    def _remove_spurious_minutiae(
        self, minutiae: List[Minutiae], distance_threshold: int = 15
    ) -> List[Minutiae]:
        """Eliminar minucias aisladas que no tengan otras cercanas.

        Args:
            minutiae (List[Minutiae]): Lista de minucias detectadas.
            distance_threshold (int, optional): Distancia mínima para considerar minucias cercanas. Defaults to 15.

        Returns:
            List[Minutiae]: Lista de minucias filtradas.
        """
        filtered_minutiae = []
        for m1 in minutiae:
            nearby_minutiae = [
                m2
                for m2 in minutiae
                if np.sqrt(
                    (m1.position[0] - m2.position[0]) ** 2
                    + (m1.position[1] - m2.position[1]) ** 2
                )
                < distance_threshold
            ]
            if (
                len(nearby_minutiae) > 1
            ):  # Si tiene más de una minucia cercana, no es aislada
                filtered_minutiae.append(m1)
        return filtered_minutiae

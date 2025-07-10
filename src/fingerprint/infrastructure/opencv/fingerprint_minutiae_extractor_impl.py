import math
from typing import List, Tuple

import numpy as np
from skimage.morphology import convex_hull_image, erosion, square

from src.fingerprint.domain.entities.minutiae import Minutiae


class FingerprintMinutiaeExtractorImpl:
    """Implementación del servicio de extracción de minucias."""

    def extract_minutiae(self, skeletonized_img: np.ndarray) -> List[Minutiae]:
        """Extraer las minucias de la imagen skeletonizada.

        Args:
            skeletonized_img (np.ndarray): Imagen de huella skeletonizada.

        Returns:
            List[Minutiae]: Lista de minucias detectadas.
        """
        # Detectar las minucias
        minutiae = self._detect_minutiae(skeletonized_img)

        # Aplicar filtrado de minucias cercanas al borde
        minutiae = self._filter_minutiae_by_distance(minutiae, skeletonized_img.shape)

        # Filtrar minucias utilizando la máscara de la región válida
        mask = self._create_convex_hull_mask(skeletonized_img)
        minutiae = self._filter_minutiae_using_mask(minutiae, mask)

        # Filtrar minucias en áreas ruidosas o cerca de bordes
        minutiae = self._filter_invalid_minutiae(minutiae, skeletonized_img)

        # Eliminar minucias aisladas
        minutiae = self._filter_isolated_minutiae(minutiae)

        return minutiae

    def _detect_minutiae(self, skeletonized_img: np.ndarray) -> List[Minutiae]:
        """Detecta minucias en la imagen skeletonizada."""
        minutiae = []
        skel = skeletonized_img == 255
        rows, cols = skel.shape
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                block = skel[i - 1 : i + 2, j - 1 : j + 2]
                block_sum = np.sum(block)
                if skel[i, j] == 1:
                    if block_sum == 2:  # Terminación
                        orientation = self._compute_orientation(block, "termination")
                        minutiae.append(Minutiae("termination", (j, i), orientation))
                    elif block_sum == 4:  # Bifurcación
                        orientation = self._compute_orientation(block, "bifurcation")
                        minutiae.append(Minutiae("bifurcation", (j, i), orientation))
        return minutiae

    def _compute_orientation(self, block: np.ndarray, minutiae_type: str) -> float:
        """Calcular la orientación de la minucia basada en el bloque 3x3 alrededor.

        Args:
            block (np.ndarray): Bloque 3x3 de la imagen.
            minutiae_type (str): Tipo de minucia ('termination' o 'bifurcation').

        Returns:
            float: Orientación de la minucia en grados.
        """
        center_x, center_y = 1, 1  # Centro del bloque 3x3
        angles = []

        for i in range(3):
            for j in range(3):
                if (i == 0 or i == 2 or j == 0 or j == 2) and block[i, j] != 0:
                    angle = -math.degrees(math.atan2(i - center_y, j - center_x))
                    angles.append(angle)

        if minutiae_type == "termination" and len(angles) == 1:
            return angles[0]
        elif minutiae_type == "bifurcation" and len(angles) == 3:
            return sum(angles) / 3
        return 0.0  # Orientación desconocida

    def _filter_minutiae_by_distance(
        self,
        minutiae: List[Minutiae],
        img_shape: Tuple[int, int],
        border_margin: int = 10,
    ) -> List[Minutiae]:
        """Filtrar las minucias que estén cerca de los bordes de la imagen.

        Args:
            minutiae (List[Minutiae]): Lista de minucias.
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

    def _create_convex_hull_mask(self, skeletonized_img: np.ndarray) -> np.ndarray:
        """Crear una máscara convexa de la región válida de la huella.

        Args:
            skeletonized_img (np.ndarray): Imagen skeletonizada.

        Returns:
            np.ndarray: Máscara convexa que representa la región de la huella.
        """
        mask = skeletonized_img > 0
        mask = convex_hull_image(mask)
        mask = erosion(mask, square(5))  # Erosionamos para mejorar la precisión
        return mask

    def _filter_minutiae_using_mask(
        self, minutiae: List[Minutiae], mask: np.ndarray
    ) -> List[Minutiae]:
        """Filtra las minucias que no están dentro de la región válida de la huella.

        Args:
            minutiae (List[Minutiae]): Lista de minucias detectadas.
            mask (np.ndarray): Máscara que define la región válida de la huella.

        Returns:
            List[Minutiae]: Lista de minucias dentro de la región válida.
        """
        valid_minutiae = [
            m for m in minutiae if mask[m.position[1], m.position[0]] != 0
        ]
        return valid_minutiae

    def _filter_invalid_minutiae(
        self, minutiae: List[Minutiae], skeletonized_img: np.ndarray
    ) -> List[Minutiae]:
        """Filtrar minucias en áreas de bordes ruidosos o en terminaciones de crestas no válidas.

        Args:
            minutiae (List[Minutiae]): Lista de minucias detectadas.
            skeletonized_img (np.ndarray): Imagen de huella skeletonizada.

        Returns:
            List[Minutiae]: Lista de minucias filtradas.
        """
        filtered_minutiae = []
        height, width = skeletonized_img.shape
        for m in minutiae:
            if self._is_valid_minutia(m, skeletonized_img, width, height):
                filtered_minutiae.append(m)
        return filtered_minutiae

    def _is_valid_minutia(
        self, minutia: Minutiae, skeletonized_img: np.ndarray, width: int, height: int
    ) -> bool:
        """Verificar si una minucia es válida. Evita falsos positivos en los bordes de la huella.

        Args:
            minutia (Minutiae): Minucia a validar.
            skeletonized_img (np.ndarray): Imagen skeletonizada.
            width (int): Ancho de la imagen.
            height (int): Alto de la imagen.

        Returns:
            bool: True si es válida, False si es un falso positivo.
        """
        x, y = minutia.position
        if x <= 5 or y <= 5 or x >= width - 5 or y >= height - 5:
            return False  # Filtra minucias muy cercanas a los bordes
        return True

    def _filter_isolated_minutiae(
        self, minutiae: List[Minutiae], distance_threshold: int = 15
    ) -> List[Minutiae]:
        """Eliminar minucias aisladas que no tengan otras cercanas.

        Args:
            minutiae (List[Minutiae]): Lista de minucias.
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

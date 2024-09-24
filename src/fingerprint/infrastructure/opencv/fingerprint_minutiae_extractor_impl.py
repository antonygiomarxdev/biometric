from typing import List

import numpy as np

from src.fingerprint.domain.entities.minutiae import Minutiae


class FingerprintMinutiaeExtractorImpl:
    """Implementación del servicio de extracción de minucias."""

    def extract_minutiae(self, skeletonized_img: np.ndarray) -> List[Minutiae]:
        """Extraer las minucias de la imagen skeletonizada.

        Args:
            skeletonized_img (np.ndarray): Imagen de huella skeletonizada.

        Returns:
            List[Minutia]: Lista de minucias detectadas.
        """
        minutiae = self._detect_minutiae(skeletonized_img)
        minutiae = self._filter_minutiae_by_distance(minutiae, skeletonized_img.shape)
        minutiae = self._filter_isolated_minutiae(minutiae)
        return minutiae

    def _detect_minutiae(self, skeletonized_img: np.ndarray) -> List[Minutiae]:
        """Detecta minucias en la imagen skeletonizada.

        Args:
            skeletonized_img (np.ndarray): Imagen skeletonizada.

        Returns:
            List[Minutiae]: Lista de minucias detectadas.
        """
        minutiae = []
        rows, cols = skeletonized_img.shape

        # Recorrer la imagen para detectar terminaciones y bifurcaciones
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if skeletonized_img[i, j] == 255:  # Si hay un píxel de cresta
                    block = skeletonized_img[i - 1 : i + 2, j - 1 : j + 2]
                    block_sum = np.sum(block)
                    if block_sum == 510:  # Terminación (solo un píxel vecino)
                        orientation = self._compute_orientation(block, "termination")
                        minutiae.append(
                            Minutiae(
                                type="termination",
                                position=(i, j),
                                orientation=orientation,
                            )
                        )
                    elif block_sum == 1020:  # Bifurcación (tres píxeles vecinos)
                        orientation = self._compute_orientation(block, "bifurcation")
                        minutiae.append(
                            Minutiae(
                                type="bifurcation",
                                position=(i, j),
                                orientation=orientation,
                            )
                        )
        return minutiae

    def _compute_orientation(self, block: np.ndarray, minutiae_type: str) -> float:
        """Calcular la orientación de la minucia basada en su tipo y la dirección de la cresta.

        Args:
            block (np.ndarray): Bloque alrededor de la minucia.
            minutiae_type (str): Tipo de minucia ('termination' o 'bifurcation').

        Returns:
            float: Orientación en grados.
        """
        (rows, cols) = block.shape
        center_x, center_y = (rows - 1) / 2, (cols - 1) / 2

        if minutiae_type == "termination":
            for i in range(rows):
                for j in range(cols):
                    if block[i, j] == 255 and (i != center_x or j != center_y):
                        return -np.degrees(np.arctan2(i - center_y, j - center_x))
        elif minutiae_type == "bifurcation":
            angles = []
            for i in range(rows):
                for j in range(cols):
                    if block[i, j] == 255 and (i != center_x or j != center_y):
                        angles.append(
                            -np.degrees(np.arctan2(i - center_y, j - center_x))
                        )
            return np.mean(angles)
        return 0.0  # Si no se detecta orientación

    def _filter_minutiae_by_distance(
        self, minutiae: List[Minutiae], img_shape: tuple, border_margin: int = 10
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

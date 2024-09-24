# infrastructure/opencv/fingerprint_minutiae_extractor_impl.py

from typing import List

import numpy as np

from src.fingerprint.application.services.minutiae_extractor_service import (
    MinutiaeExtractorService,
)
from src.fingerprint.domain.entities.minutiae import Minutiae


class FingerprintMinutiaeExtractorImpl(MinutiaeExtractorService):
    """Implementación del extractor de minucias utilizando OpenCV y Scikit-Image."""

    def extract_minutiae(self, skeleton_img: np.ndarray) -> List[Minutiae]:
        minutiae = []

        # Iterar sobre cada píxel de la imagen skeletonizada
        for i in range(1, skeleton_img.shape[0] - 1):
            for j in range(1, skeleton_img.shape[1] - 1):
                if skeleton_img[i, j] == 255:  # Píxel de cresta
                    neighbors = self.__get_neighbors(skeleton_img, i, j)
                    num_neighbors = sum(neighbors)

                    if num_neighbors == 1:
                        minutiae.append(Minutiae("termination", (i, j)))
                    elif num_neighbors == 3:
                        minutiae.append(Minutiae("bifurcation", (i, j)))

        return minutiae

    def __get_neighbors(self, img: np.ndarray, x: int, y: int) -> list:
        neighbors = [
            img[x - 1, y - 1] // 255,
            img[x - 1, y] // 255,
            img[x - 1, y + 1] // 255,
            img[x, y + 1] // 255,
            img[x + 1, y + 1] // 255,
            img[x + 1, y] // 255,
            img[x + 1, y - 1] // 255,
            img[x, y - 1] // 255,
        ]
        return neighbors

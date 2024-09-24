from abc import ABC, abstractmethod
from typing import List

import numpy as np

from src.fingerprint.domain.entities.minutiae import Minutiae


class MinutiaeExtractorService(ABC):
    """Interfaz para la extracción de minucias de huellas dactilares."""

    @abstractmethod
    def extract_minutiae(self, skeleton_img: np.ndarray) -> List[Minutiae]:
        """Extrae las minucias de una imagen skeletonizada.

        Args:
            skeleton_img (np.ndarray): Imagen de huella skeletonizada.

        Returns:
            List[Minutiae]: Lista de minucias encontradas.
        """
        pass

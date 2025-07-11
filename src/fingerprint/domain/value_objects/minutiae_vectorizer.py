import numpy as np
from typing import List

from ..entities.minutiae import Minutiae


class MinutiaeVectorizer:
    """Utility to convert minutiae lists into numeric vectors."""

    @staticmethod
    def to_vector(minutiae: List[Minutiae]) -> np.ndarray:
        """Convert a list of Minutiae into a 1D float32 numpy array."""
        data = []
        for m in minutiae:
            type_id = 0 if m.type == "termination" else 1
            data.extend([type_id, m.position[0], m.position[1], m.orientation])
        return np.asarray(data, dtype=np.float32)

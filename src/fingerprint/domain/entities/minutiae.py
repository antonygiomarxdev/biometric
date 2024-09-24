from typing import Tuple


class Minutiae:
    """Representa una minucia en una huella dactilar."""

    def __init__(self, type: str, position: Tuple[int, int]):
        self.type = type  # 'termination' o 'bifurcation'
        self.position = position

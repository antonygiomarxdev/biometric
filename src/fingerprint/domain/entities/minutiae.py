from typing import Tuple, Literal


class Minutiae:
    """Representa una minucia en una huella dactilar."""

    def __init__(
        self,
        type: Literal["termination", "bifurcation"],
        position: Tuple[int, int],
        orientation: float,
    ):
        self.type = type  # 'termination' o 'bifurcation'
        self.position = position  # (x, y)
        self.orientation = orientation  # orientación en grados

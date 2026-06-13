"""
Tipos estrictos para el dominio biométrico.
Clean Code: Definiciones inmutables y explícitas.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, NewType, Optional
import numpy as np

# Tipos primitivos para claridad
Radians = float
Degrees = float
Confidence = float  # 0.0 a 1.0

class MinutiaType(Enum):
    TERMINATION = 0
    BIFURCATION = 1
    UNKNOWN = 2

class AlgorithmOrigin(Enum):
    SKELETON = auto()
    GABOR = auto()
    ADAPTIVE = auto()
    CONSENSUS = auto()
    DEEP_LEARNING = auto()      # DL-based extraction (CNN / ViT)
    GAN_ENHANCED = auto()       # GAN-enhanced minutiae
    SEGMENTATION_AI = auto()    # AI segmentation (U-Net, etc.)

@dataclass(frozen=True, slots=True)
class MinutiaCandidate:
    """
    Representa una minucia candidata detectada por un algoritmo.
    Inmutable para garantizar thread-safety.
    """
    x: int
    y: int
    angle: float
    type: MinutiaType
    confidence: Confidence
    origin: AlgorithmOrigin

    def to_vector_part(self) -> List[float]:
        """Serializa la minucia para el vector final."""
        return [float(self.type.value), float(self.x), float(self.y), self.angle]

@dataclass(frozen=True)
class NormalizedFingerprint:
    """
    Huella normalizada lista para vectorización.
    Las coordenadas son relativas al centroide o sistema canónico.
    """
    id: str
    minutiae: List[MinutiaCandidate]
    width: int
    height: int
    image: Optional[np.ndarray] = field(default=None, repr=False)
    
    @property
    def vector(self) -> np.ndarray:
        """Genera el vector plano normalizado."""
        # Se asume que self.minutiae ya está ordenado canónicamente
        data = []
        for m in self.minutiae:
            data.extend(m.to_vector_part())
        return np.asarray(data, dtype=np.float32)

# Alias para compatibilidad con código legacy
# En el futuro, migrar todo a NormalizedFingerprint
Fingerprint = NormalizedFingerprint
Minutiae = MinutiaCandidate

@dataclass
class MatchResult:
    """Resultado detallado de la comparación."""
    matched: bool
    person_id: Optional[str]
    score: float  # Combined score
    confidence: float
    
    # Métricas detalladas
    l2_distance: float
    cosine_distance: float
    combined_score: float
    
    metadata: dict = field(default_factory=dict)

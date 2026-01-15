"""
Interfaces base para el pipeline biométrico.
Clean Code: Segregación de interfaces (ISP) e Inversión de dependencias (DIP).
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

from src.core.types import MinutiaCandidate, NormalizedFingerprint, MatchResult

class IEnhancer(ABC):
    """Protocolo para estrategias de mejora de imagen (CPU/GPU)."""
    
    @abstractmethod
    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """
        Mejora la calidad de la huella.
        Args:
            img: Imagen en escala de grises (uint8).
        Returns:
            Imagen binaria o mejorada.
        """
        pass

class IFeatureExtractor(ABC):
    """Protocolo para extracción de características."""
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> List[MinutiaCandidate]:
        """
        Extrae minucias de una imagen procesada.
        """
        pass

class INormalizer(ABC):
    """Protocolo para normalización y limpieza de minucias."""
    
    @abstractmethod
    def normalize(self, minutiae: List[MinutiaCandidate], img_shape: Tuple[int, int]) -> NormalizedFingerprint:
        """
        Ordena, filtra y normaliza coordenadas, retornando una huella normalizada.
        """
        pass

class IMatcher(ABC):
    """Protocolo para comparación de vectores."""
    
    @abstractmethod
    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        """
        Busca coincidencias en la base de datos.
        """
        pass
    
    @abstractmethod
    async def match_batch(self, probes: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """
        Busca coincidencias en lote.
        """
        pass

"""
Interfaces base para el pipeline biométrico.
Clean Architecture: Dependency Inversion via Structural Subtyping (Protocols).

Python moderno (3.8+) usa Protocols en lugar de ABCs (Abstract Base Classes).
Esto permite duck-typing seguro estáticamente: las implementaciones (ej. CpuEnhancer)
no necesitan heredar explícitamente de IEnhancer, solo necesitan implementar
la firma del método `enhance`. Esto desacopla totalmente las capas.
"""

from typing import Protocol, List, Tuple
import numpy as np

from src.core.types import MinutiaCandidate, NormalizedFingerprint, MatchResult


class IEnhancer(Protocol):
    """Protocolo para estrategias de mejora de imagen (CPU/IA)."""
    
    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """
        Mejora la calidad de la huella.
        Args:
            img: Imagen en escala de grises (uint8).
            resize: Si debe redimensionar la imagen a las dimensiones estándar.
        Returns:
            Imagen mejorada (generalmente binaria o normalizada).
        """
        ...


class IFeatureExtractor(Protocol):
    """Protocolo para extracción de características."""
    
    def extract(self, image: np.ndarray) -> List[MinutiaCandidate]:
        """
        Extrae minucias de una imagen procesada.
        Args:
            image: Imagen procesada.
        Returns:
            Lista de candidatos a minucias.
        """
        ...


class INormalizer(Protocol):
    """Protocolo para normalización y limpieza de minucias."""
    
    def normalize(self, minutiae: List[MinutiaCandidate], img_shape: Tuple[int, int]) -> NormalizedFingerprint:
        """
        Ordena, filtra y normaliza coordenadas.
        Args:
            minutiae: Lista de minucias crudas.
            img_shape: Dimensiones de la imagen original (alto, ancho).
        Returns:
            Estructura inmutable con minucias finales y orientaciones globales.
        """
        ...


class IMatcher(Protocol):
    """Protocolo para el motor de búsqueda biométrica."""
    
    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        """
        Busca coincidencias para un vector único.
        Args:
            probe: Vector de características a buscar.
            top_k: Número máximo de resultados a retornar.
        Returns:
            El mejor resultado encontrado.
        """
        ...
    
    async def match_batch(self, probes: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """
        Busca coincidencias para múltiples vectores en lote.
        Args:
            probes: Matriz de vectores.
            top_k: Resultados por vector.
        Returns:
            Lista de resultados correspondientes.
        """
        ...

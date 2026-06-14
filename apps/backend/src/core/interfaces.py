"""
Interfaces base para el pipeline biométrico.
Clean Architecture: Dependency Inversion via Structural Subtyping (Protocols).
"""

from typing import Protocol, List, Tuple, runtime_checkable
import numpy as np

from src.core.types import MinutiaCandidate, NormalizedFingerprint, MatchResult


class PreProcessResult:
    """Resultado de un pre-procesador: imagen procesada + máscara de calidad opcional."""
    def __init__(self, image: np.ndarray, quality_mask: np.ndarray | None = None) -> None:
        self.image = image
        self.quality_mask = quality_mask


@runtime_checkable
class IPreProcessor(Protocol):
    """Protocolo para hooks de pre-procesamiento (QualityMask, Binarization, etc.)."""
    def process(self, image: np.ndarray) -> PreProcessResult:
        """
        Procesa la imagen antes de la extracción.
        Args:
            image: Imagen en escala de grises (uint8).
        Returns:
            PreProcessResult con la imagen procesada y máscara de calidad opcional.
        """
        ...


@runtime_checkable
class IPostProcessor(Protocol):
    """Protocolo para hooks de post-procesamiento (filtrado topológico, fusión, etc.)."""
    def filter(
        self,
        candidates: List[MinutiaCandidate],
        quality_mask: np.ndarray | None = None,
    ) -> List[MinutiaCandidate]:
        """
        Filtra y limpia la lista de minucias después de la extracción.
        Args:
            candidates: Lista cruda de minucias detectadas.
            quality_mask: Máscara de calidad opcional (True = zona válida).
        Returns:
            Lista filtrada de minucias.
        """
        ...


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


@runtime_checkable
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

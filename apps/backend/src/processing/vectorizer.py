"""Vectorización de minutiae para comparación."""

import numpy as np
from typing import List

from src.core.types import MinutiaCandidate, MinutiaType
from src.core.metrics import timed


class MinutiaeVectorizer:
    """Convierte minutiae a vectores numéricos."""
    
    @staticmethod
    @timed("vectorize_minutiae")
    def to_vector(minutiae: List[MinutiaCandidate]) -> np.ndarray:
        """Convierte lista de minutiae a vector float32.
        
        Formato: [type, x, y, orientation, type, x, y, orientation, ...]
        
        Args:
            minutiae: Lista de minutiae
            
        Returns:
            Vector numpy de dimensión (4 * len(minutiae),)
        """
        data = []
        for m in minutiae:
            type_id = 0 if m.type == MinutiaType.TERMINATION else 1
            data.extend([
                type_id,
                float(m.x),
                float(m.y),
                m.angle
            ])
        return np.asarray(data, dtype=np.float32)
    
    @staticmethod
    def from_vector(vector: np.ndarray) -> List[MinutiaCandidate]:
        """Reconstruye minutiae desde un vector.
        
        Args:
            vector: Vector numpy
            
        Returns:
            Lista de minutiae
        """
        from src.core.types import AlgorithmOrigin
        minutiae = []
        for i in range(0, len(vector), 4):
            if i + 3 >= len(vector): break
            type_id = int(vector[i])
            m_type = MinutiaType.TERMINATION if type_id == 0 else MinutiaType.BIFURCATION
            x = int(vector[i + 1])
            y = int(vector[i + 2])
            angle = float(vector[i + 3])
            
            # Reconstrucción simplificada (confidence y origin por defecto)
            minutiae.append(MinutiaCandidate(
                 x=x, y=y, angle=angle, 
                 type=m_type, 
                 confidence=1.0, 
                 origin=AlgorithmOrigin.SKELETON
             ))
        return minutiae
    
    @staticmethod
    def pad_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Rellena un vector a una dimensión objetivo.
        
        Args:
            vector: Vector a rellenar
            target_dim: Dimensión objetivo
            
        Returns:
            Vector rellenado con ceros
        """
        if len(vector) >= target_dim:
            return vector[:target_dim]
        
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:len(vector)] = vector
        return padded

"""
Clase base y configuración para Enhancers.
"""
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from src.core.interfaces import IEnhancer

@dataclass
class EnhancerConfig:
    """Configuración inmutable del pipeline de mejora."""
    ridge_segment_blksze: int = 16
    ridge_segment_thresh: float = 0.1
    gradient_sigma: float = 1.0
    block_sigma: float = 7.0
    orient_smooth_sigma: float = 7.0
    ridge_freq_blksze: int = 38
    ridge_freq_windsze: int = 5
    min_wave_length: int = 5
    max_wave_length: int = 15
    relative_scale_factor_x: float = 0.65
    relative_scale_factor_y: float = 0.65
    angle_inc: int = 3
    ridge_filter_thresh: float = -3.0

class BaseEnhancer(IEnhancer, ABC):
    """Clase abstracta con lógica común."""
    
    def __init__(self, config: EnhancerConfig):
        self.config = config

    @abstractmethod
    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        pass

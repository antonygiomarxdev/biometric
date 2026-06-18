"""
Clase base y configuración para Enhancers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.core.config import config
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

    @classmethod
    def from_env(cls) -> "EnhancerConfig":
        """Build an :class:`EnhancerConfig` populated from
        :class:`src.core.config.EnhancerDefaultsConfig` (env-overridable).
        """
        ed = config.enhancer_defaults
        return cls(
            ridge_segment_blksze=ed.ridge_segment_blksze,
            ridge_segment_thresh=ed.ridge_segment_thresh,
            gradient_sigma=ed.gradient_sigma,
            block_sigma=ed.block_sigma,
            orient_smooth_sigma=ed.orient_smooth_sigma,
            ridge_freq_blksze=ed.ridge_freq_blksze,
            ridge_freq_windsze=ed.ridge_freq_windsze,
            min_wave_length=ed.min_wave_length,
            max_wave_length=ed.max_wave_length,
            relative_scale_factor_x=ed.relative_scale_factor_x,
            relative_scale_factor_y=ed.relative_scale_factor_y,
            angle_inc=ed.angle_inc,
        )

class BaseEnhancer(IEnhancer, ABC):
    """Clase abstracta con lógica común."""

    def __init__(self, config: EnhancerConfig):
        self.config = config

    @abstractmethod
    def enhance(self, img: np.ndarray, *, resize: bool = True) -> np.ndarray:
        pass

"""
Punto de entrada para el módulo de Enhancement.
Clean Code: Factory Pattern para selección transparente de estrategia.
"""
import logging
from typing import Optional

from src.core.interfaces import IEnhancer
from src.processing.enhancers.base import EnhancerConfig
from src.processing.enhancers.cpu import CpuEnhancer
from src.core.gpu_utils import GPUConfig

logger = logging.getLogger(__name__)

def create_enhancer(config: Optional[EnhancerConfig] = None) -> IEnhancer:
    """
    Crea la instancia de enhancer apropiada (GPU o CPU) según la configuración del sistema.
    """
    if config is None:
        config = EnhancerConfig()
        
    if GPUConfig.is_enabled():
        try:
            from src.processing.enhancers.gpu import GpuEnhancer
            logger.info("🚀 Inicializando GpuEnhancer")
            return GpuEnhancer(config)
        except ImportError as e:
            logger.warning(f"⚠️ Error importando GpuEnhancer, fallback a CPU: {e}")
            return CpuEnhancer(config)
        except Exception as e:
            logger.error(f"❌ Error inicializando GPU: {e}")
            return CpuEnhancer(config)
    
    logger.info("💻 Inicializando CpuEnhancer")
    return CpuEnhancer(config)


"""AI module: ONNX model management, GPU detection, and inference configuration."""

from .config import AiConfig
from .model_manager import ModelManager
from src.processing.enhancers.ai import SegmentationEnhancer, EnhancementEnhancer
from src.processing.extractor import AiFeatureExtractor

__all__ = [
    "ModelManager",
    "AiConfig",
    "SegmentationEnhancer",
    "EnhancementEnhancer",
    "AiFeatureExtractor",
]

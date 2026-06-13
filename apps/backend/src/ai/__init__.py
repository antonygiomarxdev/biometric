"""AI module: ONNX model management, GPU detection, and inference configuration."""

from .config import AiConfig
from .model_manager import ModelManager

__all__ = ["AiConfig", "ModelManager"]

"""ONNX Runtime model lifecycle manager.

Centralises session creation, caching, and GPU/CPU provider selection
so that AI components (segmentation, enhancement, extraction) share a
single point of model loading and do not duplicate sessions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.ai.config import AiConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ONNX Runtime inference sessions for all AI models.

    Designed as a singleton loaded during the FastAPI lifespan. Provides
    session caching, GPU/CPU provider fallback, and typed inference
    methods for each supported model kind.

    Usage::

        manager = ModelManager(AiConfig())
        mask = manager.run_segmentation(image_tensor)
    """

    def __init__(self, config: AiConfig) -> None:
        """Bind the manager to an :class:`AiConfig`.

        Args:
            config: AI configuration including model paths and provider.
        """
        self.config = config
        self._sessions: dict[str, ort.InferenceSession] = {}
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def load_model(self, name: str) -> ort.InferenceSession:
        """Load an ONNX model from disk, caching the session.

        Subsequent calls with the same *name* return the cached session.

        Args:
            name: Model basename (``.onnx`` suffix is appended).

        Returns:
            An ONNX Runtime inference session.

        Raises:
            FileNotFoundError: If the ``{name}.onnx`` file does not exist.
        """
        if name in self._sessions:
            return self._sessions[name]

        path = self.model_dir / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        session = ort.InferenceSession(
            str(path),
            providers=[self.config.provider, "CPUExecutionProvider"],
        )
        self._sessions[name] = session
        logger.info(
            "Loaded ONNX model '%s' (%s, %d inputs)",
            name,
            self.config.provider,
            len(session.get_inputs()),
        )
        return session

    def get_session(self, name: str) -> ort.InferenceSession:
        """Retrieve a cached session, loading it on first access.

        Args:
            name: Model basename.

        Returns:
            An ONNX Runtime inference session.
        """
        return self.load_model(name)

    def unload_model(self, name: str) -> None:
        """Evict a single model session from the cache."""
        self._sessions.pop(name, None)

    def unload_all(self) -> None:
        """Clear all cached sessions."""
        self._sessions.clear()

    @property
    def loaded_models(self) -> list[str]:
        """Names of models whose sessions are currently cached."""
        return list(self._sessions.keys())

    # ------------------------------------------------------------------
    # Typed inference helpers
    # ------------------------------------------------------------------

    def _run_single(self, name: str, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference for a single-input, single-output model.

        Args:
            name: Model basename.
            input_tensor: Pre-processed input tensor.

        Returns:
            Raw model output tensor.
        """
        session = self.get_session(name)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_tensor})
        return result[0]

    def run_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Run the segmentation model and return a binary mask.

        Args:
            image: Pre-processed input image tensor.

        Returns:
            Segmentation mask as a NumPy array.
        """
        return self._run_single("segment", image)

    def run_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Run the enhancement model and return the enhanced image.

        Args:
            image: Pre-processed input image tensor.

        Returns:
            Enhanced image as a NumPy array.
        """
        return self._run_single("enhance", image)

    def run_extraction(self, image: np.ndarray) -> np.ndarray:
        """Run the extraction model and return raw model output.

        Args:
            image: Pre-processed input image tensor.

        Returns:
            Raw model output as a NumPy array.
        """
        return self._run_single("extract", image)

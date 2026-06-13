"""Enhancer factory — AI-first, CPU fallback."""

from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np

from src.ai.model_manager import ModelManager
from src.core.interfaces import IEnhancer
from src.processing.enhancers.ai import EnhancementEnhancer, SegmentationEnhancer
from src.processing.enhancers.base import EnhancerConfig
from src.processing.enhancers.cpu import CpuEnhancer

logger = logging.getLogger(__name__)

EnhancerKind = Literal["cpu", "segmentation", "enhancement", "full_ai"]
"""Supported enhancer types. ``"full_ai"`` chains segmentation → enhancement."""


def create_enhancer(
    kind: str = "cpu",
    config: Optional[EnhancerConfig] = None,
    model_manager: Optional[ModelManager] = None,
) -> IEnhancer:
    """Create an :class:`IEnhancer` instance for the requested *kind*.

    Args:
        kind: One of ``"cpu"``, ``"segmentation"``, ``"enhancement"``, or
            ``"full_ai"``. Defaults to ``"cpu"``.
        config: Optional :class:`EnhancerConfig` for CPU enhancer.
        model_manager: Required for AI enhancer kinds. An initialised
            :class:`ModelManager` with the relevant ONNX models loaded.

    Returns:
        An :class:`IEnhancer` implementation.

    Raises:
        ValueError: If *kind* is unknown, or if an AI kind is requested
            without a *model_manager*.
    """
    # ── Valid kinds ────────────────────────────────────────────────────
    valid_ai_kinds = frozenset({"segmentation", "enhancement", "full_ai"})

    if kind == "cpu":
        logger.info("Creating CpuEnhancer (fallback)")
        return CpuEnhancer(config or EnhancerConfig())

    if kind not in valid_ai_kinds:
        raise ValueError(f"Unknown enhancer kind: {kind!r}")

    # AI kinds require a model_manager
    if model_manager is None:
        raise ValueError("model_manager is required for AI enhancer kinds")

    if kind == "segmentation":
        logger.info("Creating SegmentationEnhancer")
        return SegmentationEnhancer(model_manager)

    if kind == "enhancement":
        logger.info("Creating EnhancementEnhancer")
        return EnhancementEnhancer(model_manager)

    # kind == "full_ai"
    logger.info("Creating chained AI enhancer (segmentation → enhancement)")
    seg = SegmentationEnhancer(model_manager)
    enh = EnhancementEnhancer(model_manager)
    return _ChainedAiEnhancer(seg, enh)


class _ChainedAiEnhancer(IEnhancer):
    """Chains segmentation then enhancement for the full AI pipeline.

    First crops the fingerprint from the background via
    :class:`SegmentationEnhancer`, then cleans / reconstructs the ridge
    structure via :class:`EnhancementEnhancer`.
    """

    def __init__(
        self,
        segmentation: SegmentationEnhancer,
        enhancement: EnhancementEnhancer,
    ) -> None:
        """Initialise with the two sub-enhancers.

        Args:
            segmentation: The segmentation stage.
            enhancement: The enhancement stage.
        """
        self.segmentation = segmentation
        self.enhancement = enhancement

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """Run segmentation then enhancement in sequence.

        Args:
            img: Input grayscale image (H, W), dtype uint8.
            resize: Passed through to both sub-enhancers.

        Returns:
            The enhanced image after segmentation + enhancement.
        """
        cropped = self.segmentation.enhance(img, resize=resize)
        return self.enhancement.enhance(cropped, resize=resize)

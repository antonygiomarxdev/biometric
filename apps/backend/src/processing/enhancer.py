"""Enhancer factory — CPU-only (Phase 11 architecture).

The previous AI-first / CPU-fallback factory was removed with the
legacy ONNX pipeline.  Production enhancement is the Gabor-based
CPU implementation, which is the input to the ridge graph topology
extractor.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from src.core.interfaces import IEnhancer
from src.processing.enhancers.base import EnhancerConfig
from src.processing.enhancers.cpu import CpuEnhancer

logger = logging.getLogger(__name__)

EnhancerKind = Literal["cpu"]
"""The only supported enhancer kind is ``"cpu"``."""


def create_enhancer(
    config: Optional[EnhancerConfig] = None,
) -> IEnhancer:
    """Create a :class:`CpuEnhancer` instance.

    Args:
        config: Optional :class:`EnhancerConfig`; defaults to
            :meth:`EnhancerConfig.from_env` which reads from
            :class:`EnhancerDefaultsConfig` (env-overridable).

    Returns:
        A :class:`CpuEnhancer`.
    """
    return CpuEnhancer(config or EnhancerConfig.from_env())

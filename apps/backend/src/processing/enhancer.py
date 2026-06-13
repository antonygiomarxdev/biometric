"""Enhancer factory — always returns CpuEnhancer (GPU not available)."""

import logging
from typing import Optional

from src.core.interfaces import IEnhancer
from src.processing.enhancers.base import EnhancerConfig
from src.processing.enhancers.cpu import CpuEnhancer

logger = logging.getLogger(__name__)


def create_enhancer(config: Optional[EnhancerConfig] = None) -> IEnhancer:
    if config is None:
        config = EnhancerConfig()
    logger.info("Initializing CpuEnhancer")
    return CpuEnhancer(config)

"""AI-powered IEnhancer implementations for fingerprint processing.

Provides ONNX Runtime-based enhancers via ModelManager:

- **SegmentationEnhancer**: Crops the fingerprint from its background
  using a U-Net segmentation model.
- **EnhancementEnhancer** (in Task 2): Cleans / reconstructs a latent
  fingerprint using a U-Net MobileNetV2 enhancement model.

Architecture decision (D-03): U-Net with MobileNetV2 encoder was chosen
over a CNN autoencoder based on the Phase 2 spike evaluation. U-Net
achieves PSNR 21.69 / SSIM 0.9175 vs the CNN baseline of PSNR 6.97 /
SSIM 0.5068. See scripts/spike_findings.md for details.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ai.model_manager import ModelManager
from src.ai.segmentation import SegmentationProcessor
from src.core.interfaces import IEnhancer

logger = logging.getLogger(__name__)


class SegmentationEnhancer(IEnhancer):
    """U-Net based fingerprint segmentation + ROI crop.

    Applies the segmentation mask to isolate the fingerprint from the
    background, then crops to the bounding box of the mask region.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        processor: SegmentationProcessor | None = None,
    ) -> None:
        """Initialise the enhancer.

        Args:
            model_manager: Initialised :class:`ModelManager` with a loaded
                segmentation model.
            processor: Optional custom processor. A default
                :class:`SegmentationProcessor` is created if omitted.
        """
        self.model_manager = model_manager
        self.processor = processor or SegmentationProcessor()

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        """Segment the fingerprint and return the masked + cropped region.

        Args:
            img: Input grayscale image (H, W), dtype uint8.
            resize: Unused — kept for interface compatibility.

        Returns:
            Masked and cropped grayscale image (uint8). The output is the
            original image multiplied by the binary mask, then cropped to
            the mask's bounding rectangle.

        Raises:
            ValueError: If the segmentation produces an empty mask (no
                foreground pixels).
        """
        # Run full segmentation pipeline
        mask = self.processor.segment(img, self.model_manager)

        # Apply mask to original image
        masked = cv2.bitwise_and(img, img, mask=mask)

        # Find bounding box of the mask
        coords = cv2.findNonZero(mask)
        if coords is None or len(coords) == 0:
            raise ValueError(
                "Segmentation produced an empty mask — no fingerprint "
                "foreground detected."
            )

        x, y, w, h = cv2.boundingRect(coords)
        cropped = masked[y:y + h, x:x + w]
        logger.debug(
            "Segmentation crop: bounding box (%d, %d, %d, %d), output shape=%s",
            x, y, w, h,
            cropped.shape,
        )
        return cropped

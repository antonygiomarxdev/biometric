"""U-Net fingerprint segmentation inference logic.

Provides pre-processing and post-processing for the segmentation ONNX
model, wrapping the raw ModelManager output into a usable binary mask.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ai.config import AiConfig
from src.ai.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SegmentationProcessor:
    """Pre-process / post-process pipeline for the U-Net segmentation model.

    The segmentation model takes a 512×512 normalised grayscale image and
    produces a single-channel probability map. The processor handles
    padding, thresholding, and resizing back to the original dimensions.
    """

    def __init__(self, config: AiConfig | None = None) -> None:
        """Initialise with an optional *config* (defaults to :class:`AiConfig`).

        Args:
            config: AI configuration. Falls back to default if omitted.
        """
        self.config = config or AiConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Normalise a grayscale image and pad to the model's input size.

        Args:
            img: Input grayscale image (H, W), dtype uint8.

        Returns:
            Float32 tensor with shape (1, 1, input_size, input_size) and
            values in the [0, 1] range.
        """
        target = self.config.input_size
        # Normalise to [0, 1]
        normalized = img.astype(np.float32) / 255.0
        # Create a square canvas filled with zeros (black padding)
        h, w = normalized.shape
        canvas = np.zeros((target, target), dtype=np.float32)
        # Centre the image inside the canvas
        y_offset = (target - h) // 2
        x_offset = (target - w) // 2
        canvas[y_offset:y_offset + h, x_offset:x_offset + w] = normalized
        # Add batch and channel dimensions → (1, 1, H, W)
        return canvas[np.newaxis, np.newaxis, :, :]

    def postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        """Convert raw model output into a binary mask at the original size.

        Args:
            output: Raw ONNX output tensor (1, 1, H, W).
            original_shape: Desired output shape (height, width).

        Returns:
            Binary mask as uint8 (0 or 255) with shape *original_shape*.
        """
        # Squeeze batch and channel dimensions
        mask = output.squeeze()  # (H, W)
        # Threshold at 0.5 to produce a hard binary mask
        mask = (mask > self.config.confidence_threshold).astype(np.uint8) * 255
        # Resize back to the original input dimensions
        h, w = original_shape
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(np.uint8)

    def segment(
        self,
        image: np.ndarray,
        model_manager: ModelManager,
    ) -> np.ndarray:
        """Run the full segmentation pipeline: preprocess → inference → postprocess.

        Args:
            image: Input grayscale image (H, W), dtype uint8.
            model_manager: Initialised :class:`ModelManager` with a loaded
                segmentation model.

        Returns:
            Binary mask (uint8, 0 or 255) at the same spatial dimensions as
            the input *image*.
        """
        logger.debug(
            "Segmenting image shape=%s dtype=%s",
            image.shape,
            image.dtype,
        )
        tensor = self.preprocess(image)
        raw = model_manager.run_segmentation(tensor)
        mask = self.postprocess(raw, image.shape[:2])
        logger.debug(
            "Segmentation complete — mask shape=%s, foreground=%d/%d pixels",
            mask.shape,
            int(mask.sum() // 255),
            mask.size,
        )
        return mask

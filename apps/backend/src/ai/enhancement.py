"""U-Net MobileNetV2 fingerprint enhancement inference logic.

Provides pre-processing and post-processing for the enhancement ONNX
model, wrapping the raw ModelManager output into a cleaned / reconstructed
grayscale image.

Architecture (D-03 / spike_findings.md):
    U-Net with MobileNetV2 encoder, trained with L1+SSIM perceptual loss,
    512×512 input resolution, exported via ONNX opset 18.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.ai.config import AiConfig
from src.ai.model_manager import ModelManager

logger = logging.getLogger(__name__)


class EnhancementProcessor:
    """Pre-process / post-process pipeline for the U-Net enhancement model.

    The enhancement model takes a 512×512 normalised grayscale image and
    produces a cleaned / reconstructed fingerprint. The processor handles
    letterbox resizing, normalisation, and scaling back to uint8.
    """

    def __init__(self, config: AiConfig | None = None) -> None:
        """Initialise with an optional *config* (defaults to :class:`AiConfig`).

        Args:
            config: AI configuration. Falls back to default if omitted.
        """
        self.config = config or AiConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Normalise and letterbox-resize an image to the model input size.

        Preserves the original aspect ratio by padding (letterboxing) the
        shorter dimension so the model always receives a square input.

        Args:
            img: Input grayscale image (H, W), dtype uint8.

        Returns:
            Float32 tensor with shape (1, 1, input_size, input_size) and
            values in the [0, 1] range.
        """
        target = self.config.input_size
        h, w = img.shape

        # Resize the longer edge to target while preserving aspect ratio
        scale = target / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Letterbox pad to square
        canvas = np.zeros((target, target), dtype=np.float32)
        y_offset = (target - new_h) // 2
        x_offset = (target - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = normalized

        # Add batch and channel dimensions → (1, 1, H, W)
        return canvas[np.newaxis, np.newaxis, :, :]

    def postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int],
    ) -> np.ndarray:
        """Convert raw model output back to a uint8 image at the original size.

        Args:
            output: Raw ONNX output tensor (1, 1, H, W).
            original_shape: Desired output shape (height, width).

        Returns:
            Enhanced grayscale image as uint8 with shape *original_shape*.
        """
        # Squeeze batch and channel dimensions
        enhanced = output.squeeze()  # (H, W)
        # Clip to valid probability range
        enhanced = np.clip(enhanced, 0.0, 1.0)
        # Scale to [0, 255] uint8
        enhanced = (enhanced * 255.0).astype(np.uint8)
        # Resize back to the original input dimensions
        h, w = original_shape
        if enhanced.shape != (h, w):
            enhanced = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LINEAR)
        return enhanced

    def enhance(
        self,
        image: np.ndarray,
        model_manager: ModelManager,
    ) -> np.ndarray:
        """Run the full enhancement pipeline: preprocess → inference → postprocess.

        Args:
            image: Input grayscale image (H, W), dtype uint8.
            model_manager: Initialised :class:`ModelManager` with a loaded
                enhancement model.

        Returns:
            Enhanced grayscale image (uint8) at the same spatial dimensions
            as the input *image*.
        """
        logger.debug(
            "Enhancing image shape=%s dtype=%s",
            image.shape,
            image.dtype,
        )
        tensor = self.preprocess(image)
        raw = model_manager.run_enhancement(tensor)
        result = self.postprocess(raw, image.shape[:2])
        logger.debug(
            "Enhancement complete — output shape=%s, range=[%d, %d]",
            result.shape,
            int(result.min()),
            int(result.max()),
        )
        return result

"""Deep-learning fingerprint minutiae extraction inference.

Handles pre-processing and post-processing for a neural minutiae
detection model. The model output (multi-channel heatmap) is decoded
into :class:`MinutiaCandidate` instances with
:attr:`AlgorithmOrigin.DEEP_LEARNING`.

Architecture
------------
The extraction model takes a 512×512 normalised grayscale image and
produces a two-channel heatmap where each channel corresponds to a
minutia type (terminations, bifurcations). The processor handles
canvas padding, confidence-based non-maximum suppression, and
coordinate remapping.

Usage::

    processor = ExtractionProcessor(AiConfig())
    tensor = processor.preprocess(image)
    raw = model_manager.run_extraction(tensor)
    minutiae = processor.postprocess(raw, image.shape[:2])
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.ndimage import label, maximum_filter

from src.ai.config import AiConfig
from src.core.types import AlgorithmOrigin, MinutiaCandidate, MinutiaType

logger = logging.getLogger(__name__)


class ExtractionProcessor:
    """Pre / post processing pipeline for the DL minutiae extraction model.

    The model outputs a multi-channel heatmap. Each channel corresponds
    to a :class:`MinutiaType`. Local maxima above a confidence threshold
    are decoded into :class:`MinutiaCandidate` instances.
    """

    def __init__(self, config: AiConfig | None = None) -> None:
        """Initialise with an optional *config* (defaults to :class:`AiConfig`).

        Args:
            config: AI configuration. Falls back to default if omitted.
        """
        self.config = config or AiConfig()

    # ── Public API ──────────────────────────────────────────────────────────

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Normalise a grayscale image and pad it to the model's input size.

        Multi-channel images (BGR or single-channel 3D) are converted to
        grayscale first. The image is centred on a square canvas filled
        with zeros and the result is normalised to the [0, 1] range.

        Args:
            image: Input image (H, W), (H, W, 1), or (H, W, 3) with dtype
                uint8.

        Returns:
            Float32 tensor with shape ``(1, 1, input_size, input_size)``
            and values in [0, 1].
        """
        # Handle multi-channel inputs
        if image.ndim == 3:
            if image.shape[2] == 1:
                image = image[:, :, 0]
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        target = self.config.input_size
        # Normalise to [0, 1]
        normalised = image.astype(np.float32) / 255.0
        h, w = normalised.shape

        # Create a square canvas filled with zeros (black padding)
        canvas = np.zeros((target, target), dtype=np.float32)

        # Centre the image inside the canvas (crop if larger than target)
        h_content = min(h, target)
        w_content = min(w, target)
        y_offset = (target - h_content) // 2
        x_offset = (target - w_content) // 2
        canvas[y_offset:y_offset + h_content, x_offset:x_offset + w_content] = (
            normalised[:h_content, :w_content]
        )

        # Add batch and channel dimensions → (1, 1, H, W)
        return canvas[np.newaxis, np.newaxis, :, :]

    def postprocess(
        self,
        raw_output: np.ndarray,
        original_shape: tuple[int, int],
        confidence_threshold: float | None = None,
    ) -> list[MinutiaCandidate]:
        """Decode raw model output into a list of :class:`MinutiaCandidate`.

        Expected model output layout
        ----------------------------
        The output is a multi-channel heatmap where each (H, W) plane
        represents the confidence for a specific :class:`MinutiaType`:

        * Channel 0 — terminations
        * Channel 1 (optional) — bifurcations

        Local maxima above the confidence threshold are extracted using
        non-maximum suppression (3×3 neighbourhood) and mapped back to
        the original image coordinate space.

        Args:
            raw_output: Raw ONNX output tensor. Supported shapes:
                - ``(1, 1, H, W)`` — single-channel (general minutiae)
                - ``(1, 2, H, W)`` — two-channel (terminations + bifurcations)
                - ``(1, H, W)`` or ``(H, W)`` — also accepted.
            original_shape: ``(height, width)`` of the original image
                (before padding).
            confidence_threshold: Minimum confidence to keep a detection.
                Falls back to the config value when ``None``.

        Returns:
            List of :class:`MinutiaCandidate` instances with
            :attr:`AlgorithmOrigin.DEEP_LEARNING`.
        """
        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.config.confidence_threshold
        )

        # Squeeze leading dimensions to get a minimal spatial representation
        output = raw_output.squeeze()  # → (C, H, W) or (H, W)

        # Channel routing based on the squeezed shape
        if output.ndim == 2:
            # Single heatmap — treat as general / unknown minutiae
            candidates = self._decode_heatmap(
                output, threshold, MinutiaType.UNKNOWN
            )
        elif output.ndim == 3 and output.shape[0] >= 2:
            # Multi-channel: [terminations, bifurcations, …]
            terminations = self._decode_heatmap(
                output[0], threshold, MinutiaType.TERMINATION
            )
            bifurcations = self._decode_heatmap(
                output[1], threshold, MinutiaType.BIFURCATION
            )
            candidates = terminations + bifurcations
        elif output.ndim == 3 and output.shape[0] == 1:
            # Single channel with batch dim preserved — treat as unknown
            candidates = self._decode_heatmap(
                output[0], threshold, MinutiaType.UNKNOWN
            )
        else:
            logger.warning(
                "Unexpected model output shape: %s", raw_output.shape
            )
            return []

        # Remap coordinates from model output space → canvas space → original space
        h_orig, w_orig = original_shape
        size = self.config.input_size
        h_content = min(h_orig, size)
        w_content = min(w_orig, size)
        y_offset = (size - h_content) // 2
        x_offset = (size - w_content) // 2

        # The model output may be at a different spatial resolution than the
        # input canvas.  Compute scale factors to map output-space coordinates
        # into canvas-space coordinates.
        out_h, out_w = output.shape[-2:]
        scale_x = size / out_w
        scale_y = size / out_h

        adjusted: list[MinutiaCandidate] = []
        for m in candidates:
            # Scale from output space to canvas space
            x_canvas = m.x * scale_x
            y_canvas = m.y * scale_y
            # Subtract padding offset to get original image coordinates
            x_adj = int(round(x_canvas - x_offset))
            y_adj = int(round(y_canvas - y_offset))
            if 0 <= x_adj < w_orig and 0 <= y_adj < h_orig:
                adjusted.append(
                    MinutiaCandidate(
                        x=x_adj,
                        y=y_adj,
                        angle=m.angle,
                        type=m.type,
                        confidence=m.confidence,
                        origin=AlgorithmOrigin.DEEP_LEARNING,
                    )
                )

        logger.info(
            "DL extraction: %d candidates (terminations: %d, bifurcations: %d)",
            len(adjusted),
            sum(1 for m in adjusted if m.type == MinutiaType.TERMINATION),
            sum(1 for m in adjusted if m.type == MinutiaType.BIFURCATION),
        )
        return adjusted

    # ── Internal helpers ────────────────────────────────────────────────────

    def _decode_heatmap(
        self,
        heatmap: np.ndarray,
        threshold: float,
        mtype: MinutiaType,
    ) -> list[MinutiaCandidate]:
        """Extract local maxima from a heatmap above a confidence threshold.

        Non-maximum suppression (3×3 neighbourhood) is applied first,
        then connected-component labelling groups surviving pixels into
        individual minutiae. Sub-pixel accuracy is obtained via a
        weighted centre-of-mass within each group.

        Args:
            heatmap: 2D confidence map (H, W).
            threshold: Minimum confidence value.
            mtype: :class:`MinutiaType` to assign to extracted points.

        Returns:
            List of :class:`MinutiaCandidate` at the positions of
            detected local maxima.
        """
        # Non-maximum suppression: keep only local maxima in 3x3 window
        neighbourhood = 3
        local_max = heatmap == maximum_filter(heatmap, size=neighbourhood)
        detected = (heatmap > threshold) & local_max

        if not np.any(detected):
            return []

        # Label connected components
        labelled, num_features = label(detected)
        candidates: list[MinutiaCandidate] = []

        for feat_id in range(1, num_features + 1):
            ys, xs = np.where(labelled == feat_id)
            if len(xs) == 0:
                continue

            # Weighted centre of mass for sub-pixel accuracy
            region_vals = heatmap[ys, xs]
            total = region_vals.sum()
            if total == 0:
                continue

            cx = float(np.average(xs, weights=region_vals))
            cy = float(np.average(ys, weights=region_vals))
            confidence = float(region_vals.max())

            candidates.append(
                MinutiaCandidate(
                    x=int(round(cx)),
                    y=int(round(cy)),
                    angle=0.0,  # Computed from local ridge orientation post-hoc
                    type=mtype,
                    confidence=min(confidence, 1.0),
                    origin=AlgorithmOrigin.DEEP_LEARNING,
                )
            )

        return candidates

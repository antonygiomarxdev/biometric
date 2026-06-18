"""Scale normalization: resize fingerprint image to 256×256 with padding.

Preserves aspect ratio by adding black borders (letterbox/pillarbox)
so that any input size produces a comparable 256×256 output.
All downstream minutiae positions are normalised to (x/256, y/256).
"""

from __future__ import annotations

import cv2
import numpy as np

TARGET_SIZE = 256


def normalize_to_256(image: np.ndarray) -> np.ndarray:
    """Resize *image* to 256×256 while preserving aspect ratio.

    The input should be a 2-D grayscale array (uint8).  The output is a
    256×256 uint8 array with black (0) padding on the sides that do not
    fill the target square.

    Downscaling uses ``INTER_AREA`` (avoid aliasing), upscaling uses
    ``INTER_CUBIC`` (smooth edges).  Both are OpenCV conventions.
    """
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError(f"Cannot normalise empty image ({h}×{w})")

    scale = TARGET_SIZE / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    x_off = (TARGET_SIZE - new_w) // 2
    y_off = (TARGET_SIZE - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas

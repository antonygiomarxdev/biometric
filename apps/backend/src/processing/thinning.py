"""Thinning (skeletonisation) of binary fingerprint images.

Produces a single-pixel-wide skeleton using ``skimage.morphology.skeletonize``
(Zhang-Suen thinning).  The input should be a binary uint8 (0/255) or boolean
array; the output is always uint8 with values 0 or 1.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize


def thin(image: np.ndarray) -> np.ndarray:
    """Compute the single-pixel skeleton of a binary fingerprint image.

    Steps:
    1. Otsu binarisation if the image is not already binary.
    2. Morphological close (3×3) to fill small gaps.
    3. ``skimage.morphology.skeletonize`` (Zhang-Suen).

    Returns a uint8 array with 0 for background and 1 for skeleton.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_bool = binary > 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_bool = cv2.morphologyEx(
        binary_bool.astype(np.uint8), cv2.MORPH_CLOSE, kernel,
    ) > 0

    skel_bool = skeletonize(binary_bool)
    return skel_bool.astype(np.uint8)

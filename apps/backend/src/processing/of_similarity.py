"""Orientation Field similarity comparison.

Defines the OF similarity algorithm for the pre-filter in Phase 26.
Uses RMS of complex ``e^{2iθ}`` vectors, masked by coherence,
following the same calibration strategy from ``scripts/calibrate_of_threshold.py``.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.processing.pre_hooks import OrientationFieldAnalyzer


OF_SIMILARITY_THRESHOLD: float = 0.50
OF_COHERENCE_MIN: float = 0.35


class OFSimilarityError(ValueError):
    """Raised when OF inputs are invalid."""


class OFSimilarity:
    """Reference orientation field for one fingerprint.

    Stores the complex ``e^{2iθ}`` vector of the OF so that
    ``compare()`` can compute RMS against another probe OF.
    """

    def __init__(
        self,
        ori: np.ndarray,
        coh: np.ndarray,
        block_size: int = 16,
    ) -> None:
        if ori.shape != coh.shape:
            msg = f"ori shape {ori.shape} != coh shape {coh.shape}"
            raise OFSimilarityError(msg)
        if ori.ndim != 2:
            msg = f"expected 2D arrays, got {ori.ndim}D"
            raise OFSimilarityError(msg)
        if block_size <= 0:
            msg = f"block_size must be positive, got {block_size}"
            raise OFSimilarityError(msg)

        self._ori = ori.copy()
        self._coh = coh.copy()
        self._block_size = block_size
        self._vector = self._build_complex_vector(ori, coh)

    @staticmethod
    def _build_complex_vector(
        ori: np.ndarray,
        coh: np.ndarray,
    ) -> np.ndarray:
        valid = coh >= OF_COHERENCE_MIN
        masked_ori = np.where(valid, ori, 0.0)
        v = np.cos(2 * masked_ori) + 1j * np.sin(2 * masked_ori)
        v *= valid
        return v

    @property
    def ori(self) -> np.ndarray:
        return self._ori

    @property
    def coh(self) -> np.ndarray:
        return self._coh

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ori.shape

    @staticmethod
    def build(
        image_or_enhanced: np.ndarray,
        block_size: int = 16,
    ) -> OFSimilarity:
        """Build an OFSimilarity from an image.

        Runs ``OrientationFieldAnalyzer`` on the input (assumed to be
        already enhanced).
        """
        if image_or_enhanced.ndim not in (2, 3):
            msg = f"expected 2D or 3D image, got {image_or_enhanced.ndim}D"
            raise OFSimilarityError(msg)
        img = image_or_enhanced
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        from src.processing.scale_normalization import normalize_to_256

        normalized = normalize_to_256(img)
        analyzer = OrientationFieldAnalyzer(block_size=block_size)
        from src.core.interfaces import PipelineContext

        ctx = PipelineContext(raw_image=normalized, fingerprint_id="of_build")
        ctx.preprocessed_image = normalized
        analyzer.process(ctx)

        ori = analyzer.orientation_field
        coh = analyzer.coherence_field
        if ori is None or coh is None:
            msg = "OrientationFieldAnalyzer produced None fields"
            raise OFSimilarityError(msg)
        return OFSimilarity(ori, coh, block_size=block_size)

    def compare(self, other: OFSimilarity) -> float:
        """RMS difference against another ``OFSimilarity``.

        Returns 0.0 (identical OF) to ~1.4 (perpendicular).
        Only blocks where *both* coherence values exceed
        ``OF_COHERENCE_MIN`` contribute to the RMS.
        """
        return self.compare_raw(other._ori, other._coh)

    def compare_raw(
        self,
        other_ori: np.ndarray,
        other_coh: np.ndarray,
    ) -> float:
        """RMS difference against raw OF arrays.

        Useful for inline comparison without constructing a second
        ``OFSimilarity`` object.
        """
        if self._ori.shape != other_ori.shape:
            msg = (
                f"shape mismatch: self {self._ori.shape} != "
                f"other {other_ori.shape}"
            )
            raise OFSimilarityError(msg)

        v_other = self._build_complex_vector(other_ori, other_coh)
        valid = (self._coh >= OF_COHERENCE_MIN) & (other_coh >= OF_COHERENCE_MIN)
        if valid.sum() == 0:
            return 1.0

        diff = np.abs(self._vector[valid] - v_other[valid])
        return float(np.sqrt(np.mean(diff**2)))


def of_pseudo_core(coh: np.ndarray) -> tuple[int, int]:
    """Coherence-weighted centroid of the orientation field.

    Returns ``(row, col)`` as the pseudo-anchor when
    ``SingularityDetector`` does not find a Core.
    """
    if coh.ndim != 2:
        msg = f"expected 2D coherence array, got {coh.ndim}D"
        raise OFSimilarityError(msg)
    if coh.sum() == 0:
        return (coh.shape[0] // 2, coh.shape[1] // 2)

    rows = np.arange(coh.shape[0], dtype=np.float64)
    cols = np.arange(coh.shape[1], dtype=np.float64)
    r_center = float(np.average(rows, weights=coh.sum(axis=1)))
    c_center = float(np.average(cols, weights=coh.sum(axis=0)))
    return (int(round(r_center)), int(round(c_center)))

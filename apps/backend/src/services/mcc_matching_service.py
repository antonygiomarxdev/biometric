"""
MccMatchingService — NIST Bozorth3 pair matching (Phase 27).

The cylinder matcher (Phase 21) and triplet matcher (Phase 25) have been
removed. Pairs is the only matcher. See ``docs/adr/009-remove-cylinders.md``
for the decision.

Clean Architecture: application service. Orchestrates:
  * ``_run_quality_pipeline`` — enhance + thin + CN + filter + quality score
  * ``enroll_pairs`` — quality pipeline → pair extraction → Qdrant
  * ``search_by_pairs`` — quality pipeline → pair extraction → KNN → Bozorth3 linking

Algorithm (Bozorth3)
--------------------
For each probe pair, KNN search the ``pair_features`` collection. Bozorth3
linking groups geometrically compatible matches per candidate person, then
scores by the size of the largest connected component.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from concurrent.futures import Executor

import cv2
import numpy as np

from src.core.config import config
from src.processing.scale_normalization import TARGET_SIZE
from src.db.qdrant_pair_repository import QdrantPairRepository
from src.dev.logger import dev_log

logger = logging.getLogger(__name__)


class SearchByPairsResult(TypedDict):
    candidates: list[dict]
    probe_minutiae: list[dict]


class MccMatchingService:
    """NIST Bozorth3 pair matching service.

    Handles both enrollment and search. Constructor DI: pass ``pair_repo``
    in tests; defaults are constructed on first use.
    """

    def __init__(
        self,
        pair_repo: QdrantPairRepository | None = None,
        pool: Executor | None = None,
    ) -> None:
        self._pair_repo = pair_repo or QdrantPairRepository.from_host()
        self._pool = pool
        self._pair_repo.ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm_to_pixel_coords(
        norm_minutiae: list[dict[str, Any]],
        enhanced_shape: tuple[int, ...],
    ) -> list[dict[str, object]]:
        """Convert normalised minutiae (0-1) to pixel coords of *enhanced* image.

        Reverse of the ``scale_normalization.py`` transform so that
        visualisation can overlay minutiae on the original enhanced image.
        """
        h_enh, w_enh = enhanced_shape[:2]
        scale_px = TARGET_SIZE / max(h_enh, w_enh)
        new_w = int(round(w_enh * scale_px))
        new_h = int(round(h_enh * scale_px))
        x_off = (TARGET_SIZE - new_w) // 2
        y_off = (TARGET_SIZE - new_h) // 2
        result: list[dict[str, object]] = []
        for m in norm_minutiae:
            nx = float(m["x"]) * TARGET_SIZE
            ny = float(m["y"]) * TARGET_SIZE
            px = (nx - x_off) / scale_px
            py = (ny - y_off) / scale_px
            result.append({
                "x": int(round(px)),
                "y": int(round(py)),
                "angle": float(m["angle"]),
                "type": int(m.get("type", 2)),
            })
        return result

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            msg = "Failed to decode image bytes"
            raise ValueError(msg)
        return img

    def preview(self, image_bytes: bytes) -> dict[str, Any]:
        """Run the quality pipeline for preview purposes (no DB / no Qdrant I/O).

        Returns a dict with:
          - ``minutiae``: list of dicts with x, y, angle, type
          - ``enhanced_image``: the Gabor-filtered image as a numpy array
            (350px-tall, uint8). Caller serializes to base64 PNG.
        """
        result = self._run_quality_pipeline(image_bytes)
        minutiae: list[dict[str, Any]] = [
            {"x": int(m["x"]), "y": int(m["y"]), "angle": float(m["angle"]), "type": int(m["type"])}
            for m in result["minutiae"]
        ]
        return {"minutiae": minutiae, "enhanced_image": result["enhanced_image"]}

    def preview(self, image_bytes: bytes) -> dict[str, Any]:
        """Run the quality pipeline for live preview purposes.

        Returns a dict with:
          - ``minutiae``: list of dicts with x, y, angle, type
          - ``enhanced_image``: the Gabor-filtered image as a numpy array
            (350px-tall, uint8). Caller serializes to base64 PNG.
        """
        result = self._run_quality_pipeline(image_bytes)
        minutiae: list[dict[str, Any]] = [
            {"x": int(m["x"]), "y": int(m["y"]), "angle": float(m["angle"]), "type": int(m["type"])}
            for m in result["minutiae"]
        ]
        return {"minutiae": minutiae, "enhanced_image": result["enhanced_image"]}

    def preview_thinning(self, image_bytes: bytes) -> dict[str, Any]:
        """Thinning + Crossing Number pipeline for preview.

        Scale-normalises to 256×256, enhances (Gabor), thins (Zhang-Suen),
        detects minutiae via Crossing Number, and filters false ones.
        """
        import time as _time

        from src.processing.crossing_number import extract_minutiae_cn
        from src.processing.false_minutiae_filter import filter_false_minutiae
        from src.processing.scale_normalization import normalize_to_256
        from src.processing.thinning import thin

        t0 = _time.monotonic()
        image = self._decode(image_bytes)
        t_decode = _time.monotonic()

        from src.processing.enhancer import create_enhancer

        enhancer = create_enhancer()
        enhanced = enhancer.enhance(image, resize=True)
        t_enhance = _time.monotonic()

        normalized = normalize_to_256(enhanced)
        t_norm = _time.monotonic()

        skeleton = thin(normalized)
        t_thin = _time.monotonic()

        raw_minutiae = extract_minutiae_cn(skeleton)
        t_cn = _time.monotonic()

        filtered = filter_false_minutiae(raw_minutiae, normalized.shape)
        t_filter = _time.monotonic()

        h, w = normalized.shape[:2]
        normalised: list[dict[str, Any]] = [
            {
                "x": float(m["x"]) / w,
                "y": float(m["y"]) / h,
                "angle": float(m["angle"]),
                "type": int(m["type"]),
            }
            for m in filtered
        ]

        dev_log(
            "mcc.preview_thinning",
            image_bytes=len(image_bytes),
            raw_minutiae=len(raw_minutiae),
            filtered=len(filtered),
            normalised=len(normalised),
            decode_ms=round((t_decode - t0) * 1000, 1),
            norm_ms=round((t_norm - t_decode) * 1000, 1),
            enhance_ms=round((t_enhance - t_norm) * 1000, 1),
            thin_ms=round((t_thin - t_enhance) * 1000, 1),
            cn_ms=round((t_cn - t_thin) * 1000, 1),
            filter_ms=round((t_filter - t_cn) * 1000, 1),
        )

        return {"minutiae": normalised, "enhanced_image": enhanced}

    # ------------------------------------------------------------------
    # Quality pipeline (used by both enroll_pairs and search_by_pairs)
    # ------------------------------------------------------------------

    def _run_quality_pipeline(
        self, image_bytes: bytes,
    ) -> dict[str, Any]:
        """Run thinning + CN pipeline and add per-minutia quality scores.

        Pipeline order:
          1. Decode original image
          2. Enhance (Gabor at 350px height)
          3. Normalise enhanced image to 256×256
          4. Thin at 256×256
          5. Crossing Number at 256×256
          6. Filter false minutiae
          7. Score each minutia for quality

        Returns:
            dict with keys:
            - ``minutiae``: list with ``quality`` added to each dict
            - ``skeleton``: 256×256 uint8 skeleton
            - ``normalized_shape``: (h, w) of normalised image
            - ``enhanced_image``: Gabor-filtered image
        """
        from src.processing.crossing_number import extract_minutiae_cn
        from src.processing.false_minutiae_filter import filter_false_minutiae
        from src.processing.minutia_quality import score_minutia
        from src.processing.scale_normalization import normalize_to_256
        from src.processing.thinning import thin

        image = self._decode(image_bytes)
        from src.processing.enhancer import create_enhancer
        enhancer = create_enhancer()
        enhanced = enhancer.enhance(image, resize=True)
        normalized = normalize_to_256(enhanced)
        skeleton = thin(normalized)
        raw_minutiae = extract_minutiae_cn(skeleton)
        filtered = filter_false_minutiae(raw_minutiae, normalized.shape)

        h, w = normalized.shape[:2]
        norm_minutiae: list[dict[str, Any]] = [
            {
                "x": float(m["x"]) / w,
                "y": float(m["y"]) / h,
                "angle": float(m["angle"]),
                "type": int(m["type"]),
            }
            for m in filtered
        ]

        normalized_shape = normalized.shape
        for m in norm_minutiae:
            m["quality"] = score_minutia(m, skeleton, normalized_shape)

        return {
            "minutiae": norm_minutiae,
            "enhanced_image": enhanced,
            "skeleton": skeleton,
            "normalized_shape": normalized_shape,
        }

    # ------------------------------------------------------------------
    # Pair-based enrollment (Phase 27, Plan 27-01)
    # ------------------------------------------------------------------

    def enroll_pairs(
        self,
        capture_id: str,
        fingerprint_id: str,
        person_id: str,
        image_bytes: bytes,
    ) -> int:
        """Quality pipeline → pair extraction → persist in Qdrant.

        Returns the number of pairs inserted.
        """
        import time as _time
        from src.processing.pair_extractor import extract_pairs

        t0 = _time.monotonic()
        result = self._run_quality_pipeline(image_bytes)
        norm_minutiae = result["minutiae"]
        t_pipeline = _time.monotonic()

        pairs = extract_pairs(norm_minutiae, min_quality=config.matching.min_pair_quality)
        t_pairs = _time.monotonic()

        num_pairs = self._pair_repo.bulk_insert_pairs(
            person_id=person_id,
            fingerprint_id=fingerprint_id,
            capture_id=capture_id,
            pair_dicts=pairs,
        )
        t_qdrant = _time.monotonic()

        dev_log(
            "mcc.enroll_pairs",
            capture_id=capture_id,
            person_id=person_id,
            minutiae=len(norm_minutiae),
            pairs=len(pairs),
            inserted=num_pairs,
            pipeline_ms=round((t_pipeline - t0) * 1000, 1),
            pairs_ms=round((t_pairs - t_pipeline) * 1000, 1),
            qdrant_ms=round((t_qdrant - t_pairs) * 1000, 1),
        )
        logger.info(
            "Enrolled pairs: capture %s, %d pairs for %s",
            capture_id, num_pairs, person_id,
        )
        return num_pairs

    # ------------------------------------------------------------------
    # Pair-based search (Phase 27, Plan 27-01)
    # ------------------------------------------------------------------

    def search_by_pairs(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        knn_per_pair: int = 10,
    ) -> SearchByPairsResult:
        """Search using NIST Bozorth3-style pair linking.

        For each probe pair, KNN search the ``pair_features`` collection.
        Bozorth3 linking groups geometrically compatible matches per
        candidate person and scores by largest connected component size.

        Supports multi-orientation probing: if the env var
        ``MCC_MULTI_ORIENT`` is set, the probe is matched at 0°, 30°, 60°,
        and 90° rotations, and the best result is returned.

        Returns a dict with ``candidates`` and ``probe_minutiae`` keys
        matching the frontend's ``SearchByPairsResult`` shape.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor
        import time as _time

        # Multi-orientation probing: try 4 rotations in parallel
        # Each orientation is independent, so ThreadPool gives ~4x speedup.
        multi_orient = os.getenv("MCC_MULTI_ORIENT", "0") == "1"
        angles = [0, 30, 60, 90] if multi_orient else [0]

        def _rotated_image(angle: int) -> bytes:
            """Return image bytes rotated by *angle* degrees."""
            if angle == 0:
                return image_bytes
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            src = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            h, w = src.shape[:2]
            m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(src, m, (w, h), borderValue=0)
            ok, buf = cv2.imencode(".bmp", rotated)
            return buf.tobytes() if ok else image_bytes

        images = {angle: _rotated_image(angle) for angle in angles}
        with ThreadPoolExecutor(max_workers=len(angles)) as pool:
            results = list(pool.map(
                lambda a: self._search_one_orient(images[a], top_k, knn_per_pair),
                angles,
            ))

        best_result: SearchByPairsResult | None = None
        best_score = -1.0
        for result in results:
            if result["candidates"]:
                top_score = result["candidates"][0]["score"]
                if top_score > best_score:
                    best_result = result
                    best_score = top_score

        final = best_result if best_result is not None else results[0]
        return final  # type: ignore[return-value]

    def _search_one_orient(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        knn_per_pair: int = 10,
    ) -> SearchByPairsResult:
        """Single-orientation search (internal, used by multi-orientation wrapper)."""
        import time as _time
        from src.processing.bozorth3_linker import Bozorth3Linker
        from src.processing.pair_extractor import extract_pairs, pair_to_vector

        t0 = _time.monotonic()
        result = self._run_quality_pipeline(image_bytes)
        norm_minutiae = result["minutiae"]
        enhanced = result["enhanced_image"]
        t_pipeline = _time.monotonic()

        probe_pairs = extract_pairs(norm_minutiae, min_quality=config.matching.min_pair_quality)
        t_pairs = _time.monotonic()

        pixel_minutiae = self._norm_to_pixel_coords(norm_minutiae, enhanced.shape)

        if not probe_pairs:
            return {"candidates": [], "probe_minutiae": pixel_minutiae}

        query_vectors = [pair_to_vector(p) for p in probe_pairs]
        t_vectors = _time.monotonic()

        all_hits = self._pair_repo.knn_search_pairs(
            query_vectors, top_k_per_vector=knn_per_pair,
        )
        t_knn = _time.monotonic()

        dev_log(
            "mcc.search.pair_knn",
            probe_pairs=len(probe_pairs),
            raw_hits=len(all_hits),
            pipeline_ms=round((t_pipeline - t0) * 1000, 1),
            pairs_ms=round((t_pairs - t_pipeline) * 1000, 1),
            vectors_ms=round((t_vectors - t_pairs) * 1000, 1),
            knn_ms=round((t_knn - t_vectors) * 1000, 1),
        )

        if not all_hits:
            return {"candidates": [], "probe_minutiae": pixel_minutiae}

        t_link_start = _time.monotonic()
        linker = Bozorth3Linker(
            dx_tol=config.matching.link_dx_tol,
            dy_tol=config.matching.link_dy_tol,
            dtheta_tol=config.matching.link_dtheta_tol,
            saturation=config.matching.confidence_saturation,
        )
        link_results = linker.link(probe_pairs, all_hits, top_k=top_k)
        t_link = _time.monotonic()

        dev_log(
            "mcc.search.linking",
            persons=len(link_results),
            top_score=link_results[0]["score"] if link_results else 0.0,
            top_person=link_results[0]["person_id"] if link_results else None,
            link_ms=round((t_link - t_link_start) * 1000, 1),
        )

        probe_pair_by_idx = {i: p for i, p in enumerate(probe_pairs)}
        candidates: list[dict] = []
        for lr in link_results:
            person_hits = lr["supporting_pairs"]
            supporting_pairs: list[dict] = []
            for h in person_hits:
                pp = probe_pair_by_idx.get(h["query_pair_index"])
                if pp is None:
                    continue
                supporting_pairs.append({
                    "probe_mi_idx": pp["i"],
                    "probe_mj_idx": pp["j"],
                    "candidate_mi_x": float(h["mi_x"]),
                    "candidate_mi_y": float(h["mi_y"]),
                    "candidate_mi_angle": float(h["mi_angle"]),
                    "candidate_mj_x": float(h["mj_x"]),
                    "candidate_mj_y": float(h["mj_y"]),
                    "candidate_mj_angle": float(h["mj_angle"]),
                    "candidate_fingerprint_id": str(h["fingerprint_id"]),
                    "candidate_capture_id": str(h["capture_id"]),
                    "similarity": float(h["similarity"]),
                })

            candidates.append({
                "person_id": lr["person_id"],
                "score": lr["score"],
                "peak_votes": lr["validated_count"],
                "num_probe_pairs": len(probe_pairs),
                "supporting_pairs": supporting_pairs,
                "full_name": None,
                "external_id": None,
            })

        return {"candidates": candidates, "probe_minutiae": pixel_minutiae}

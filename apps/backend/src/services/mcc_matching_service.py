"""
MccMatchingService — Phase 21 (MCC production matching).

Clean Architecture: application service. Orchestrates:

  * An inline MCC-specific mini-pipeline (enhance + orientation + quality +
    skeletonize + RidgeGraphExtractor) that produces minutiae directly from
    the ridge graph — bypassing FingerprintService, which routes through
    SkeletonMinutiaeExtractor and destroys the binary skeleton with a
    ``>127`` re-binarization step.
  * ``extract_cylinders`` — builds L2-normalized 144-D descriptors per minutia.
  * ``QdrantMccRepository`` — persists/searches cylinders in Qdrant.

Algorithm (MCC)
---------------
For each minutia, build a 3-D cylinder aligned to the local ridge
orientation: 12 angular sectors x 4 radial rings x 3 structural features
(orientation, ridge count, frequency). The cylinder is rotation-invariant
(because the orientation field is subtracted) and scale-normalized
(because ridge counts are divided by local ridge frequency).

Search is cosine-KNN per cylinder, votes aggregated per-person, then
normalized by the number of enrolled cylinders to remove population
bias. Final ranking sorts persons by normalized total score descending.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from concurrent.futures import Executor

import cv2
import numpy as np

from src.core.config import config
from src.processing.scale_normalization import TARGET_SIZE
from src.core.types import (
    MatchTraceEntry,
    MccCylinderHit,
    MccSearchHit,
    MinutiaSummary,
)
from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.dev.logger import dev_log

logger = logging.getLogger(__name__)


class SearchByPairsResult(TypedDict):
    candidates: list[dict]
    probe_minutiae: list[dict]


class MccMatchingService:
    """MCC production matching service.

    Handles both enrollment and search. Constructor DI: pass ``mcc_repo``
    in tests; defaults are constructed on first use.
    """

    def __init__(
        self,
        mcc_repo: QdrantMccRepository | None = None,
        pool: Executor | None = None,
    ) -> None:
        self._mcc_repo = mcc_repo or QdrantMccRepository.from_host()
        self._pool = pool
        self._mcc_repo.ensure_collection()

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
                "type": int(m["type"]),
            })
        return result

    def _decode(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            msg = "Failed to decode image bytes"
            raise ValueError(msg)
        return img

    def _run_mcc_pipeline(
        self, image: np.ndarray
    ) -> tuple[
        list[dict[str, Any]],
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray,
    ]:
        """Run the MCC-specific mini-pipeline (Phase 21).

        Mirrors ``scripts/spike_mcc.py`` — uses :class:`RidgeGraphExtractor`
        directly to get minutiae, bypassing :class:`SkeletonMinutiaeExtractor`
        which destroys the binary skeleton with a ``>127`` re-binarization
        step (the root cause of the original 0-cylinder bug for SOCOFing).

        Returns:
            ``(minutiae_dicts, skeleton, orientation_field, frequency_map, enhanced_image)``
            where each ``minutia_dict`` has keys ``(x, y, angle)`` and
            ``enhanced_image`` is the Gabor-filtered 350px-tall image.
        """
        from src.core.interfaces import PipelineContext
        from src.processing.enhancer import create_enhancer
        from src.processing.gabor import QualityMaskStep
        from src.processing.graph_extractor import RidgeGraphExtractor
        from src.processing.pre_hooks import (
            OrientationFieldAnalyzer,
            SingularityDetector,
        )
        from src.processing.skeletonize_step import SkeletonizationStep
        from src.processing.spurious_filter import SkeletonCleanerStep

        ctx = PipelineContext(raw_image=image, fingerprint_id="mcc")
        enh = create_enhancer()
        enhanced = enh.enhance(image, resize=True)
        ctx.enhanced_image = enhanced
        ctx.preprocessed_image = enhanced

        OrientationFieldAnalyzer().process(ctx)
        QualityMaskStep().process(ctx)
        orientation_field = ctx.orientation_field
        frequency_map = ctx.freq_image

        SingularityDetector(roi_radius=140).process(ctx)
        SkeletonizationStep(min_island_size=20).process(ctx)
        SkeletonCleanerStep().process(ctx)
        RidgeGraphExtractor().process(ctx)

        rg = ctx.ridge_graph
        skeleton = ctx.skeleton
        empty_skel = skeleton if skeleton is not None else np.zeros((1, 1), dtype=np.uint8)
        if rg is None or not rg.nodes:
            return ([], empty_skel, orientation_field, frequency_map, enhanced)

        minutiae_dicts = [
            {"x": float(n.x), "y": float(n.y), "angle": float(n.angle)}
            for n in rg.nodes
        ]
        return (minutiae_dicts, empty_skel, orientation_field, frequency_map, enhanced)

    def preview(self, image_bytes: bytes) -> dict[str, Any]:
        """Run the MCC pipeline for preview purposes (no DB / no Qdrant I/O).

        Returns a dict with:
          - ``minutiae``: list of dicts with x, y, angle, type (type=2 = unknown)
          - ``enhanced_image``: the Gabor-filtered image as a numpy array
            (350px-tall, uint8). Caller serializes to base64 PNG.

        This replaces the old ``/fingerprints/preview`` implementation that
        used ``FingerprintService`` with ``SkeletonMinutiaeExtractor``,
        which re-binarized the skeleton and produced 0 minutiae on small
        images (the root cause of empty previews on SOCOFing).
        """
        import time as _time

        t0 = _time.monotonic()
        image = self._decode(image_bytes)
        t_decode = _time.monotonic()
        minutiae_dicts, _skel, _orient, _freq, enhanced = self._run_mcc_pipeline(image)
        t_pipeline = _time.monotonic()

        minutiae: list[dict[str, Any]] = [
            {"x": int(m["x"]), "y": int(m["y"]), "angle": float(m["angle"]), "type": 2}
            for m in minutiae_dicts
        ]
        dev_log(
            "mcc.preview",
            image_bytes=len(image_bytes),
            minutiae=len(minutiae),
            enhanced_shape=list(enhanced.shape),
            decode_ms=round((t_decode - t0) * 1000, 1),
            pipeline_ms=round((t_pipeline - t_decode) * 1000, 1),
        )
        return {"minutiae": minutiae, "enhanced_image": enhanced}

    def preview_thinning(self, image_bytes: bytes) -> dict[str, Any]:
        """Thinning + Crossing Number pipeline for preview.

        Scale-normalises to 256×256, enhances (Gabor), thins (Zhang-Suen),
        detects minutiae via Crossing Number, and filters false ones.

        Returns a dict with:
          - ``minutiae``: list of dicts with x, y, angle (radians), type (1/3)
            in NORMALISED coordinates (x/256, y/256).
          - ``enhanced_image``: the Gabor-filtered image as a numpy array.
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

        # Convert to normalised coordinates
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

    @staticmethod
    def _build_cylinders(
        minutiae_dicts: list[dict[str, Any]],
        skeleton: np.ndarray,
        orientation_field: np.ndarray | None,
        frequency_map: np.ndarray | None,
    ) -> tuple[list[np.ndarray], list[tuple[int, int, float]]]:
        """Build MCC cylinder descriptors from pipeline output.

        Returns:
            ``(cylinders, positions)`` where each element of ``positions``
            is ``(x, y, angle)`` for the corresponding cylinder index.
        """
        from src.processing.mcc_descriptor import extract_cylinders

        if not minutiae_dicts or skeleton is None or skeleton.sum() == 0:
            return [], []

        positions: list[tuple[int, int, float]] = [
            (int(m["x"]), int(m["y"]), float(m["angle"]))
            for m in minutiae_dicts
        ]
        cylinders = extract_cylinders(
            minutiae_dicts,
            skeleton,
            orientation_field=orientation_field,
            frequency_map=frequency_map,
        )
        # extract_cylinders may filter out minutiae that cannot form
        # a valid cylinder; keep only positions matching the output count.
        n = len(cylinders)
        return cylinders, positions[:n]

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(
        self,
        capture_id: str,
        fingerprint_id: str,
        person_id: str,
        image_bytes: bytes,
    ) -> int:
        """Extract minutiae → cylinders → persist in Qdrant.

        Returns the number of cylinders inserted.
        """
        import time as _time
        t0 = _time.monotonic()
        image = self._decode(image_bytes)
        t_decode = _time.monotonic()
        pipeline_result = self._run_mcc_pipeline(image)
        minutiae_dicts, skeleton, orient, freq, _enhanced = pipeline_result
        t_pipeline = _time.monotonic()
        cylinders, positions = self._build_cylinders(minutiae_dicts, skeleton, orient, freq)
        t_cyl = _time.monotonic()
        dev_log(
            "mcc.enroll",
            capture_id=capture_id,
            fingerprint_id=fingerprint_id,
            person_id=person_id,
            image_bytes=len(image_bytes),
            minutiae=len(minutiae_dicts),
            cylinders=len(cylinders),
            decode_ms=round((t_decode - t0) * 1000, 1),
            pipeline_ms=round((t_pipeline - t_decode) * 1000, 1),
            cylinders_ms=round((t_cyl - t_pipeline) * 1000, 1),
        )
        if not cylinders:
            logger.info("No cylinders for capture %s; skipping insert", capture_id)
            return 0
        n = self._mcc_repo.bulk_insert_cylinders(
            person_id=person_id,
            fingerprint_id=fingerprint_id,
            capture_id=capture_id,
            vectors=cylinders,
            cylinder_positions=positions,
        )
        t_qdrant = _time.monotonic()
        dev_log(
            "mcc.enroll.persisted",
            capture_id=capture_id,
            inserted=n,
            qdrant_ms=round((t_qdrant - t_cyl) * 1000, 1),
        )
        logger.info(
            "Enrolled capture %s: %d cylinders for person %s",
            capture_id, n, person_id,
        )
        return n

    # ------------------------------------------------------------------
    # Quality pipeline (Phase 25, Plan 25-01)
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

        # Add quality scores
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
    # Triplet-based enrollment (Phase 25, Plan 25-02)
    # ------------------------------------------------------------------

    def enroll_triplets(
        self,
        capture_id: str,
        fingerprint_id: str,
        person_id: str,
        image_bytes: bytes,
    ) -> tuple[int, int, dict[str, object] | None]:
        """Quality pipeline → triplet extraction → persist in Qdrant.

        Also computes the orientation field on the enhanced image and
        returns it as a serialisable dict so the caller can persist it
        via ``OFRegistry``.

        Returns ``(num_minutiae, num_triplets, of_result)`` where
        ``of_result`` is ``None`` if OF computation fails.
        """
        import time as _time
        from src.processing.triplet_extractor import extract_triplets

        t0 = _time.monotonic()
        result = self._run_quality_pipeline(image_bytes)
        minutiae = result["minutiae"]
        skeleton = result["skeleton"]
        normalized_shape = result["normalized_shape"]
        enhanced = result["enhanced_image"]
        t_pipeline = _time.monotonic()

        triplets = extract_triplets(
            minutiae, skeleton, normalized_shape,
        )
        t_triplets = _time.monotonic()

        self._mcc_repo.ensure_triplet_collection()
        num_triplets = self._mcc_repo.bulk_insert_triplets(
            person_id=person_id,
            fingerprint_id=fingerprint_id,
            capture_id=capture_id,
            triplet_dicts=triplets,
        )
        t_qdrant = _time.monotonic()

        # Build OF for pre-filter (Phase 26)
        of_result: dict[str, object] | None = None
        try:
            from src.processing.of_similarity import OFSimilarity

            of = OFSimilarity.build(enhanced, block_size=16)
            of_result = {
                "ori": of.ori.tolist(),
                "coh": of.coh.tolist(),
                "block_size": of.block_size,
            }
        except Exception as exc:
            logger.warning("OF build failed for capture %s: %s", capture_id, exc)

        t_of = _time.monotonic()
        dev_log(
            "mcc.enroll_triplets",
            capture_id=capture_id,
            person_id=person_id,
            minutiae=len(minutiae),
            triplets=num_triplets,
            has_of=of_result is not None,
            pipeline_ms=round((t_pipeline - t0) * 1000, 1),
            triplets_ms=round((t_triplets - t_pipeline) * 1000, 1),
            qdrant_ms=round((t_qdrant - t_triplets) * 1000, 1),
            of_ms=round((t_of - t_qdrant) * 1000, 1),
        )
        logger.info(
            "Enrolled triplets: capture %s, %d minutiae, %d triplets for %s (OF=%s)",
            capture_id, len(minutiae), num_triplets, person_id,
            "yes" if of_result else "no",
        )
        return len(minutiae), num_triplets, of_result

    # ------------------------------------------------------------------
    # Triplet-based search (Phase 25, Plan 25-02)
    # ------------------------------------------------------------------

    def search_by_triplets(
        self,
        image_bytes: bytes,
        top_k: int = 10,
        knn_per_triplet: int = 5,
        enrolled_ofs: dict[str, Any] | None = None,
    ) -> SearchByPairsResult:
        """Search using triplet-based matching with growing algorithm.

        For each probe triplet, KNN search against the triplet_features
        collection.  The growing algorithm then validates geometric
        consistency of triplet matches per candidate person, replacing
        the old Hough voting with the standard AFIS approach.

        Parameters
        ----------
        enrolled_ofs:
            Pre-fetched OF records keyed by ``fingerprint_id`` (str).
            If provided, KNN hits are pre-filtered by OF similarity
            before the growing algorithm (Phase 26 OF pre-filter).

        Returns a dict with keys:
          - ``candidates``: list of candidate dicts, each with:
            - ``person_id``, ``score``, ``validated_count``,
              ``confirming_triplets``, ``transformation``,
              ``num_probe_triplets``, ``supporting_triplets``
          - ``probe_minutiae``: list of dicts with ``x``, ``y``,
            ``angle``, ``type`` in pixel coords of enhanced image
        """
        import time as _time

        from src.processing.growing_matcher import grow_matches
        from src.processing.triplet_extractor import extract_triplets, triplet_to_vector

        t0 = _time.monotonic()
        result = self._run_quality_pipeline(image_bytes)
        norm_minutiae = result["minutiae"]
        skeleton = result["skeleton"]
        normalized_shape = result["normalized_shape"]
        enhanced = result["enhanced_image"]
        t_pipeline = _time.monotonic()

        probe_triplets = extract_triplets(norm_minutiae, skeleton, normalized_shape)
        t_triplets = _time.monotonic()

        pixel_minutiae = self._norm_to_pixel_coords(norm_minutiae, enhanced.shape)

        if not probe_triplets:
            return {"candidates": [], "probe_minutiae": pixel_minutiae}

        query_vectors = [triplet_to_vector(t) for t in probe_triplets]
        t_vectors = _time.monotonic()

        all_hits = self._mcc_repo.knn_search_triplets(
            query_vectors, top_k_per_vector=knn_per_triplet,
        )
        t_knn = _time.monotonic()

        dev_log(
            "mcc.search.triplet_knn",
            probe_triplets=len(probe_triplets),
            raw_hits=len(all_hits),
            pipeline_ms=round((t_pipeline - t0) * 1000, 1),
            triplets_ms=round((t_triplets - t_pipeline) * 1000, 1),
            vectors_ms=round((t_vectors - t_triplets) * 1000, 1),
            knn_ms=round((t_knn - t_vectors) * 1000, 1),
        )

        if not all_hits:
            return {"candidates": [], "probe_minutiae": pixel_minutiae}

        # OF pre-filter (Phase 26) — drop hits whose enrolled OF is
        # inconsistent with the probe before running growing algorithm.
        # The threshold is configurable: the default (0.50) was tuned
        # for non-altered prints, but SOCOFing Altered-Easy/CR raises
        # the OF RMS to 0.80-0.95 (rotation changes the field). Use
        # MCC_OF_THRESHOLD=0.95 (or higher) when matching altered
        # prints, or MCC_OF_THRESHOLD=0.50 for clean prints.
        if enrolled_ofs is not None:
            try:
                import os as _os
                from src.processing.of_filter import OFFilter
                from src.processing.of_similarity import OFSimilarity

                of_threshold = float(_os.getenv("MCC_OF_THRESHOLD", "0.50"))
                probe_of = OFSimilarity.build(enhanced, block_size=16)
                filter_ = OFFilter(threshold=of_threshold)
                before_n = len(all_hits)
                all_hits = filter_.filter_hits(probe_of, all_hits, enrolled_ofs)
                logger.warning(
                    "OF pre-filter: %d -> %d hits (threshold=%.2f)",
                    before_n, len(all_hits), of_threshold,
                )
            except Exception as exc:
                logger.warning("OF filter failed (continuing without): %s", exc)

            if not all_hits:
                return {"candidates": [], "probe_minutiae": pixel_minutiae}
            t_of = _time.monotonic()
            dev_log(
                "mcc.search.of_filtered",
                remaining_hits=len(all_hits),
                of_ms=round((t_of - t_knn) * 1000, 1),
            )

        # Growing algorithm
        t_grow_start = _time.monotonic()
        growth_results = grow_matches(probe_triplets, all_hits)
        t_grow = _time.monotonic()

        dev_log(
            "mcc.search.growing",
            persons=len(growth_results),
            top_score=growth_results[0].score if growth_results else 0.0,
            top_person=growth_results[0].person_id if growth_results else None,
            grow_ms=round((t_grow - t_grow_start) * 1000, 1),
        )

        # Build response — map triplet fields to frontend's MatchCandidate shape
        probe_triplet_by_idx = {i: t for i, t in enumerate(probe_triplets)}
        candidates: list[dict] = []
        for gr in growth_results[:top_k]:
            # Use only validated hits — these are the ones that fit the
            # confirmed transformation. Including raw KNN hits would mix
            # in noise from other captures of the same person.
            person_hits = gr.validated_hits

            # Flatten triplet hits into supporting_pairs (frontend expects pairs)
            supporting_pairs: list[dict] = []
            for h in person_hits:
                pt = probe_triplet_by_idx.get(h["query_triplet_index"])
                if pt is None:
                    continue
                supporting_pairs.append({
                    "probe_mi_idx": pt["mi_idx"],
                    "probe_mj_idx": pt["mj_idx"],
                    "candidate_mi_x": h["mi_x"],
                    "candidate_mi_y": h["mi_y"],
                    "candidate_mi_angle": h["mi_angle"],
                    "candidate_mj_x": h["mj_x"],
                    "candidate_mj_y": h["mj_y"],
                    "candidate_mj_angle": h["mj_angle"],
                    "candidate_fingerprint_id": h["fingerprint_id"],
                    "candidate_capture_id": h["capture_id"],
                    "similarity": h["similarity"],
                })

            candidates.append({
                "person_id": gr.person_id,
                "score": gr.score,
                "peak_votes": gr.validated_count,
                "peak_transformation": {
                    "dx": round(gr.transform.dx, 4),
                    "dy": round(gr.transform.dy, 4),
                    "dtheta": round(gr.transform.angle, 4),
                },
                "num_probe_pairs": gr.total_probe_triplets,
                "supporting_pairs": supporting_pairs,
                "full_name": None,
                "external_id": None,
            })

        return {"candidates": candidates, "probe_minutiae": pixel_minutiae}

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

        pairs = extract_pairs(norm_minutiae)
        t_pairs = _time.monotonic()

        num_pairs = self._mcc_repo.bulk_insert_pairs(
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

        Returns a dict with ``candidates`` and ``probe_minutiae`` keys
        matching the frontend's ``SearchByPairsResult`` shape.
        """
        import time as _time
        from src.processing.bozorth3_linker import Bozorth3Linker
        from src.processing.pair_extractor import extract_pairs, pair_to_vector

        t0 = _time.monotonic()
        result = self._run_quality_pipeline(image_bytes)
        norm_minutiae = result["minutiae"]
        skeleton = result["skeleton"]
        normalized_shape = result["normalized_shape"]
        enhanced = result["enhanced_image"]
        t_pipeline = _time.monotonic()

        probe_pairs = extract_pairs(norm_minutiae)
        t_pairs = _time.monotonic()

        pixel_minutiae = self._norm_to_pixel_coords(norm_minutiae, enhanced.shape)

        if not probe_pairs:
            return {"candidates": [], "probe_minutiae": pixel_minutiae}

        query_vectors = [pair_to_vector(p) for p in probe_pairs]
        t_vectors = _time.monotonic()

        all_hits = self._mcc_repo.knn_search_pairs(
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

        # Bozorth3 linking
        t_link_start = _time.monotonic()
        linker = Bozorth3Linker(
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

        # Build response
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

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        image_bytes: bytes,
        top_k: int = 10,
    ) -> tuple[list[MinutiaSummary], list[MccSearchHit]]:
        """Search enrolled cylinders for matches to a probe image.

        Returns a tuple of:
          - ``probe_minutiae``: list of :class:`MinutiaSummary` for the
            probe image, in the same order as the cylinder vectors.
          - ``candidates``: ranked list of :class:`MccSearchHit` with
            ``match_trace`` populated (per-candidate per-cylinder pairs,
            top-1 hit per probe cylinder).
        """
        import time as _time
        t0 = _time.monotonic()
        image = self._decode(image_bytes)
        pipeline_result = self._run_mcc_pipeline(image)
        minutiae_dicts, skeleton, orient, freq, _enhanced = pipeline_result
        query_cylinders, query_positions = self._build_cylinders(
            minutiae_dicts, skeleton, orient, freq,
        )
        t_build = _time.monotonic()

        probe_minutiae = [
            MinutiaSummary(
                x=int(m["x"]),
                y=int(m["y"]),
                angle=float(m["angle"]),
                type=2,  # unknown — RidgeGraphExtractor does not classify types
            )
            for m in minutiae_dicts
        ][: len(query_cylinders)]

        if not query_cylinders:
            dev_log(
                "mcc.search.no_cylinders",
                minutiae=len(minutiae_dicts),
                hint="Image too low quality — pipeline produced 0 valid minutiae",
            )
            return probe_minutiae, []

        cylinder_hits = self._exhaustive_match(query_cylinders)
        t_match = _time.monotonic()
        dev_log(
            "mcc.search.exhaustive",
            query_cylinders=len(query_cylinders),
            raw_hits=len(cylinder_hits),
            build_ms=round((t_build - t0) * 1000, 1),
            match_ms=round((t_match - t_build) * 1000, 1),
        )
        if not cylinder_hits:
            return probe_minutiae, []

        # Group hits by person; also bucket by fingerprint for contributing_fingerprints
        per_person: dict[str, list[MccCylinderHit]] = {}
        for h in cylinder_hits:
            per_person.setdefault(h.person_id, []).append(h)

        # Per-cylinder top-1: bucket hits by (person, query_cylinder_index),
        # keep the highest similarity, then flatten.
        per_person_top: dict[str, list[MccCylinderHit]] = {}
        for person_id, hits in per_person.items():
            best_per_probe: dict[int, MccCylinderHit] = {}
            for h in hits:
                cur = best_per_probe.get(h.query_cylinder_index)
                if cur is None or h.similarity > cur.similarity:
                    best_per_probe[h.query_cylinder_index] = h
            per_person_top[person_id] = list(best_per_probe.values())

        # Hough voting per person to find a spatially consistent peak
        # transformation. We vote with the per-probe-cylinder top-1 hit
        # (one vote per query cylinder) so that the votes cluster around
        # the true alignment when the match is genuine.
        hough_result = self._hough_align_hits(query_positions, per_person_top)

        enrolled_counts = self._count_enrolled_by_person()
        query_cylinder_count = len(query_cylinders)
        candidates: list[MccSearchHit] = []
        for person_id, top_hits in per_person_top.items():
            peak_votes, aligned_hits = hough_result.get(person_id, (0, []))

            # Re-pick top-1 per probe cylinder from Hough-filtered hits.
            # These are the geometrically consistent matches.
            best_per_probe: dict[int, MccCylinderHit] = {}
            for h in aligned_hits:
                cur = best_per_probe.get(h.query_cylinder_index)
                if cur is None or h.similarity > cur.similarity:
                    best_per_probe[h.query_cylinder_index] = h
            working_hits = list(best_per_probe.values())

            aligned_similarity = sum(h.similarity for h in working_hits)
            denom = max(query_cylinder_count, 1)
            aligned_score = aligned_similarity / denom

            # NIST-style score for latent matching: count of aligned
            # cylinder matches (``len(working_hits)``) normalised to
            # 0-1 by a saturation point. This matches what the user
            # sees in the UI ("12 match") and aligns with NBIS
            # Bozorth3's convention of reporting the size of the
            # largest connected component. Using ``peak_votes`` (Hough
            # bin votes) instead would give a different number than
            # what the UI shows, which is confusing.
            confidence_saturation = config.matching.confidence_saturation
            score = min(1.0, len(working_hits) / confidence_saturation)
            contributing = sorted({h.fingerprint_id for h in top_hits})

            # Build match_trace in probe-cyl-index order for deterministic UI
            match_trace: list[MatchTraceEntry] = []
            sorted_hits = sorted(working_hits, key=lambda h: h.query_cylinder_index)
            for h in sorted_hits:
                idx = h.query_cylinder_index
                if idx >= len(probe_minutiae) or idx >= len(query_positions):
                    continue
                probe_x, probe_y, probe_angle = query_positions[idx]
                match_trace.append(
                    MatchTraceEntry(
                        probe_cylinder_index=idx,
                        probe_x=probe_x,
                        probe_y=probe_y,
                        probe_angle=probe_angle,
                        candidate_capture_id=h.capture_id,
                        candidate_fingerprint_id=h.fingerprint_id,
                        candidate_x=h.candidate_x,
                        candidate_y=h.candidate_y,
                        candidate_angle=h.candidate_angle,
                        similarity=h.similarity,
                    )
                )
            candidates.append(
                MccSearchHit(
                    person_id=person_id,
                    total_score=score,
                    hits=len(working_hits),
                    contributing_fingerprints=contributing,
                    match_trace=match_trace,
                )
            )
            dev_log(
                "mcc.search.person",
                person_id=person_id,
                peak_votes=peak_votes,
                aligned_score=round(aligned_score, 4),
                score=round(score, 4),
                hits=len(working_hits),
            )

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        dev_log(
            "mcc.search.ranked",
            persons=len(candidates),
            top_score=round(candidates[0].total_score, 4) if candidates else 0.0,
            top_person_id=candidates[0].person_id if candidates else None,
            enrolled_total=sum(enrolled_counts.values()),
        )
        return probe_minutiae, candidates[:top_k]

    def _search_cylinders(
        self,
        query_cylinders: list[np.ndarray],
        top_k: int = 10,
    ) -> list[MccSearchHit]:
        """Search with pre-built query cylinders (benchmark / testing hook).

        Runs KNN against the enrolled collection and aggregates scores by
        person using the same per-fingerprint normalization as :meth:`search`.
        Underscore-prefixed: intended for same-module callers (e.g. the
        Phase 21 SOCOFing benchmark) that need to feed synthetic or
        perturbed cylinders without re-running the full image pipeline.

        .. note::
            This legacy path does NOT build ``match_trace`` because it
            does not have access to the probe minutiae positions. Call
            :meth:`search` for the full match-trace path.
        """
        if not query_cylinders:
            return []

        cylinder_hits = self._mcc_repo.knn_search(
            query_cylinders,
            top_k_per_vector=config.matching.top_k_per_cylinder,
        )
        if not cylinder_hits:
            return []

        enrolled_counts = self._count_enrolled_by_person()
        person_hits = self._mcc_repo.aggregate_scores_by_person(
            cylinder_hits,
            query_cylinder_count=len(query_cylinders),
            enrolled_counts=enrolled_counts,
        )
        return [
            MccSearchHit(
                person_id=p.person_id,
                total_score=p.total_score,
                hits=p.hits,
                contributing_fingerprints=p.contributing_fingerprints,
            )
            for p in person_hits[:top_k]
        ]

    def _exhaustive_match(
        self,
        query_cylinders: list[np.ndarray],
    ) -> list[MccCylinderHit]:
        """Position-aware exhaustive all-pairs matching.

        The standard KNN approach returns top-K candidates by descriptor
        similarity alone — but the MCC descriptor is position-invariant,
        so the KNN top-K often contains candidates at random positions.
        Hough voting cannot find a coherent peak from random-position
        votes, which makes latent matching fail.

        Instead, we compute the similarity between EVERY (probe, candidate)
        pair and keep those above a threshold. This preserves position
        information: each surviving hit is at a specific (x, y, angle),
        and the Hough voter can find the global transformation by clustering
        them.

        Cost is O(M_p x N) similarity computations. For our 5-person /
        500-cylinder demo this is ~50k dot products — negligible. For
        production scale (millions of cylinders) we'd add a KNN
        pre-filter to restrict to a spatial window.
        """
        sim_threshold = max(config.matching.exhaustive_sim_threshold, 0.0)
        top_k_per = max(config.matching.top_k_per_cylinder, 1)

        candidates = self._mcc_repo.scroll_all_cylinders()
        if not candidates or not query_cylinders:
            return []

        # Stack candidate vectors for vectorised cosine (== dot product,
        # since MCC descriptors are L2-normalised).
        cand_vecs = np.stack(
            [np.asarray(c["vector"], dtype=np.float32) for c in candidates],
            axis=0,
        )
        q_vecs = np.stack(
            [np.asarray(q, dtype=np.float32) for q in query_cylinders],
            axis=0,
        )

        # All-pairs similarity: shape (n_query, n_cand)
        sims = q_vecs @ cand_vecs.T

        hits: list[MccCylinderHit] = []
        for q_idx in range(sims.shape[0]):
            row = sims[q_idx]
            # Mask by threshold, then take top-K of the surviving ones.
            mask = row >= sim_threshold
            if not mask.any():
                continue
            cand_indices = np.where(mask)[0]
            cand_sims = row[cand_indices]
            # Partial sort: top-k by similarity, descending.
            if len(cand_indices) > top_k_per:
                top_local = np.argpartition(-cand_sims, top_k_per)[:top_k_per]
                cand_indices = cand_indices[top_local]
                cand_sims = cand_sims[top_local]
            order = np.argsort(-cand_sims)
            for j in order:
                ci = int(cand_indices[int(j)])
                cand = candidates[ci]
                payload = cand.get("payload") or {}
                if not isinstance(payload, dict):
                    continue
                hits.append(
                    MccCylinderHit(
                        person_id=str(payload.get("person_id", "")),
                        fingerprint_id=str(payload.get("fingerprint_id", "")),
                        capture_id=str(payload.get("capture_id", "")),
                        similarity=float(cand_sims[int(j)]),
                        query_cylinder_index=q_idx,
                        candidate_x=int(payload.get("x", 0)),
                        candidate_y=int(payload.get("y", 0)),
                        candidate_angle=float(payload.get("angle", 0.0)),
                    ),
                )
        return hits

    def _hough_align_hits(
        self,
        query_positions: list[tuple[int, int, float]],
        per_person_top: dict[str, list[MccCylinderHit]],
    ) -> dict[str, tuple[int, list[MccCylinderHit]]]:
        """Per-person Hough voting on the rigid (Δx, Δy, Δθ) transformation.

        ``per_person_top`` is the per-probe-cylinder top-1 hit grouped by
        person (one vote per query cylinder per person). We discretize
        the 3-D parameter space into bins (size from config) and find
        the dominant peak per person. Only hits near the peak are
        returned — these are the geometrically consistent matches.

        Returns ``{person_id: (peak_votes, aligned_hits)}``.
        """
        dx_bin = max(config.matching.hough_dx_bin, 1)
        dy_bin = max(config.matching.hough_dy_bin, 1)
        dtheta_bin = max(
            math.radians(config.matching.hough_dtheta_bin_deg), 0.01,
        )
        tol = max(config.matching.hough_peak_tolerance_bins, 0)

        result: dict[str, tuple[int, list[MccCylinderHit]]] = {}
        for person_id, person_hits in per_person_top.items():
            votes: dict[tuple[int, int, int], int] = defaultdict(int)
            hit_by_bin: dict[tuple[int, int, int], list[MccCylinderHit]] = (
                defaultdict(list)
            )
            for h in person_hits:
                if h.query_cylinder_index >= len(query_positions):
                    continue
                qx, qy, qtheta = query_positions[h.query_cylinder_index]
                dx = h.candidate_x - qx
                dy = h.candidate_y - qy
                dtheta = (h.candidate_angle - qtheta + math.pi) % (2 * math.pi) - math.pi
                bin_key = (
                    int(dx // dx_bin),
                    int(dy // dy_bin),
                    int(dtheta // dtheta_bin),
                )
                votes[bin_key] += 1
                hit_by_bin[bin_key].append(h)

            if not votes:
                result[person_id] = (0, [])
                continue

            peak_bin = max(votes, key=lambda k: (votes[k], k))
            peak_votes = votes[peak_bin]

            aligned: list[MccCylinderHit] = []
            for bin_key, bin_hits in hit_by_bin.items():
                if (
                    abs(bin_key[0] - peak_bin[0]) <= tol
                    and abs(bin_key[1] - peak_bin[1]) <= tol
                    and abs(bin_key[2] - peak_bin[2]) <= tol
                ):
                    aligned.extend(bin_hits)

            result[person_id] = (peak_votes, aligned)
        return result

    def _count_enrolled_by_person(self) -> dict[str, int]:
        """Return {person_id: cylinder_count} for all enrollees."""
        counts: dict[str, int] = {}
        offset: object = None
        seen_persons: set[str] = set()
        while True:
            records, offset = self._mcc_repo._client.scroll(
                collection_name=self._mcc_repo._collection,
                limit=256,
                offset=offset,  # type: ignore[arg-type]
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                pid = (rec.payload or {}).get("person_id")
                if pid and pid not in seen_persons:
                    counts[pid] = self._mcc_repo.count_by_person(pid)
                    seen_persons.add(pid)
            if offset is None:
                break
        return counts

"""Orientation field pre-filter for KNN hits (Phase 26, Plan 26-01, T6).

Drops hits whose person_id's enrolled OF is inconsistent with the
probe's OF before the growing algorithm runs.  This is the fix for
Phase 25's 0/5 crop acceptance failure.
"""

from __future__ import annotations

import logging
from typing import Any

from src.processing.of_similarity import (
    OFSimilarity,
    OFSimilarityError,
    OF_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


class OFFilter:
    """Filters KNN hits by orientation field similarity.

    Usage::

        filter_ = OFFilter(threshold=0.50)
        filtered = filter_.filter_hits(probe_of, knn_hits, enrolled_records)

    The caller is responsible for fetching ``OFRecord`` dicts from
    ``OFRegistry`` (async) before calling this sync method.
    """

    def __init__(self, threshold: float = OF_SIMILARITY_THRESHOLD) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    def filter_hits(
        self,
        probe_of: OFSimilarity,
        knn_hits: list[dict[str, Any]],
        enrolled_ofs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Drop hits whose person has an inconsistent enrolled OF.

        Parameters
        ----------
        probe_of:
            OF of the probe image (built from ``OFSimilarity.build``).
        knn_hits:
            Raw KNN hits from ``knn_search_triplets``.
        enrolled_ofs:
            Dict mapping ``fingerprint_id`` (str) → ``OFRecord``,
            pre-fetched by the caller via ``OFRegistry.get_many``.

        Returns
        -------
        Filtered list of hits (only person_ids whose RMS <= threshold).
        """
        if not knn_hits or not enrolled_ofs:
            dev_log_filter(
                "mcc.search.of_filter",
                probe_persons=len({h.get("person_id", "") for h in knn_hits}),
                kept_persons=len({h.get("person_id", "") for h in knn_hits}),
                dropped_persons=0,
                min_rms=-1.0,
                max_rms=-1.0,
                reason="no_of_data",
            )
            return list(knn_hits)

        # Build per-person fingerprint IDs from hits
        person_fingerprints: dict[str, set[str]] = {}
        for hit in knn_hits:
            pid = str(hit.get("person_id", ""))
            fid = str(hit.get("fingerprint_id", ""))
            if fid:
                person_fingerprints.setdefault(pid, set()).add(fid)

        rms_scores: dict[str, float] = {}
        for pid, fids in person_fingerprints.items():
            # Compute RMS for each enrolled fingerprint of this person
            person_rms: list[float] = []
            for fid in fids:
                rec = enrolled_ofs.get(fid)
                if rec is None:
                    logger.debug(
                        "No OF record for fingerprint %s (person=%s); skipping",
                        fid, pid,
                    )
                    continue
                try:
                    import numpy as np

                    ori_arr = np.array(rec["of_ori"], dtype=np.float32)
                    coh_arr = np.array(rec["of_coh"], dtype=np.float32)
                    rms = probe_of.compare_raw(ori_arr, coh_arr)
                    person_rms.append(rms)
                except (OFSimilarityError, Exception) as exc:
                    logger.debug(
                        "OF compare failed for %s/%s: %s", pid, fid, exc,
                    )
                    continue

            if person_rms:
                rms_scores[pid] = min(person_rms)
            else:
                rms_scores[pid] = 0.0  # no OF data → keep (don't filter)

        # Filter hits
        kept_persons: set[str] = set()
        dropped_persons: set[str] = set()
        for pid, rms in rms_scores.items():
            if rms <= self._threshold:
                kept_persons.add(pid)
            else:
                dropped_persons.add(pid)

        filtered = [h for h in knn_hits if h.get("person_id", "") in kept_persons]

        all_rms = [rms_scores[p] for p in rms_scores if rms_scores[p] >= 0]

        dev_log_filter(
            "mcc.search.of_filter",
            probe_persons=len(person_fingerprints),
            kept_persons=len(kept_persons),
            dropped_persons=len(dropped_persons),
            min_rms=round(min(all_rms), 4) if all_rms else -1.0,
            max_rms=round(max(all_rms), 4) if all_rms else -1.0,
            threshold=self._threshold,
        )

        return filtered


def build_probe_of(enhanced_image: Any) -> OFSimilarity:
    """Build probe OF from the enhanced image.

    Convenience wrapper around ``OFSimilarity.build`` that hides the
    import for callers.
    """
    return OFSimilarity.build(enhanced_image, block_size=16)


def dev_log_filter(event: str, **fields: Any) -> None:
    """Thin dev_log wrapper that gates on the env var."""
    from src.dev.logger import dev_log as _dev_log

    _dev_log(event, **fields)

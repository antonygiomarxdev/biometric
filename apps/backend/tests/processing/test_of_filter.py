"""Tests for OF filter (Phase 26, Plan 26-01, T6)."""

from __future__ import annotations

import numpy as np
import pytest

from src.processing.of_filter import OFFilter
from src.processing.of_similarity import (
    OFSimilarity,
    OF_SIMILARITY_THRESHOLD,
)


@pytest.fixture
def probe_of() -> OFSimilarity:
    ori = np.zeros((4, 4), dtype=np.float32)
    coh = np.ones((4, 4), dtype=np.float32) * 0.8
    return OFSimilarity(ori, coh)


@pytest.fixture
def enrolled_ofs() -> dict[str, dict]:
    """Two persons: one with OF similar to probe, one with perpendicular."""
    ori_same = np.zeros((4, 4), dtype=np.float32)
    coh_same = np.ones((4, 4), dtype=np.float32) * 0.8

    ori_diff = np.ones((4, 4), dtype=np.float32) * (np.pi / 2)
    coh_diff = np.ones((4, 4), dtype=np.float32) * 0.8

    return {
        "fid_same": {
            "fingerprint_id": "fid_same",
            "of_ori": ori_same.tolist(),
            "of_coh": coh_same.tolist(),
            "block_size": 16,
            "pseudo_core": (2, 2),
            "enrolled_at": None,
        },
        "fid_diff": {
            "fingerprint_id": "fid_diff",
            "of_ori": ori_diff.tolist(),
            "of_coh": coh_diff.tolist(),
            "block_size": 16,
            "pseudo_core": (2, 2),
            "enrolled_at": None,
        },
    }


def _make_hit(person_id: str, fingerprint_id: str) -> dict:
    return {
        "query_triplet_index": 0,
        "similarity": 0.9,
        "person_id": person_id,
        "fingerprint_id": fingerprint_id,
        "capture_id": "cap1",
        "mi_idx": 0, "mj_idx": 1, "mk_idx": 2,
        "mi_x": 0.0, "mi_y": 0.0, "mi_angle": 0.0,
        "mj_x": 0.1, "mj_y": 0.0, "mj_angle": 0.0,
        "mk_x": 0.0, "mk_y": 0.1, "mk_angle": 0.0,
        "type_triple": 0, "quality_min": 0.5,
    }


# ---------------------------------------------------------------------------
# Basic filter behaviour
# ---------------------------------------------------------------------------


def test_filter_drops_high_rms_persons(
    probe_of: OFSimilarity,
    enrolled_ofs: dict,
) -> None:
    filter_ = OFFilter(threshold=0.50)
    hits = [
        _make_hit("person_same", "fid_same"),
        _make_hit("person_diff", "fid_diff"),
    ]
    filtered = filter_.filter_hits(probe_of, hits, enrolled_ofs)
    kept_persons = {h["person_id"] for h in filtered}
    assert "person_same" in kept_persons
    assert "person_diff" not in kept_persons


def test_filter_keeps_low_rms_persons(
    probe_of: OFSimilarity,
    enrolled_ofs: dict,
) -> None:
    filter_ = OFFilter(threshold=2.5)  # high threshold — keeps all
    hits = [
        _make_hit("person_same", "fid_same"),
        _make_hit("person_diff", "fid_diff"),
    ]
    filtered = filter_.filter_hits(probe_of, hits, enrolled_ofs)
    assert len(filtered) == 2


def test_filter_with_empty_registry_passes_all(
    probe_of: OFSimilarity,
) -> None:
    filter_ = OFFilter()
    hits = [_make_hit("person_same", "fid_same")]
    filtered = filter_.filter_hits(probe_of, hits, {})
    assert len(filtered) == 1


def test_filter_with_no_knn_hits(
    probe_of: OFSimilarity,
    enrolled_ofs: dict,
) -> None:
    filter_ = OFFilter()
    filtered = filter_.filter_hits(probe_of, [], enrolled_ofs)
    assert filtered == []


def test_threshold_configurable() -> None:
    filter_ = OFFilter(threshold=0.75)
    assert filter_.threshold == 0.75
    filter_.threshold = 0.25
    assert filter_.threshold == 0.25


def test_filter_handles_missing_of_gracefully(
    probe_of: OFSimilarity,
) -> None:
    filter_ = OFFilter()
    hits = [
        _make_hit("person_unknown", "fid_unknown"),
    ]
    enrolled = {}  # no OF records for this fingerprint
    filtered = filter_.filter_hits(probe_of, hits, enrolled)
    # No OF data → keep (don't filter out an unknown)
    assert len(filtered) == 1


def test_default_threshold() -> None:
    f = OFFilter()
    assert f.threshold == OF_SIMILARITY_THRESHOLD

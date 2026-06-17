"""Unit tests for MccMatchingService (Phase 21)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import QdrantClient

from src.db.qdrant_mcc_repository import QdrantMccRepository
from src.services.mcc_matching_service import (
    MccMatchingService,
    MccSearchHit,
)

_FAKE_IMAGE = np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _mock_imdecode() -> None:
    """Mock cv2.imdecode to return a valid grayscale image."""
    with patch("src.services.mcc_matching_service.cv2.imdecode", return_value=_FAKE_IMAGE):
        yield


@pytest.fixture
def repo() -> QdrantMccRepository:
    client = QdrantClient(location=":memory:")
    r = QdrantMccRepository(client, collection="test_mcc_svc")
    r.ensure_collection()
    return r


def _make_pipeline_result(num_minutiae: int = 3):
    """Build the (minutiae_dicts, skeleton, orient, freq) tuple that
    ``MccMatchingService._run_mcc_pipeline`` would return.

    The skeleton is a solid 80x80 block of 1s so that
    ``extract_cylinders`` produces one cylinder per minutia.
    """
    minutiae_dicts = [
        {"x": float(10 + i * 20), "y": float(20 + i * 10), "angle": 0.1 * i}
        for i in range(num_minutiae)
    ]
    skeleton = np.zeros((100, 100), dtype=np.uint8)
    skeleton[10:90, 10:90] = 1
    return minutiae_dicts, skeleton, None, None


def test_enroll_returns_count(repo: QdrantMccRepository) -> None:
    svc = MccMatchingService(mcc_repo=repo)
    with patch.object(svc, "_run_mcc_pipeline", return_value=_make_pipeline_result(3)):
        n = svc.enroll(
            capture_id="c1",
            fingerprint_id="f1",
            person_id="p1",
            image_bytes=b"fake-bytes",
        )
    assert n == 3
    assert repo.count_by_person("p1") == 3


def test_enroll_returns_zero_for_insufficient_minutiae(
    repo: QdrantMccRepository,
) -> None:
    svc = MccMatchingService(mcc_repo=repo)
    with patch.object(svc, "_run_mcc_pipeline", return_value=_make_pipeline_result(0)):
        n = svc.enroll(capture_id="c1", fingerprint_id="f1", person_id="p1", image_bytes=b"x")
    assert n == 0


def test_search_finds_enrolled_match(repo: QdrantMccRepository) -> None:
    svc = MccMatchingService(mcc_repo=repo)
    with patch.object(svc, "_run_mcc_pipeline", return_value=_make_pipeline_result(3)):
        svc.enroll("c1", "f1", "p1", b"fake")
        svc.enroll("c2", "f2", "p2", b"fake")

        hits = svc.search(b"fake", top_k=5)
    assert len(hits) >= 1
    assert all(isinstance(h, MccSearchHit) for h in hits)
    for a, b in zip(hits, hits[1:]):
        assert a.total_score >= b.total_score


def test_search_returns_empty_when_no_enrollment(
    repo: QdrantMccRepository,
) -> None:
    svc = MccMatchingService(mcc_repo=repo)
    with patch.object(svc, "_run_mcc_pipeline", return_value=_make_pipeline_result(3)):
        hits = svc.search(b"fake", top_k=5)
    assert hits == []


def test_search_respects_top_k(repo: QdrantMccRepository) -> None:
    svc = MccMatchingService(mcc_repo=repo)
    with patch.object(svc, "_run_mcc_pipeline", return_value=_make_pipeline_result(3)):
        for i in range(5):
            svc.enroll(f"c{i}", f"f{i}", f"p{i}", b"fake")
        hits = svc.search(b"fake", top_k=3)
    assert len(hits) <= 3

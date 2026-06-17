"""Unit tests for MccMatchingService (Phase 21)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


@pytest.fixture
def fp_service_mock() -> MagicMock:
    mock = MagicMock()
    normalized = MagicMock()
    mock._process_image = MagicMock(return_value=normalized)
    return mock


def _make_normalized(num_minutiae: int = 3) -> MagicMock:
    n = MagicMock()
    n.minutiae = [
        MagicMock(x=10 + i * 20, y=20 + i * 10, angle=0.1 * i)
        for i in range(num_minutiae)
    ]
    # Simulate the skeleton/field attributes that extract_cylinders looks for
    n.image = np.zeros((100, 100), dtype=np.uint8)
    n.image[10:90, 10:90] = 1  # non-zero skeleton region
    # Explicitly set optional fields to None so getattr(…, None) returns None
    n.orientation_field = None
    n.freq_image = None
    n.skeleton = None
    return n


def test_enroll_returns_count(repo: QdrantMccRepository, fp_service_mock: MagicMock) -> None:
    fp_service_mock._process_image.return_value = _make_normalized(3)
    svc = MccMatchingService(fingerprint_service=fp_service_mock, mcc_repo=repo)
    n = svc.enroll(
        capture_id="c1",
        fingerprint_id="f1",
        person_id="p1",
        image_bytes=b"fake-bytes",
    )
    assert n == 3
    assert repo.count_by_person("p1") == 3


def test_enroll_returns_zero_for_insufficient_minutiae(
    repo: QdrantMccRepository, fp_service_mock: MagicMock
) -> None:
    fp_service_mock._process_image.return_value = _make_normalized(0)
    svc = MccMatchingService(fingerprint_service=fp_service_mock, mcc_repo=repo)
    n = svc.enroll(capture_id="c1", fingerprint_id="f1", person_id="p1", image_bytes=b"x")
    assert n == 0


def test_search_finds_enrolled_match(
    repo: QdrantMccRepository, fp_service_mock: MagicMock
) -> None:
    fp_service_mock._process_image.return_value = _make_normalized(3)
    svc = MccMatchingService(fingerprint_service=fp_service_mock, mcc_repo=repo)
    svc.enroll("c1", "f1", "p1", b"fake")
    svc.enroll("c2", "f2", "p2", b"fake")

    hits = svc.search(b"fake", top_k=5)
    assert len(hits) >= 1
    assert all(isinstance(h, MccSearchHit) for h in hits)
    for a, b in zip(hits, hits[1:]):
        assert a.total_score >= b.total_score


def test_search_returns_empty_when_no_enrollment(
    repo: QdrantMccRepository, fp_service_mock: MagicMock
) -> None:
    fp_service_mock._process_image.return_value = _make_normalized(3)
    svc = MccMatchingService(fingerprint_service=fp_service_mock, mcc_repo=repo)
    hits = svc.search(b"fake", top_k=5)
    assert hits == []


def test_search_respects_top_k(
    repo: QdrantMccRepository, fp_service_mock: MagicMock
) -> None:
    fp_service_mock._process_image.return_value = _make_normalized(3)
    svc = MccMatchingService(fingerprint_service=fp_service_mock, mcc_repo=repo)
    for i in range(5):
        svc.enroll(f"c{i}", f"f{i}", f"p{i}", b"fake")
    hits = svc.search(b"fake", top_k=3)
    assert len(hits) <= 3

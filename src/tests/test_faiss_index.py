import numpy as np
import pytest
from pathlib import Path
from src.fingerprint.infrastructure.vector_index.faiss_index import FingerprintVectorIndex


def test_add_and_search(tmp_path: Path) -> None:
    index_path = tmp_path / "test.index"
    index = FingerprintVectorIndex(str(index_path), dim=4)

    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    idx = index.add(vec)
    assert idx == 0

    ids, distances = index.search(vec, k=1)
    assert ids[0] == 0
    assert distances[0] == pytest.approx(0.0, abs=1e-6)

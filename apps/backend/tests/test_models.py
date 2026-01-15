"""Tests para modelos de datos."""

import pytest
import numpy as np

from src.core.types import MinutiaCandidate, NormalizedFingerprint, MatchResult, MinutiaType, AlgorithmOrigin


def test_minutiae_creation():
    """Test creación de Minutiae."""
    minutiae = MinutiaCandidate(
        x=100,
        y=150,
        angle=45.0,
        type=MinutiaType.TERMINATION,
        confidence=1.0,
        origin=AlgorithmOrigin.SKELETON
    )
    
    assert minutiae.type == MinutiaType.TERMINATION
    assert minutiae.x == 100
    assert minutiae.y == 150
    assert minutiae.angle == 45.0


def test_fingerprint_to_vector():
    """Test conversión de Fingerprint a vector."""
    minutiae = [
        MinutiaCandidate(10, 20, 0.1, MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON),
        MinutiaCandidate(30, 40, 0.2, MinutiaType.BIFURCATION, 1.0, AlgorithmOrigin.SKELETON),
    ]
    
    fingerprint = NormalizedFingerprint(id="test_001", minutiae=minutiae, width=200, height=200)
    vector = fingerprint.vector
    
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert len(vector) == 8  # 4 valores por minutia
    
    # Verificar valores
    assert vector[0] == 0  # termination = 0
    assert vector[1] == 10
    assert vector[2] == 20
    assert vector[4] == 1  # bifurcation = 1


def test_fingerprint_num_minutiae():
    """Test conteo de minutiae."""
    minutiae = [
        MinutiaCandidate(10, 20, 0.1, MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON),
        MinutiaCandidate(30, 40, 0.2, MinutiaType.BIFURCATION, 1.0, AlgorithmOrigin.SKELETON),
        MinutiaCandidate(50, 60, 0.3, MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON),
    ]
    
    fingerprint = NormalizedFingerprint(id="test", minutiae=minutiae, width=200, height=200)
    assert len(fingerprint.minutiae) == 3


def test_match_result_creation():
    """Test creación de MatchResult."""
    result = MatchResult(
        matched=True,
        person_id="P001",
        score=0.95,
        confidence=0.9,
        l2_distance=100.0,
        cosine_distance=0.1,
        combined_score=0.95,
        metadata={"name": "Juan Pérez", "document": "12345678"}
    )
    
    assert result.matched is True
    assert result.person_id == "P001"
    assert result.score == 0.95
    assert result.metadata["name"] == "Juan Pérez"

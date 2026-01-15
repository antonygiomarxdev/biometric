"""Tests para el vectorizador de minutiae."""

import pytest
import numpy as np

from src.core.types import MinutiaCandidate, MinutiaType, AlgorithmOrigin
from src.processing.vectorizer import MinutiaeVectorizer


def test_to_vector():
    """Test conversión de minutiae a vector."""
    minutiae = [
        MinutiaCandidate(10, 20, 30.0, MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON),
        MinutiaCandidate(40, 50, 60.0, MinutiaType.BIFURCATION, 1.0, AlgorithmOrigin.SKELETON),
    ]
    
    vector = MinutiaeVectorizer.to_vector(minutiae)
    
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert len(vector) == 8
    
    # Verificar valores
    np.testing.assert_array_equal(
        vector,
        np.array([0, 10, 20, 30, 1, 40, 50, 60], dtype=np.float32)
    )


def test_from_vector():
    """Test reconstrucción de minutiae desde vector."""
    original = [
        MinutiaCandidate(10, 20, 30.0, MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON),
        MinutiaCandidate(40, 50, 60.0, MinutiaType.BIFURCATION, 1.0, AlgorithmOrigin.SKELETON),
    ]
    
    vector = MinutiaeVectorizer.to_vector(original)
    reconstructed = MinutiaeVectorizer.from_vector(vector)
    
    assert len(reconstructed) == len(original)
    assert reconstructed[0].type == MinutiaType.TERMINATION
    assert reconstructed[0].x == 10
    assert reconstructed[0].y == 20
    assert reconstructed[1].type == MinutiaType.BIFURCATION


def test_pad_vector():
    """Test padding de vectores."""
    vector = np.array([1, 2, 3, 4], dtype=np.float32)
    
    # Pad a mayor dimensión
    padded = MinutiaeVectorizer.pad_vector(vector, 10)
    assert len(padded) == 10
    np.testing.assert_array_equal(padded[:4], vector)
    np.testing.assert_array_equal(padded[4:], np.zeros(6))
    
    # Truncar a menor dimensión
    truncated = MinutiaeVectorizer.pad_vector(vector, 2)
    assert len(truncated) == 2
    np.testing.assert_array_equal(truncated, vector[:2])


def test_empty_minutiae():
    """Test vectorización de lista vacía."""
    vector = MinutiaeVectorizer.to_vector([])
    assert len(vector) == 0

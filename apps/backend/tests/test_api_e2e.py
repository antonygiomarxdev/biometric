"""Tests E2E para la API REST."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import io
import cv2

from src.api.rest import app
from src.core.types import NormalizedFingerprint, MinutiaCandidate, MinutiaType, AlgorithmOrigin, MatchResult

# Cliente de prueba
client = TestClient(app)

@pytest.fixture
def mock_image_bytes():
    """Genera bytes de una imagen válida."""
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), 255, 2)
    _, encoded = cv2.imencode('.png', img)
    return encoded.tobytes()

@pytest.fixture
def mock_fingerprint():
    """Fingerprint mock."""
    return NormalizedFingerprint(
        id="test_id",
        minutiae=[
            MinutiaCandidate(10, 10, 0.0, MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON),
            MinutiaCandidate(20, 20, 0.0, MinutiaType.BIFURCATION, 1.0, AlgorithmOrigin.SKELETON)
        ],
        width=200,
        height=200
    )

@pytest.fixture
def mock_services(mock_fingerprint):
    """Mockea los servicios subyacentes."""
    with patch("src.api.rest.fingerprint_service") as fps_mock, \
         patch("src.api.rest.comparison_service") as cs_mock:
        
        # Setup fingerprint service
        fps_mock.process_image_from_bytes.return_value = mock_fingerprint
        
        # Setup comparison service
        cs_mock.register_fingerprint.return_value = 123
        cs_mock.identify.return_value = MatchResult(
            matched=True,
            person_id="P001",
            score=0.95,
            confidence=0.9,
            l2_distance=100.0,
            cosine_distance=0.1,
            combined_score=0.95,
            metadata={"name": "Test User", "document": "12345678"}
        )
        
        yield fps_mock, cs_mock

def test_health_check():
    """Test endpoint de salud."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database_records" in data

def test_extract_minutiae(mock_image_bytes, mock_services):
    """Test endpoint de extracción."""
    response = client.post(
        "/extract",
        files={"file": ("test.png", mock_image_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["minutiae_count"] == 2
    assert data["terminations"] == 1
    assert data["bifurcations"] == 1

def test_register_fingerprint(mock_image_bytes, mock_services):
    """Test endpoint de registro."""
    response = client.post(
        "/register",
        data={
            "person_id": "P001",
            "name": "Test User",
            "document": "12345678"
        },
        files={"file": ("test.png", mock_image_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["record_id"] == 123
    assert data["person_id"] == "P001"

def test_identify_fingerprint(mock_image_bytes, mock_services):
    """Test endpoint de identificación."""
    response = client.post(
        "/identify",
        files={"file": ("test.png", mock_image_bytes, "image/png")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["matched"] is True
    assert data["person_id"] == "P001"
    assert data["name"] == "Test User"
    assert data["score"] == 0.95

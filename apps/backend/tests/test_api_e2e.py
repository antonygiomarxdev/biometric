"""Tests E2E para la API REST."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import io
import cv2

"""
Tests E2E para la API REST — modular router architecture.

NOTE: These tests were originally written for the monolithic ``rest.py``.
The endpoints ``/extract``, ``/register``, and ``/identify`` have been
migrated to the modular routers (``huellas_conocidas``, ``matching``, etc.)
and require updated test coverage.  The health check test has been
preserved for the new ``src.main`` entrypoint.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np
import io
import cv2

from src.main import app
from src.core.types import NormalizedFingerprint, MinutiaCandidate, MinutiaType, AlgorithmOrigin, MatchResult
from src.services.fingerprint_service import fingerprint_service
from src.services.comparison_service import comparison_service

# Cliente de prueba
client = TestClient(app)


def test_health_check():
    """Test endpoint de salud."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

"""Tests for fingerprints endpoints — Phase 17 (CRUD) + Phase 23 (/preview)."""

from __future__ import annotations

import base64
import io
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_fingerprint_service
from src.api.routers.fingerprints import router


@pytest.mark.asyncio
async def test_preview_endpoint() -> None:
    """POST /api/v1/fingerprints/preview returns processed_image + minutiae (Phase 23)."""
    # Mock FingerprintService.process_image_from_bytes to return a NormalizedFingerprint-like
    mock_image = np.zeros((100, 100), dtype=np.uint8)
    mock_minutia = MagicMock()
    mock_minutia.x = 10
    mock_minutia.y = 20
    mock_minutia.angle = 0.5
    mock_minutia.type = MagicMock()
    mock_minutia.type.value = 1  # bifurcation

    mock_normalized = MagicMock()
    mock_normalized.minutiae = [mock_minutia]
    mock_normalized.image = mock_image

    mock_fp_service = MagicMock()
    mock_fp_service.process_image_from_bytes = MagicMock(return_value=mock_normalized)

    app = FastAPI()
    app.dependency_overrides[get_fingerprint_service] = lambda: mock_fp_service
    app.include_router(router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/fingerprints/preview",
            files={"file": ("test.bmp", io.BytesIO(b"fake-bmp-content"), "image/bmp")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "processed_image" in data
    assert isinstance(data["processed_image"], str)
    # It's base64; verify it decodes to bytes
    decoded = base64.b64decode(data["processed_image"])
    assert len(decoded) > 0

    assert "minutiae" in data
    assert isinstance(data["minutiae"], list)
    assert len(data["minutiae"]) == 1
    m = data["minutiae"][0]
    assert m["x"] == 10
    assert m["y"] == 20
    assert m["type"] == 1

    assert "terminations" in data
    assert data["terminations"] == 0  # the one minutia is type=1 (bifurcation)
    assert "bifurcations" in data
    assert data["bifurcations"] == 1
    assert "image_shape" in data
    assert data["image_shape"] == [100, 100]
    assert "image_dtype" in data
    assert data["image_dtype"] == "uint8"

    # 400 on empty file
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/fingerprints/preview",
            files={"file": ("empty.bmp", b"", "image/bmp")},
        )
    assert response.status_code == 400

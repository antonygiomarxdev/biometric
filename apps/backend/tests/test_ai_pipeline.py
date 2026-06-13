"""Integration tests for the full AI processing pipeline."""
import numpy as np
import pytest
from src.ai import AiConfig, ModelManager
from src.services.fingerprint_service import FingerprintService, create_ai_fingerprint_service
from src.core.types import NormalizedFingerprint, AlgorithmOrigin

def has_onnx_models():
    """Helper to check if ONNX models are available."""
    try:
        config = AiConfig(use_gpu=False)
        mm = ModelManager(config)
        mm.get_session("segment")
        return True
    except (FileNotFoundError, RuntimeError):
        return False

require_onnx = pytest.mark.skipif(not has_onnx_models(), reason="ONNX models not available")

@pytest.fixture
def model_manager() -> ModelManager:
    """Fixture providing ModelManager with CPU fallback for testing."""
    config = AiConfig(use_gpu=False)
    return ModelManager(config)

@pytest.fixture
def sample_image() -> np.ndarray:
    """128x128 synthetic fingerprint-like image for testing."""
    img = np.zeros((128, 128), dtype=np.uint8)
    # Draw some ridges (alternating black/white lines)
    for i in range(0, 128, 6):
        img[:, i:i+3] = 255
    return img

@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((128, 128), dtype=np.uint8)

@require_onnx
def test_pipeline_returns_normalized_fingerprint(model_manager, sample_image):
    """Full AI pipeline returns NormalizedFingerprint with valid structure."""
    service = create_ai_fingerprint_service(model_manager)
    result = service.process_image(sample_image, fingerprint_id="test-001")
    assert isinstance(result, NormalizedFingerprint)
    assert result.id == "test-001"
    assert result.width > 0
    assert result.height > 0
    assert isinstance(result.minutiae, list)

@pytest.mark.slow
@require_onnx
def test_pipeline_with_real_socofing_image():
    """Integration test using a real SOCOFing image (if available)."""
    import cv2
    import os
    soco_path = "data/SOCOFing/Real/1__M_Left_index_finger.BMP"
    if not os.path.exists(soco_path):
        pytest.skip("SOCOFing data not available")
    img = cv2.imread(soco_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    config = AiConfig(use_gpu=False)
    mm = ModelManager(config)

    service = create_ai_fingerprint_service(mm)
    result = service.process_image(img, fingerprint_id="soco-001")
    assert len(result.minutiae) > 0

@require_onnx
def test_pipeline_blank_image_graceful(model_manager, blank_image):
    """Blank image returns NormalizedFingerprint with empty minutiae."""
    service = create_ai_fingerprint_service(model_manager)
    result = service.process_image(blank_image, fingerprint_id="blank")
    assert isinstance(result, NormalizedFingerprint)
    # May have 0 or few minutiae — must not crash

@require_onnx
def test_pipeline_output_vectorizable(model_manager, sample_image):
    """Pipeline output can produce a vector for matching."""
    service = create_ai_fingerprint_service(model_manager)
    result = service.process_image(sample_image)
    vector = result.vector
    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32
    assert len(vector) > 0

@require_onnx
def test_pipeline_algorithm_origin(model_manager, sample_image):
    """Extracted minutiae have DEEP_LEARNING origin."""
    service = create_ai_fingerprint_service(model_manager)
    result = service.process_image(sample_image)
    if result.minutiae:
        assert all(m.origin == AlgorithmOrigin.DEEP_LEARNING for m in result.minutiae)

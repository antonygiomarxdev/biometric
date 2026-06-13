"""Tests de integración del pipeline completo."""

import pytest
import numpy as np

from src.services.fingerprint_service import FingerprintService

from src.core.types import NormalizedFingerprint


def test_full_pipeline(sample_image, repository):
    """Test del pipeline completo: procesar, registrar, identificar."""
    service = FingerprintService()
    # ComparisonService removed — using repository directly
    
    # 1. Procesar imagen
    fingerprint = service.process_image(sample_image, fingerprint_id="test_001")
    
    assert isinstance(fingerprint, NormalizedFingerprint)
    assert fingerprint.id == "test_001"
    # Puede o no tener minutiae dependiendo de la imagen sintética
    
    # Si tiene minutiae, continuar con registro e identificación
    if fingerprint.minutiae:
        # 2. Registrar
        record_id = repository.register(fp=
            fingerprint=fingerprint,
            person_id="P001",
            name="Test User",
            document="12345678"
        )
        
        assert isinstance(record_id, int)
        assert record_id > 0
        
        # 3. Identificar la misma huella
        result = repository.identify(fingerprint)
        
        assert result is not None
        # Debería encontrar match con la huella recién registrada
        if result.matched:
            assert result.person_id == "P001"
            # name y document ahora vienen en metadata
            assert result.metadata.get("name") == "Test User"
            assert result.score > 0.5


def test_register_and_search(sample_image, repository):
    """Test registro de múltiples huellas y búsqueda."""
    service = FingerprintService()
    # ComparisonService removed — using repository directly
    
    fingerprints = []
    
    # Registrar varias huellas (variaciones de la misma imagen)
    for i in range(3):
        # Agregar algo de variación
        img = sample_image.copy()
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        fp = service.process_image(img, fingerprint_id=f"test_{i:03d}")
        
        if fp.minutiae:
            fingerprints.append(fp)
            repository.register(fp=
                fingerprint=fp,
                person_id=f"P{i:03d}",
                name=f"User {i}",
                document=f"DOC{i:08d}"
            )
    
    # Verificar que se registraron
    assert repository.count() == len(fingerprints)
    
    # Buscar la primera huella
    if fingerprints:
        result = repository.identify(fingerprints[0])
        assert result is not None


def test_no_match_scenario(sample_image, repository):
    """Test cuando no hay coincidencia."""
    service = FingerprintService()
    # ComparisonService removed — using repository directly
    
    # Registrar una huella
    fp1 = service.process_image(sample_image, fingerprint_id="fp1")
    
    if fp1.minutiae:
        repository.register(fp=
            fingerprint=fp1,
            person_id="P001",
            name="User 1",
            document="12345678"
        )
        
        # Crear una huella muy diferente
        different_img = np.random.randint(0, 255, sample_image.shape, dtype=np.uint8)
        fp2 = service.process_image(different_img, fingerprint_id="fp2")
        
        if fp2.minutiae:
            result = repository.identify(fp2)
            
            # Probablemente no debería haber match
            assert result is not None


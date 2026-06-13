"""Benchmarks de performance."""

import pytest
import time
import numpy as np

from src.services.fingerprint_service import FingerprintService

from src.core.metrics import metrics


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics antes de cada test."""
    metrics.reset()
    yield
    metrics.reset()


def test_extraction_performance(sample_image):
    """Benchmark de extracción de minutiae."""
    service = FingerprintService()
    
    iterations = 10
    start = time.perf_counter()
    
    for i in range(iterations):
        fingerprint = service.process_image(sample_image)
    
    duration = (time.perf_counter() - start) * 1000
    avg_time = duration / iterations
    
    print(f"\nExtracción promedio: {avg_time:.2f}ms")
    
    # Objetivo: < 500ms por imagen
    assert avg_time < 500, f"Extracción demasiado lenta: {avg_time:.2f}ms"


def test_vectorization_performance():
    """Benchmark de vectorización."""
    from src.core.types import MinutiaCandidate, NormalizedFingerprint, MinutiaType, AlgorithmOrigin
    
    # Crear fingerprint con muchas minutiae
    minutiae = [
        MinutiaCandidate(i*10, i*10, float(i), MinutiaType.TERMINATION, 1.0, AlgorithmOrigin.SKELETON)
        for i in range(100)
    ]
    
    fingerprint = NormalizedFingerprint(id="test", minutiae=minutiae, width=200, height=200)
    
    iterations = 1000
    start = time.perf_counter()
    
    for _ in range(iterations):
        vector = fingerprint.vector
    
    duration = (time.perf_counter() - start) * 1000
    avg_time = duration / iterations
    
    print(f"\nVectorización promedio: {avg_time:.4f}ms")
    
    # Objetivo: < 5ms
    assert avg_time < 5, f"Vectorización demasiado lenta: {avg_time:.4f}ms"


def test_search_performance(sample_image, repository):
    """Benchmark de búsqueda en índice."""
    service = FingerprintService()
    # ComparisonService removed — using repository directly
    
    # Registrar varias huellas
    num_samples = 50
    
    for i in range(num_samples):
        fp = service.process_image(sample_image, fingerprint_id=f"fp_{i}")
        if fp.minutiae:
            repository.register(
        fp=fp,
                person_id=f"P{i:03d}",
                name=f"User {i}",
                document=f"DOC{i:08d}"
            )
    
    # Medir búsqueda
    fp_query = service.process_image(sample_image, fingerprint_id="query")
    
    if fp_query.minutiae:
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            result = repository.identify(fp_query)
        
        duration = (time.perf_counter() - start) * 1000
        avg_time = duration / iterations
        
        print(f"\nBúsqueda promedio (índice de {num_samples}): {avg_time:.2f}ms")
        
        # Objetivo: < 50ms para índice pequeño
        assert avg_time < 50, f"Búsqueda demasiado lenta: {avg_time:.2f}ms"


def test_end_to_end_latency(sample_image, repository):
    """Benchmark latencia end-to-end."""
    service = FingerprintService()
    # ComparisonService removed — using repository directly
    
    # Registrar una huella de referencia
    fp_ref = service.process_image(sample_image)
    if fp_ref.minutiae:
        repository.register(
            fingerprint=fp_ref,
            person_id="P001",
            name="Reference",
            document="12345678"
        )
    
    # Medir pipeline completo: procesar + identificar
    iterations = 10
    start = time.perf_counter()
    
    for _ in range(iterations):
        fp = service.process_image(sample_image)
        if fp.minutiae:
            result = repository.identify(fp)
    
    duration = (time.perf_counter() - start) * 1000
    avg_time = duration / iterations
    
    print(f"\nLatencia end-to-end promedio: {avg_time:.2f}ms")
    
    # Objetivo MVP: < 1000ms
    assert avg_time < 1000, f"Latencia E2E demasiado alta: {avg_time:.2f}ms"


def test_metrics_collection():
    """Verifica que las métricas se están recolectando."""
    service = FingerprintService()
    
    # Generar imagen simple
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Procesar
    fingerprint = service.process_image(img)
    
    # Verificar métricas
    stats = metrics.get_stats("process_image_full")
    
    if stats:
        print(f"\nMétricas recolectadas:")
        print(f"  - Ejecuciones: {stats['count']}")
        print(f"  - Promedio: {stats['mean']:.2f}ms")
        assert stats['count'] > 0

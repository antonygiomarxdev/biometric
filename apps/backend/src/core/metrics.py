"""
Instrumentación de performance en memoria.

NOTA ARQUITECTÓNICA: Este módulo es útil para benchmarks locales
y scripts de consola. En un entorno de producción distribuido
(múltiples workers de Uvicorn/Gunicorn), este estado en memoria
no se compartirá. Para producción real, migrar a OpenTelemetry.
"""

import time
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Colector de métricas de performance."""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
    
    def record(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registra una métrica de duración."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        entry = {
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
        if metadata:
            entry.update(metadata)
        
        self.metrics[operation].append(entry)
        logger.debug(f"{operation}: {duration_ms:.2f}ms {metadata or ''}")
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Obtiene estadísticas de una operación."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = [m["duration_ms"] for m in self.metrics[operation]]
        durations.sort()
        n = len(durations)
        
        return {
            "count": n,
            "mean": sum(durations) / n,
            "p50": durations[n // 2],
            "p95": durations[int(n * 0.95)] if n > 20 else durations[-1],
            "p99": durations[int(n * 0.99)] if n > 100 else durations[-1],
            "min": durations[0],
            "max": durations[-1],
        }
    
    def reset(self):
        """Limpia todas las métricas."""
        self.metrics.clear()


# Instancia global
metrics = PerformanceMetrics()


@contextmanager
def measure_time(operation: str, **metadata):
    """Context manager para medir tiempo de ejecución."""
    start = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.record(operation, duration_ms, metadata)


def timed(operation_name: Optional[str] = None):
    """Decorador para medir tiempo de funciones."""
    def decorator(func):
        op_name = operation_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with measure_time(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

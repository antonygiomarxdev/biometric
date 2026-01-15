
---

## ⚡ Nueva Arquitectura v2 (Clean Code + Hybrid Matching)

Hemos refactorizado el pipeline para maximizar el uso de tu RTX 4070 mediante una arquitectura híbrida y Clean Code.

### Mejoras Implementadas
1.  **Pipeline Híbrido**:
    - **CPU (ProcessPoolExecutor)**: Para saturar todos los núcleos en tareas de I/O y pre-procesamiento.
    - **GPU (CuPy)**: Convoluciones vectorizadas y filtros Gabor (sin bucles Python).
2.  **Matching Avanzado**:
    - **Distancia Coseno + L2**: Combinación ponderada para mayor precisión.
    - **Batch Processing**: Búsquedas paralelas usando `asyncio`.
3.  **Tipado Estricto**: Eliminación de `Any`, uso de interfaces `IEnhancer`, `IFeatureExtractor`.

### Benchmark Nuevo Pipeline

Ejecuta el script de validación para ver el rendimiento real:

```powershell
python scripts/benchmark_parallel.py
```

### Tiempos Esperados (Estimados)

| Métrica | Pipeline v1 (Serial) | Pipeline v2 (Paralelo/GPU) |
|---------|----------------------|----------------------------|
| Batch 100 Img | ~5 min | **~1-2 min** |
| Matching | L2 (Rápido) | Híbrido (Más preciso) |
| Uso CPU | 1 núcleo (15%) | Todos los núcleos (100%) |
| Uso GPU | Bajo | Optimizado (Vectorizado) |

### Diagrama de Flujo

Ver [PIPELINE_FLOW.md](docs/PIPELINE_FLOW.md) para el detalle completo de la arquitectura.

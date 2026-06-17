# Plan: Phase 1 — Investigación y Benchmark de Matching

**Goal:** Determinar el algoritmo de matching óptimo para uso forense
**Requirements:** AFIS-01, AFIS-02, AFIS-03

## Tasks

### T1: Descargar y preparar dataset SOCOFing
- Descargar SOCOFing desde Kaggle (ruizgara/socofing) vía API
- Extraer estructura: Real/, Altered/, segmentation masks
- Verificar metadatos (600 sujetos, 10 dedos cada uno, 3 alteraciones)
- Ubicar en `data/socofing/`
- **Output:** dataset listo para benchmark

### T2: Construir benchmark de accuracy (Qdrant actual)
- Leer benchmark existente en `apps/backend/scripts/benchmark.py`
- Crear script `scripts/benchmark_socofing.py` que:
  - Carga imágenes SOCOFing
  - Procesa con pipeline actual (normalize → enhance → extract → vectorize)
  - Para cada imagen: busca contra las demás
  - Mide rank-1 accuracy (genuino vs impostores)
  - Mide FNMR @ varias FMR thresholds
  - Mide tiempo de procesamiento por imagen
  - Mide tiempo de búsqueda por consulta
- Genera reporte JSON + plot

### T3: Evaluar impacto de alteraciones
- Separar benchmark para conjunto Real vs Altered/*/ (obliteration, rotation, z-cut)
- Medir degradación de accuracy por tipo/severidad
- **Output:** sección en reporte sobre robustness

### T4: (Opcional) Compilar y probar NIST NBIS
- Descargar NBIS desde NIST
- Compilar mindtct + bozorth3
- Extraer minutiae con mindtct en mismo dataset
- Correr matching con bozorth3
- Comparar accuracy vs Qdrant actual
- **Output:** tabla comparativa

### T5: Documentar recomendación final
- Comparar resultados en tabla:
  - Rank-1 accuracy (real vs altered)
  - FNMR @ FMR=1e-3, 1e-4
  - Tiempo de procesamiento
  - Tiempo de búsqueda (1k, 10k, 100k records)
- Decidir: mantener Qdrant, migrar a NBIS, o híbrido
- Escribir en SPEC.md o ADR la decisión
- **Output:** Decisión documentada con evidencia

## Success Criteria Check
1. ✅ Documento RESEARCH.md con comparación de enfoques (Qdrant vs NIST vs otros)
2. ❌ Benchmark cuantitativo ejecutado contra SOCOFing
3. ❌ Decisión documentada sobre qué algoritmo implementar
4. ❌ Reproducibilidad del benchmark (scripts + datos de prueba)

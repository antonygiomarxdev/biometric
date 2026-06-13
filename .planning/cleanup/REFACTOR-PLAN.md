---
phase: refactor-arquitectura
plan: 01
type: execute
status: draft
---

# Plan: Refactorización — Decisiones de Supervivencia

## Decisión 1: GPU/CUDA → ELIMINAR

`gpu_utils.py` con fallback `import numpy as cp` + `GpuEnhancer` con `pass` en resize.
- `create_enhancer()` chequea `GPUConfig.is_enabled()` que siempre es `False`
- GPU nunca se usa en la práctica
- Código GPU = dead code con coste de mantenimiento

**Acción:** 
- Eliminar `processing/enhancers/gpu.py`
- Simplificar `gpu_utils.py` — quitar CuPy, dejar solo CPU config
- `create_enhancer()` siempre retorna `CpuEnhancer`
- `fingerprint_service.py` deja de importar GPU

## Decisión 2: comparison_service.py → ELIMINAR

`ComparisonService` hace `register_fingerprint()` + `identify()`.
- Completamente reemplazado por `MatchingService.search_latent()` + `register_known()`
- Solo lo usan: `cli.py` y tests viejos
- Depende de `repository.py` y `storage` viejos

**Acción:**
- Eliminar `comparison_service.py` y `comparison_service` de `__init__.py`
- Actualizar `cli.py` para usar `MatchingService`
- Actualizar tests (`test_api_e2e.py`, `test_integration.py`, `test_performance.py`)

## Decisión 3: auth_service.py → REPARAR (arquitectura)

`auth_service.py` importa `from src.api.dependencies import get_db`.
- Violación de Clean Architecture (service → API)
- `get_db` es una dependencia de FastAPI, no de negocio

**Acción:**
- Mover la lógica de DB a `AuthRepository` o inyectar session
- `auth_service.py` recibe session por constructor/parámetro

## Decisión 4: Migraciones → REPARAR

`0002_add_users_table.py` y `0002_seed_data.py` tienen misma versión.

**Acción:**
- Renumerar `0002_seed_data.py` → `0003_seed_data.py`
- Verificar que Alembic reconoce ambas

## Decisión 5: Router → DB directo → REPARAR

Casi todos los routers importan `db.models` directamente. Deberían pasar por servicios.

**Acción (progresivo):**
- Crear `CaseService`, `EvidenceService`, `DecisionService` simples si hace falta
- O al menos crear repositorios dedicados
- No blocker — se puede hacer gradual

## Decisión 6: Frontend old components → MANTENER (temporal)

`ScannerPage.tsx`, `FingerprintViewer.tsx`, etc. son del UI viejo.
- Nuevo UI (Dashboard, ComparisonView) convive con viejo
- Se reemplazarán en fases futuras de frontend

**Acción:** No tocar ahora. Reemplazar cuando se haga UI phase.

## Decisión 7: Root scripts → REVISAR uno por uno

| Script | Decisión |
|--------|----------|
| `scripts/benchmark_parallel.py` | Mantener (referencia, no duplicado) |
| `scripts/identify_single.py` | Actualizar para usar MatchingService |
| `scripts/load_socofing.py` | Mantener |
| `scripts/quick_test.py` | Mantener |
| `scripts/test_socofing.py` | Unificar con benchmark_soco.py? |
| `scripts/visualize_minutiae.py` | Mantener |

---

## Waves de Ejecución

```
Wave 1: GPU cleanup (eliminar gpu.py, simplificar gpu_utils)
Wave 2: Eliminar comparison_service + actualizar CLI/tests
Wave 3: Arreglar auth_service architecture violation
Wave 4: Arreglar migraciones (0002 → 0003)
Wave 5: Consolidar scripts (solo updates mínimos)
```

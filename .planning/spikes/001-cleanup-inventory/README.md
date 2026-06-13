---
spike: 001
name: cleanup-inventory
type: standard
validates: "Given the biometric codebase, when analyzed for dead code, naming issues, and quality problems, then a complete inventory is produced"
verdict: VALIDATED
tags: [cleanup, naming, dead-code, spanish]
---

# Spike 001: Cleanup Inventory

## Results

### Dead Code to Delete
- `apps/backend/src/services/biometrics/` (5 files) — entire package unused
- `apps/backend/migrations/` (2 raw SQL files) — replaced by Alembic
- `scripts/benchmark_socofing.py` — duplicate of `apps/backend/scripts/benchmark_soco.py`
- `scripts/benchmark.py`, `e2e_test.py` — orphaned root-level scripts

### Duplicate Model
- `FingerprintVector` in both `db/models.py:124` and `storage/vector_index.py:22`

### Spanish Comments (High Priority)
- `storage/repository.py`, `storage/vector_index.py`, `services/fingerprint_service.py`, `processing/enhancers/gpu.py`, `processing/enhancers/cpu.py`, `core/gpu_utils.py`

### Spanish Docstrings
- `core/__init__.py`, `processing/__init__.py`, `services/__init__.py`, `storage/__init__.py`

### Spanish Function Names
- `generar_dictamen` in `dictamenes.py:30`
- `list_evidencias` in `evidencias.py:129`
- `_require_perito_role` in `decisiones.py:87`

### `Any` / bare dict
- 16 files with `dict[str, Any]` or similar
- 4 files with bare `dict` without generics

### Unused Imports
- `Any` in 3 files, `asyncio` in 1 file

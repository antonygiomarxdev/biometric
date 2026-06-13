---
title: Codebase Cleanup — Naming, Dead Code & Conventions
status: ready
created: 2026-06-13
based_on: .planning/spikes/001-cleanup-inventory/README.md
---

# Cleanup Plan: Codebase Hygiene

## Overview

Apply conventions from AGENTS.md retroactively: English-only code, proper typing, remove dead code. All router files, API paths, function names, comments, and docstrings to be renamed to English.

---

## Wave 1: Delete Dead Code

### Task 1.1 — Remove unused biometrics/ package
- apps/backend/src/services/biometrics/__init__.py
- apps/backend/src/services/biometrics/base.py
- apps/backend/src/services/biometrics/factory.py
- apps/backend/src/services/biometrics/providers/__init__.py
- apps/backend/src/services/biometrics/providers/face.py
- apps/backend/src/services/biometrics/providers/fingerprint.py

### Task 1.2 — Remove raw SQL migrations (replaced by Alembic)
- apps/backend/migrations/001_init_pgvector.sql
- apps/backend/migrations/002_add_image_path.sql

### Task 1.3 — Remove orphaned root scripts
- scripts/benchmark_socofing.py (duplicate)
- benchmark.py, e2e_test.py

### Task 1.4 — Remove stub files
- apps/backend/src/db/migrations/versions/.gitkeep

---

## Wave 2: Consolidate Duplicate Model

Unify FingerprintVector: remove from vector_index.py, import from db/models.py

---

## Wave 3: Rename Files (Spanish → English)

| Current | New | API Prefix |
|---------|-----|------------|
| auditoria.py | audit.py | /api/v1/audit |
| decisiones.py | decisions.py | /api/v1/decisions |
| dictamenes.py | reports.py | /api/v1/reports |
| evidencias.py | evidence.py | /api/v1/evidence |
| huellas_conocidas.py | known_fingerprints.py | /api/v1/known-fingerprints |

Impact: __init__.py, main.py, tests, frontend API calls.

---

## Wave 4: Rename Functions (Spanish → English)

- dictamenes.py:30: generar_dictamen() → generate_report()
- evidencias.py:129: list_evidencias() → list_evidence()
- decisiones.py:87,166: _require_perito_role() → _require_examiner_role()

---

## Wave 5: Translate Comments & Docstrings

- 4 __init__.py files (core, processing, services, storage)
- ~80 lines Spanish comments in 8 source files
- Router module docstrings
- pdf_generator.py:190 Spanish HTML template

---

## Wave 6: Type Annotations Cleanup

- Fix bare dict without generics (4 files)
- Remove unused imports (4 files)

---

## Order: Wave 1 → 2 → 5 → 6 → 3 → 4

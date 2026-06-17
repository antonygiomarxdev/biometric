# Phase 19: Naming Convention Cleanup

## Goal

Eliminate naming inconsistencies across the backend so every developer
(and every AI agent) can navigate the codebase predictably.

## Conventions to enforce

1. **Async public API**: Only `search()`, never `search_async()`.
   Sync versions become `_search_sync()` private.
2. **Router prefix**: Centralize `/api/v1` in `main.py`.
   Routers only declare their resource path (`/cases`, `/persons`).
3. **Session parameter**: Always `session`, never `db`.
4. **Route paths**: Always relative to the router prefix, never absolute.
5. **File = Class**: Each service file exports exactly one main class,
   filename matches the class (snake_case).
6. **Constructor DI**: Every service receives its dependencies via
   `__init__`, never creates them internally.
7. **Private attrs `_`**: All instance attributes of a service are `self._x`.
8. **Test mocks**: Always `mock_session` for the fake `AsyncSession`.
9. **Language (per AGENTS.md)**: Code, comments, docstrings, identifiers,
   log messages → English. Only user-facing error responses → Spanish.

## Tasks

### Wave 1: Rename async methods (no logic change)
- [ ] `QdrantRagMatchingService`: `search()` → `_search_sync()`, `search_async()` → `search()`
- [ ] `FingerprintService`: `process_image()` → `_process_sync()`, `process_image_async()` → `process_image()`
- [ ] `auth_service.py`: `verify_password_async()` → `verify_password()` (make it async, drop sync version), `get_password_hash_async()` → `get_password_hash()` (make it async)
- [ ] Update all callers

### Wave 2: Centralize API version prefix
- [ ] Remove `prefix="/api/v1/..."` from every router
- [ ] In `main.py`, create `api_v1 = APIRouter(prefix="/api/v1")`, mount all routers under it
- [ ] Fix all route decorators to use relative paths (e.g. `@router.get("/{case_id}")` instead of `@router.get("/api/v1/cases/{case_id}")`)

### Wave 3: Standardize session parameter name
- [ ] Rename all `db: AsyncSession` to `session: AsyncSession`
- [ ] Rename all `db: Session` (deprecated sync) to `session: Session`
- [ ] Update all tests

### Wave 4: File naming & service DI
- [ ] Rename `rag_matching_service.py` → `qdrant_rag_matching_service.py`
- [ ] Rename `fingerprint_enrollment_service.py` → `fingerprint_enrollment_service.py` (already correct)
- [ ] Audit all services: ensure deps come via `__init__`, not created internally
- [ ] Standardize `self._x` prefix on all private service attributes

### Wave 5: Test naming consistency
- [ ] Rename `mock_db` → `mock_session` across all test files
- [ ] Rename `mock_sync` → `mock_sync_session`
- [ ] Standardize test helper names

### Wave 6: Language audit (per AGENTS.md)
- [ ] User-facing error messages (`detail="..."`) → Spanish
- [ ] Code, comments, docstrings, identifiers → English (audit for stray Spanish)
- [ ] Log messages → English

## Acceptance Criteria

- [ ] Zero `_async` suffix on any public method (only private `_sync`)
- [ ] `/api/v1` appears exactly once in the entire codebase (in `main.py`)
- [ ] `db: AsyncSession` has zero occurrences (only `session`)
- [ ] All route decorators use relative paths
- [ ] All test files use `mock_session`
- [ ] User-facing error messages are in Spanish
- [ ] Code, comments, and docstrings are in English
- [ ] All 707+ tests pass

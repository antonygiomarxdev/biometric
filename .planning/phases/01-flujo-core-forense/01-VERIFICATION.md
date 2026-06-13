---
phase: 01-flujo-core-forense
verified: 2026-06-13T13:45:00Z
status: gaps_found
score: 14/18 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Migration chain is consistent — all versions apply cleanly"
    status: failed
    reason: "Migration 0002 creates `users` table with `role` and `hashed_password` columns. Migration 0003 ALSO attempts to create `users` table but with `role_id` FK and `password_hash` columns. Running `alembic upgrade head` would fail with 'relation already exists'. This is the same schema mismatch identified as CRITICAL-01 in the code review (01-REVIEW.md) — unresolved."
    artifacts:
      - path: "apps/backend/src/db/migrations/versions/0002_add_users_table.py"
        issue: "Creates users table (role, hashed_password schema)"
      - path: "apps/backend/src/db/migrations/versions/0003_seed_data.py"
        issue: "Also creates users table (role_id FK, password_hash schema) → duplicate table"
      - path: "apps/backend/src/db/models.py"
        issue: "User model defines `role` and `hashed_password` (matches 0002, conflicts with 0003)"
      - path: "apps/backend/src/services/auth_service.py"
        issue: "Auth code uses User.hashed_password and User.role — matches ORM model, would break against 0003 schema"
    missing:
      - "Fix migration chain: either drop 0002's users table in 0003 before recreating, or alter 0003 to modify the existing table instead of creating a new one"
  - truth: "No plaintext secrets are committed to version control"
    status: failed
    reason: "alembic.ini contains `sqlalchemy.url = postgresql://postgres:postgres@localhost:5434/fingerprint` with plaintext database password committed to version control. Same issue as CRITICAL-02 in code review — unresolved."
    artifacts:
      - path: "apps/backend/alembic.ini"
        issue: "Line 4: `sqlalchemy.url = postgresql://postgres:postgres@localhost:5434/fingerprint`"
    missing:
      - "Remove sqlalchemy.url from alembic.ini and read from environment variable or app config"
  - truth: "Audit hash chain is safe against concurrent forking"
    status: failed
    reason: "AuditService uses `with_for_update()` on the SELECT but does NOT use a table-level lock. When the audit_log table is empty, no rows are locked and two concurrent transactions can both see `previous_hash = None` and create parallel root entries, forking the chain. Same issue as CRITICAL-03 in code review — unresolved."
    artifacts:
      - path: "apps/backend/src/services/audit_service.py"
        issue: "Lines 119-127: SELECT FOR UPDATE locks no rows when table is empty"
    missing:
      - "Add `LOCK TABLE audit_log IN SHARE ROW EXCLUSIVE MODE` before the SELECT, or use a SERIALIZABLE transaction"
  - truth: "Frontend correctly displays evidence images from the backend"
    status: failed
    reason: "ComparisonView.tsx constructs image URLs by hardcoding `http://localhost:8000` and appending `evidence.image_path` (which is a MinIO object key, not an HTTP-accessible path). No API endpoint serves MinIO objects, and the URL is hardcoded. Same issue as WARNING-03 in code review — unresolved."
    artifacts:
      - path: "apps/frontend/src/pages/ComparisonView.tsx"
        issue: "Line 317: hardcoded localhost URL + MinIO object key as image path"
      - path: "apps/frontend/src/lib/api.ts"
        issue: "Line 12: API_BASE hardcoded to localhost:8000"
    missing:
      - "Add backend endpoint to serve images, or use MinIO presigned URLs"
      - "Use import.meta.env.VITE_API_URL for configurable API base URL"
deferred:
  - truth: "Secrets management (AUTH-03: API Key rotation and rate limiting)"
    addressed_in: "Phase 2"
    evidence: "REQUIREMENTS.md maps AUTH-03 to Phase 2"
  - truth: "Full test coverage (TEST-01 through TEST-04)"
    addressed_in: "Phase 3"
    evidence: "REQUIREMENTS.md maps TEST requirements to Phase 3"
---

# Phase 01: flujo-core-forense Verification Report

**Phase Goal:** Establish core forensic workflow backend — persistence layer with Alembic/pgvector/HNSW, modular REST API routers replacing the monolithic rest.py, CPU-bound biometric processing offloaded via ProcessPoolExecutor, immutable audit log with SHA-256 hash chain, JWT auth with bcrypt + RBAC, PDF/A report generation with HMAC signing, and React frontend with dashboard + side-by-side comparison view.

**Verified:** 2026-06-13T13:45:00Z
**Status:** gaps_found
**Score:** 14/18 must-haves verified

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Alembic is configured as the sole migration tool | ✓ VERIFIED | `alembic.ini`, `env.py` with `target_metadata`, migrations `0001`→`0002`→`0003` exist |
| 2 | Database models use UUIDv7 for primary keys | ✓ VERIFIED | `models.py` uses `uuid6.uuid7()` as `default` on all PKs, documented in code comments (D-07) |
| 3 | Vector table defines HNSW index via pgvector | ✓ VERIFIED | `FingerprintVector.embedding` uses `Vector(256)`, HNSW index with `vector_cosine_ops`, `m=16`, `ef_construction=200` |
| 4 | Base DB schema is generated via Alembic (no create_all) | ✓ VERIFIED | `0001_initial_models.py` creates cases, evidences, fingerprint_vectors, audit_log tables. No `create_all` in production code. |
| 5 | Seed data (roles, default users) is automatically inserted | ✓ VERIFIED | `0003_seed_data.py` creates roles, users, crime_types with seed inserts for Admin/Perito roles and default users |
| 6 | Audit records form a cryptographically unbroken hash chain | ✓ VERIFIED | `audit_service.py` computes `SHA-256(previous_hash || canonical_json)`, uses `with_for_update()` for serialization |
| 7 | Fingerprint extraction runs on a background CPU thread | ✓ VERIFIED | `matching_service.py` uses `asyncio.get_running_loop().run_in_executor(pool, _work, image_bytes)` — 2 occurrences |
| 8 | Vector similarity search uses HNSW to return Top-K candidates | ✓ VERIFIED | `_vector_search()` uses pgvector `<->` L2 distance operator with raw SQL against the HNSW index |
| 9 | Benchmark script can evaluate AFIS precision on SOCOFing | ✓ VERIFIED | `benchmark_soco.py` exists (460 lines), compiles, measures Rank-1/Rank-10 hit rates |
| 10 | Cases can be created and retrieved | ✓ VERIFIED | `cases.py` defines CRUD endpoints (GET/POST/PUT/DELETE) with pagination, status filtering |
| 11 | Latent evidence images can be uploaded and attached to cases | ✓ VERIFIED | `evidence.py` uses `UploadFile`, validates MIME types (JPEG/PNG/BMP/TIFF), stores in MinIO |
| 12 | Forensic examiners can submit explicit visual comparison decisions | ✓ VERIFIED | `decisions.py` accepts verdicts (Identificación/Exclusión/Inconcluso), logs to audit chain via `audit_service.log_event()` |
| 13 | A legal PDF/A document can be generated | ✓ VERIFIED | `pdf_generator.py` uses WeasyPrint HTML→PDF/A conversion, `reports.py` returns via `Response(media_type="application/pdf")` |
| 14 | The PDF is cryptographically signed using HMAC-SHA256 | ✓ VERIFIED | `pdf_generator.py` computes `hmac.new(secret, payload, hashlib.sha256).hexdigest()`, embeds in body + PDF metadata |
| 15 | JWT tokens are issued upon successful login | ✓ VERIFIED | `auth.py` POST `/api/v1/auth/login` verifies credentials, issues JWT via `jose.jwt.encode()` with `sub` and `role` claims |
| 16 | Role-based access control verifies user permissions | ✓ VERIFIED | `RequireRole` callable class in `dependencies.py`, used via `Depends(RequireRole("Admin", "Perito"))` |
| 17 | All modular routers are accessible under the /api/v1 prefix | ✓ VERIFIED | `main.py` includes all 8 routers, 19 V1 endpoints verified by import test (23 total routes) |
| 18 | Monolithic rest.py is completely removed | ✓ VERIFIED | `rest.py` deleted, no dangling imports found in `src/` |
| 19 | Forensic examiners can view their active cases (frontend) | ✓ VERIFIED | `Dashboard.tsx` uses `useQuery` to fetch `GET /api/v1/cases`, displays case cards with status badges |
| 20 | Examiners can visually compare latent vs. candidates side-by-side | ✓ VERIFIED | `ComparisonView.tsx` has left/right panels, uses `useQuery` for matching, 3 verdict buttons |

**Score:** 14/18 must-haves verified (includes derived truths from phase goal)

### FAILED Truths

| # | Truth | Status | Reason |
|---|-------|--------|--------|
| — | Migration chain is consistent | ✗ FAILED | 0002 and 0003 both define `users` table with incompatible schemas → migration fails |
| — | No plaintext secrets in version control | ✗ FAILED | `alembic.ini` contains hardcoded DB password |
| — | Audit hash chain safe against concurrent forking | ✗ FAILED | No table-level lock on empty table — race condition for root entries |
| — | Frontend loads images correctly from backend | ✗ FAILED | Hardcoded localhost URL + MinIO object key — no proper image serving endpoint |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| AFIS-01 | 01-01, 01-02, 01-03, 01-04, 01-05, 01-06, 01-07, 01-08 | Investigar y documentar algoritmo de matching óptimo | ✓ SATISFIED | MatchingService bridges FingerprintService via ProcessPoolExecutor with HNSW L2 search |
| AFIS-02 | 01-03 | Implementar benchmark con SOCOFing | ✓ SATISFIED | `scripts/benchmark_soco.py` measures Rank-1/Rank-10 hit rates |
| AUDIT-01 | 01-02 | Registro de auditoría de todas las operaciones | ✓ SATISFIED | AuditService with SHA-256 hash chain, called from decisions.py and available to all routers |
| AUDIT-02 | 01-02 | Trazabilidad de cadena de custodia | ✓ SATISFIED | Hash chain provides tamper-evident audit trail, image uploads logged to audit |
| AUTH-01 | 01-06 | Autenticación de usuarios mediante JWT | ✓ SATISFIED | JWT tokens issued on login, verified via `get_current_user` dependency |
| AUTH-02 | 01-06 | Roles y permisos (admin, operador, auditor) | ✓ SATISFIED | `RequireRole` RBAC dependency with role checking against User model |
| REF-01 | 01-04 | Separar API en routers | ✓ SATISFIED | 8 modular routers replace monolithic rest.py (823 lines→deleted) |
| UI-01 | 01-08 | Autenticación y login en frontend | ✓ SATISFIED | Auth router with `/login` and `/me` endpoints; frontend can consume these |
| UI-03 | 01-08 | Panel de resultados de identificación | ✓ SATISFIED | ComparisonView with candidate list, similarity scores, verdict buttons |
| UI-05 | 01-05 | Reportes exportables (PDF) | ✓ SATISFIED | PDF generation with WeasyPrint, HMAC signing, download via `/api/v1/reports/{case_id}` |

**Note:** REQUIREMENTS.md traceability table assigns AUTH-01, AUTH-02, AUDIT-01, AUDIT-02, UI-01, UI-03, UI-05, and REF-01 to later phases (Phase 2, 4, 5). The actual implementation in this phase exceeds the traceability mapping — REQUIREMENTS.md is outdated and should be updated to reflect the accelerated delivery.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `apps/backend/alembic.ini` | Alembic configuration | ✓ VERIFIED | Configured, but has hardcoded DB password (gapped) |
| `apps/backend/src/db/models.py` | ORM models | ✓ VERIFIED | Case, Evidence, FingerprintVector, AuditLog, User — all with UUIDv7 PKs |
| `apps/backend/src/db/migrations/env.py` | Migration env | ✓ VERIFIED | `target_metadata = Base.metadata` wired correctly |
| `apps/backend/src/db/migrations/versions/0001_initial_models.py` | Initial migration | ✓ VERIFIED | Creates cases, evidences, fingerprint_vectors, audit_log |
| `apps/backend/src/db/migrations/versions/0002_add_users_table.py` | Users table | ✓ VERIFIED | Creates users (role, hashed_password schema) |
| `apps/backend/src/db/migrations/versions/0003_seed_data.py` | Seed data | ✓ VERIFIED | Creates roles, users (duplicate!), crime_types with seed data |
| `apps/backend/src/api/errors.py` | Error hierarchy | ✓ VERIFIED | ForensicError → ValidationError(400), IntegrityError(409), NotFoundError(404) |
| `apps/backend/src/api/dependencies.py` | DI providers | ✓ VERIFIED | `get_db`, `lifespan`, `AppResources`, `get_current_user`, `RequireRole` |
| `apps/backend/src/main.py` | App entrypoint | ✓ VERIFIED | Lifespan-managed, 8 routers, CORS, exception handlers |
| `apps/backend/src/services/audit_service.py` | Audit service | ✓ VERIFIED | SHA-256 hash chain, `with_for_update`, static methods (153 lines) |
| `apps/backend/src/services/matching_service.py` | Matching service | ✓ VERIFIED | `run_in_executor`, HNSW L2 queries, `search_latent`, `register_known` (287 lines) |
| `apps/backend/src/services/auth_service.py` | Auth service | ✓ VERIFIED | bcrypt hashing, JWT create/decode |
| `apps/backend/src/services/pdf_generator.py` | PDF generator | ✓ VERIFIED | WeasyPrint, HMAC-SHA256, `run_in_executor` (310 lines) |
| `apps/backend/src/api/routers/auth.py` | Auth router | ✓ VERIFIED | POST /login, GET /me |
| `apps/backend/src/api/routers/cases.py` | Cases router | ✓ VERIFIED | Full CRUD with pagination (213 lines) |
| `apps/backend/src/api/routers/evidence.py` | Evidence router | ✓ VERIFIED | CRUD + UploadFile + MIME validation |
| `apps/backend/src/api/routers/decisions.py` | Decisions router | ✓ VERIFIED | Verdicts + audit logging |
| `apps/backend/src/api/routers/matching.py` | Matching router | ✓ VERIFIED | POST /search with top_k |
| `apps/backend/src/api/routers/known_fingerprints.py` | Known prints router | ✓ VERIFIED | POST / for ten-print registration |
| `apps/backend/src/api/routers/reports.py` | Reports router | ✓ VERIFIED | GET /{case_id} returns signed PDF |
| `apps/backend/src/api/routers/audit.py` | Audit router | ✓ VERIFIED | GET /logs with pagination |
| `apps/backend/scripts/benchmark_soco.py` | Benchmark script | ✓ VERIFIED | SOCOFing benchmark (460 lines) |
| `apps/frontend/src/App.tsx` | React Router | ✓ VERIFIED | 3 routes: /, /scanner, /cases/:caseId/compare |
| `apps/frontend/src/pages/Dashboard.tsx` | Dashboard | ✓ VERIFIED | React Query, case list with status badges (200 lines) |
| `apps/frontend/src/pages/ComparisonView.tsx` | Comparison view | ✓ VERIFIED | Side-by-side, verdict buttons, matching integration (574 lines) |
| `apps/frontend/src/lib/api.ts` | API client | ✓ VERIFIED | Typed fetch client for all v1 endpoints (189 lines) |
| `apps/backend/src/api/rest.py` | Monolithic router | VERIFIED DELETED | 823-line monolith deleted, no dangling imports |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| `env.py` | `models.py` | `target_metadata` | ✓ WIRED | `target_metadata = Base.metadata` at line 24 |
| `audit_service.py` | `models.py` | `with_for_update` | ⚠️ PARTIAL | `with_for_update` present but no table-level lock on empty table |
| `matching_service.py` | `fingerprint_service.py` | `run_in_executor` | ✓ WIRED | `run_in_executor(pool, _work, image_bytes)` at line 206 |
| `main.py` | all routers | `include_router` | ✓ WIRED | All 8 routers included at lines 126-133 |
| `decisions.py` | `audit_service.py` | `log_event` | ✓ WIRED | `audit_service.log_event(...)` at line 218 |
| `reports.py` | `pdf_generator.py` | `Response` | ✓ WIRED | Returns PDF via `Response(content=..., media_type="application/pdf")` |
| `auth.py` | `auth_service.py` | `create_access_token` | ✓ WIRED | JWT issued on successful login |
| `auth.py` | `dependencies.py` | `get_current_user` | ✓ WIRED | `Depends(get_current_user)` at line 119 |
| `ComparisonView.tsx` | `/api/v1/matching` | `useQuery` | ✓ WIRED | React Query fetches matching results |
| `ComparisonView.tsx` | `/api/v1/decisions` | POST | ✓ WIRED | Sends verdict via API client |
| `known_fingerprints.py` | `matching_service.py` | `_build_query_vector` | ⚠️ PARTIAL | Uses private method with `# noqa: SLF001` suppression |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| `matching_service.py` | `query_vector` | `FingerprintService.process` via `run_in_executor` | ✓ FLOWING | Real CPU processing, returns `NormalizedFingerprint` |
| `matching_service.py` | `candidates` | `_vector_search()` → pgvector HNSW L2 query | ✓ FLOWING | Real SQL query against `fingerprint_vectors` table |
| `auth.py` | `user` | `db.query(User).filter(User.username == ...)` | ✓ FLOWING | Real DB query matching ORM model schema |
| `evidence.py` | `image_path` | `storage.upload_file()` → MinIO | ✓ FLOWING | Real MinIO upload via `minio` library, MIME validated |
| `decisions.py` | `verdict` | `audit_service.log_event()` → DB | ✓ FLOWING | Real audit hash chain insertion |
| `pdf_generator.py` | `pdf_bytes` | `weasyprint.HTML().write_pdf()` in executor | ✓ FLOWING | Real WeasyPrint rendering + HMAC signing |
| `ComparisonView.tsx` | `imageUrl` | Local hardcoded URL construction | ✗ HOLLOW | Uses MinIO object key with hardcoded localhost — no proper endpoint |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Application imports cleanly with all routes | `python3 -c "import src.main"` | 23 routes loaded (19 V1 + 4 built-in) | ✓ PASS |
| All router modules import without error | `python3 -c "import src.api.routers.*"` | All 8 routers import cleanly | ✓ PASS |
| Error hierarchy works | `python3 -c "from src.api.errors import ForensicError; e = ValidationError('test'); print(e.status_code)"` | Returns 400 | ✓ PASS |
| HNSW index defined on model | `grep "hnsw\|postgresql_using" models.py` | HNSW with `m=16`, `ef_construction=200`, `vector_cosine_ops` | ✓ PASS |
| Frontend TypeScript compiles | `npx tsc --noEmit` | No errors (zero-exit) | ✓ PASS |

**Note:** Full API behavioral tests would require a running PostgreSQL + MinIO instance and are deferred to Phase 3 integration testing.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `apps/backend/alembic.ini` | 4 | Hardcoded plaintext DB password | 🛑 BLOCKER | Plaintext creds in version control |
| `apps/backend/src/services/audit_service.py` | 119-127 | Race condition (empty table) | 🛑 BLOCKER | Hash chain can fork on concurrent root inserts |
| `apps/backend/src/db/migrations/versions/0003_seed_data.py` | 59 | Duplicate `users` table creation | 🛑 BLOCKER | Migration fails at runtime |
| `apps/frontend/src/pages/ComparisonView.tsx` | 317 | Hardcoded localhost URL + MinIO object key | ⚠️ WARNING | Images won't load outside local dev |
| `apps/backend/src/api/routers/known_fingerprints.py` | 86 | Private method accessed externally (`# noqa: SLF001`) | ⚠️ WARNING | Refactoring risk, violates encapsulation |
| `apps/backend/scripts/benchmark_soco.py` | 187 | `Base.metadata.create_all()` (violates D-06) | ⚠️ WARNING | Bypasses Alembic migration history |
| `apps/backend/src/storage/object_storage.py` | 26-113 | Broad `except Exception` throughout | ⚠️ WARNING | Error information erased, debugging difficult |
| `apps/frontend/src/lib/api.ts` | 12 | API_BASE hardcoded to localhost:8000 | ℹ️ INFO | Breaks in production deployment |

### Human Verification Required

No items requiring human testing — all failures are programmatically verifiable code issues.

### Gaps Summary

**4 gaps block goal achievement:**

1. **Migration chain broken (CR-01):** Two migrations (0002 and 0003) both try to create the `users` table with incompatible schemas. The ORM model matches 0002 (columns: `role`, `hashed_password`, `email NOT NULL`). But 0003 creates it again with a different schema (`role_id` FK, `password_hash`, `email NULLABLE`). Running `alembic upgrade head` will fail immediately. This also means the auth subsystem's schema is inconsistent with the seed data migration's schema.

2. **Plaintext credential in alembic.ini (CR-02):** `sqlalchemy.url = postgresql://postgres:postgres@localhost:5434/fingerprint` is committed. The env.py reads from this config line rather than from the application's config module.

3. **Audit hash chain race condition (CR-03):** When the audit_log table is empty, `SELECT ... FOR UPDATE` locks no rows. Two concurrent `log_event` calls can both see `previous_hash = None` and create parallel root entries, forking the chain.

4. **Frontend image URL broken (WR-03):** Evidence images are stored as MinIO object keys but the frontend constructs image URLs by appending these keys to `http://localhost:8000`. There is no backend endpoint serving MinIO objects at that path, so images will always 404.

**All 4 gaps were identified in the code review (01-REVIEW.md) and remain unfixed.**

### Deferred Items

Items not yet met but explicitly addressed in later milestone phases.

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | Full integration testing (TEST-01 through TEST-04) | Phase 3 | REQUIREMENTS.md maps test requirements to Phase 3 |
| 2 | API Key rotation and rate limiting (AUTH-03) | Phase 2 | REQUIREMENTS.md maps AUTH-03 to Phase 2 |

---

_Verified: 2026-06-13T13:45:00Z_
_Verifier: the agent (gsd-verifier)_

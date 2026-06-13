---
phase: 01-flujo-core-forense
reviewed: 2026-06-13T12:00:00Z
depth: standard
files_reviewed: 26
files_reviewed_list:
  - apps/backend/alembic.ini
  - apps/backend/scripts/benchmark_soco.py
  - apps/backend/src/api/dependencies.py
  - apps/backend/src/api/errors.py
  - apps/backend/src/api/routers/__init__.py
  - apps/backend/src/api/routers/auditoria.py
  - apps/backend/src/api/routers/auth.py
  - apps/backend/src/api/routers/cases.py
  - apps/backend/src/api/routers/decisiones.py
  - apps/backend/src/api/routers/dictamenes.py
  - apps/backend/src/api/routers/evidencias.py
  - apps/backend/src/api/routers/huellas_conocidas.py
  - apps/backend/src/api/routers/matching.py
  - apps/backend/src/db/migrations/env.py
  - apps/backend/src/db/migrations/versions/0001_initial_models.py
  - apps/backend/src/db/migrations/versions/0002_seed_data.py
  - apps/backend/src/db/models.py
  - apps/backend/src/main.py
  - apps/backend/src/services/audit_service.py
  - apps/backend/src/services/auth_service.py
  - apps/backend/src/services/fingerprint_service.py
  - apps/backend/src/services/matching_service.py
  - apps/backend/src/services/pdf_generator.py
  - apps/backend/src/storage/object_storage.py
  - apps/frontend/src/App.tsx
  - apps/frontend/src/pages/ComparisonView.tsx
  - apps/frontend/src/pages/Dashboard.tsx
findings:
  critical: 3
  warning: 4
  info: 5
  total: 12
status: issues_found
---

# Phase 01: Code Review Report — flujo-core-forense

**Reviewed:** 2026-06-13T12:00:00Z
**Depth:** standard
**Files Reviewed:** 27
**Status:** issues_found

## Summary

Review of 27 source files implementing the core forensic fingerprint workflow: ORM models, Alembic migrations, REST API routers, services (auth, matching, audit, PDF generation), and frontend components (Dashboard, ComparisonView).

**Key concerns:**
1. **CRITICAL - Schema mismatch:** The `User` ORM model and migration `0002` define incompatible table schemas (`role` vs `role_id`, `hashed_password` vs `password_hash`). The auth subsystem cannot function against a database created by these migrations.
2. **CRITICAL - Hardcoded credentials:** `alembic.ini` contains a plaintext database password. Several default secrets in `config.py` are change-me placeholders that could be missed in production.
3. **WARNING - Race condition:** `AuditService.log_event` can fork the hash chain on concurrent calls when the audit table is empty.
4. **WARNING - Encapsulation violation and broken image loading in frontend.**

---

## Critical Issues

### CR-01: Schema Mismatch — User Model vs Migration 0002

**File:** `apps/backend/src/db/models.py:178-224` and `apps/backend/src/db/migrations/versions/0002_seed_data.py:59-95`

**Issue:** The `User` ORM model and the `0002` migration define **incompatible** schemas for the `users` table. Every auth operation that uses the ORM model will fail against a database created by these migrations.

| Attribute | `models.py` (ORM) | Migration `0002` |
|-----------|-------------------|-------------------|
| Role | `role: str` (Column `String(20)`) | `role_id` (FK → `roles.id`) |
| Password | `hashed_password: str` (Column `String(255)`) | `password_hash: str` (Column `String(256)`) |
| Email | `email: str` (Column `String(255)`, NOT NULL) | `email: str` (Column `String(200)`, NULLABLE) |
| Role FK | No `role_id` column | `role_id UUID NOT NULL` |

**Impact (BLOCKER):**
- `auth.py:87` calls `verify_password(form_data.password, user.hashed_password)` — column `hashed_password` does not exist in the actual DB → SQL error on every login attempt.
- `auth.py:102` creates JWT with `user.role` — column `role` does not exist in the actual DB → SQL error.
- `auth_service.py:173` (`RequireRole`) checks `current_user.role` — same issue.
- Seed data inserts `password_hash` values into a column the ORM model can never read.
- Two separate `role` systems coexist (string column ORM-side vs FK to `roles` table migration-side) — every role check will fail.

**Fix:**
Choose one schema and make it consistent. Two options:

**Option A (align migration to model):** Remove the `roles` table from migration 0002 and add `role`, `hashed_password` columns matching the model:
```python
# In migration 0002, replace users table creation with:
op.add_column("users", sa.Column("role", sa.String(20), nullable=False, server_default="Perito"))
# and ensure hashed_password matches the model
```

**Option B (align model to migration):** Update the User model to match the FK-based schema:
```python
class User(Base):
    __tablename__ = "users"
    ...
    role_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("roles.id", ondelete="RESTRICT"),
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column("password_hash", String(256), nullable=False)
    role: Mapped["Role"] = relationship("Role")
```
This also requires creating a `Role` ORM model and updating all auth code to read role via the relationship.

**Recommendation:** Option B is architecturally superior (normalized roles enable future RBAC). But it changes the entire auth layer. If minimal fix is needed, Option A is faster.

---

### CR-02: Plaintext Database Password in `alembic.ini`

**File:** `apps/backend/alembic.ini:4`

**Issue:** The Alembic configuration file contains a hardcoded database URL with plaintext credentials:

```ini
sqlalchemy.url = postgresql://postgres:postgres@localhost:5434/fingerprint
```

This file is committed to version control and accessible to anyone with repo access. The password `postgres` for the `postgres` superuser is particularly dangerous — it's the default PostgreSQL superuser account.

**Fix:**
Remove the `sqlalchemy.url` line from `alembic.ini` and rely on the `DATABASE_URL` environment variable at migration runtime. The `env.py` migration script already imports from `src.core.config` — wire it to read the URL from config instead:

```python
# In env.py, replace:
# url = config.get_main_option("sqlalchemy.url")
# With:
from src.core.config import config as app_config
url = app_config.database_url
```

Or if keeping the ini-based approach, use an environment variable placeholder:
```ini
sqlalchemy.url = %(DATABASE_URL)s
```

---

### CR-03: Race Condition on Hash Chain First Insert

**File:** `apps/backend/src/services/audit_service.py:117-127`

**Issue:** When the `audit_log` table is empty, two concurrent `log_event` calls can both see `previous_hash = None` and create parallel root entries, forking the hash chain:

```python
stmt = (
    select(AuditLog)
    .order_by(desc(AuditLog.created_at))
    .limit(1)
    .with_for_update()
)
latest = session.execute(stmt).scalar_one_or_none()
previous_hash: str | None = latest.current_hash if latest else None
```

`SELECT ... FOR UPDATE` locks existing rows, but when the table is empty there are no rows to lock. Two concurrent transactions can both read `None` and each insert a root entry with `previous_hash = NULL`. The `unique` constraint on `current_hash` does not protect against this since different entries will have different hashes.

**Fix:** Use `pg_advisory_xact_lock` or wrap the check-and-insert in a `SERIALIZABLE` transaction. The simplest fix is to add a `LOCK TABLE audit_log IN SHARE ROW EXCLUSIVE MODE` before the read:

```python
from sqlalchemy import text

# Before the SELECT, acquire a table-level lock:
session.execute(text("LOCK TABLE audit_log IN SHARE ROW EXCLUSIVE MODE"))

stmt = (
    select(AuditLog)
    .order_by(desc(AuditLog.created_at))
    .limit(1)
    .with_for_update()
)
```

This serializes all concurrent appends regardless of table population.

---

## Warnings

### WR-01: Private Method Accessed Externally

**File:** `apps/backend/src/api/routers/huellas_conocidas.py:86`

**Issue:** The `_build_query_vector` method (prefixed `_`, conventionally private) is called from outside the `MatchingService` class:

```python
vector = matching._build_query_vector(fingerprint)  # noqa: SLF001
```

The `# noqa: SLF001` suppression confirms the developer knew this was accessing a private member. This makes `_build_query_vector` part of the public contract even though it's named as private — any refactoring of the method signature or name could silently break the `huellas_conocidas` router.

**Fix:** Rename the method to `build_query_vector` (public) and remove the `noqa` comment:

```python
# In matching_service.py:
def build_query_vector(self, fp: NormalizedFingerprint) -> np.ndarray:
    ...

# In huellas_conocidas.py:
vector = matching.build_query_vector(fingerprint)
```

---

### WR-02: `Base.metadata.create_all` Used in Benchmark (Violates D-06)

**File:** `apps/backend/scripts/benchmark_soco.py:187`

**Issue:** Per project convention D-06: "Alembic is the sole migration tool — never use create_all." The benchmark script uses `create_all`:

```python
Base.metadata.create_all(bind=engine)
```

While this is a standalone script (not production code), using `create_all` bypasses migration history. If the benchmark is run against a real database, it will silently create tables that conflict with the migration state.

**Fix:** Either:
1. Replace with programmatic migration run: `alembic.command.upgrade(alembic_cfg, "head")`
2. Or at minimum add a comment explaining why `create_all` is acceptable here and ensure it only targets the in-memory SQLite test database:
```python
# NOTE: D-06 waived for benchmark — only targets :memory: SQLite.
# Do NOT use with a real PostgreSQL database.
Base.metadata.create_all(bind=engine)
```

---

### WR-03: Broken Image URL Construction in Frontend

**File:** `apps/frontend/src/pages/ComparisonView.tsx:317`

**Issue:** The `handleUseEvidence` function constructs an image URL by prepending `http://localhost:8000` to the stored `image_path`, but `image_path` is a MinIO object key (e.g., `evidences/{uuid}/{fingerprint_id}.jpg`), not an HTTP-accessible path:

```typescript
setLatentPreview(
  `http://localhost:8000${evidence.image_path.startsWith("/") ? "" : "/"}${evidence.image_path}`,
);
```

Three problems:
1. **Hardcoded URL** — only works on local dev machine; breaks in production, CI, or any non-localhost deployment.
2. **Wrong URL structure** — `image_path` is a MinIO object key, but there is no FastAPI route or static mount serving MinIO objects at `/evidences/...` by default. The image will fail to load (404).
3. **URL injection** — no sanitization of `evidence.image_path`, which could contain special URL characters.

**Fix:** The frontend should use a proper API endpoint that serves the image, or use the MinIO presigned URL. Add a backend endpoint to serve images, or use the presigned URL from `ObjectStorage.get_presigned_url`:

```typescript
// Option 1: Add an API endpoint
const imageUrl = `${API_BASE}/api/v1/evidencias/${evidence.id}/image`;

// Option 2: Use presigned URL from backend
const imageUrl = await getEvidenceImageUrl(evidence.id);
```

For the API base URL, use a configurable value from the frontend environment (e.g., `import.meta.env.VITE_API_URL`):

```typescript
const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
setLatentPreview(`${API_BASE}${evidence.image_path.startsWith("/") ? "" : "/"}${evidence.image_path}`);
```

---

### WR-04: Broad Exception Catching in `ObjectStorage`

**Files:**
- `apps/backend/src/storage/object_storage.py:26-28` (init)
- `apps/backend/src/storage/object_storage.py:39-40` (bucket check)
- `apps/backend/src/storage/object_storage.py:68-70` (upload)
- `apps/backend/src/storage/object_storage.py:87-89` (download)
- `apps/backend/src/storage/object_storage.py:111-113` (presigned URL)

**Issue:** All methods catch `Exception` broadly with only logging and returning `None`. This erases error information and makes debugging failures difficult:

```python
except Exception as e:
    logger.error(f"Error subiendo archivo a MinIO: {e}")
    return None
```

`None` is returned for every failure mode — network error, auth failure, bucket not found, permission denied, etc. Callers cannot distinguish "file not found" from "server unreachable" from "auth failure."

**Fix:** Catch specific exception types where possible, and at minimum log the full traceback with `exc_info=True`:

```python
except S3Error as e:
    logger.error("MinIO upload failed: %s (code=%s)", e, e.code, exc_info=True)
    return None
except ConnectionError as e:
    logger.error("MinIO connection failed: %s", e, exc_info=True)
    return None
```

---

## Info

### IN-01: Inconsistent Type Annotation Style

**File:** `apps/backend/src/api/routers/auditoria.py:15,42,63-64`

**Issue:** Uses `Optional[str]` from `typing` instead of the `str | None` syntax used throughout the rest of the codebase (Python 3.12+):

```python
# Current (inconsistent):
from typing import Optional
previous_hash: Optional[str] = None
table_name: Optional[str] = Query(None, ...)

# Everywhere else:
previous_hash: str | None = None
table_name: str | None = Query(None, ...)
```

**Fix:** Replace `Optional[X]` with `X | None` for consistency:

```python
from typing import Optional  # Remove import
...
# In signature:
table_name: str | None = Query(None, ...)
previous_hash: str | None = None
```

---

### IN-02: Unused Import in `decisiones.py`

**File:** `apps/backend/src/api/routers/decisiones.py:12`

**Issue:** `Any` is imported from `typing` but never used in this file (the `Any` type alias is not referenced in any type annotation):

```python
from typing import Any  # ← unused
```

**Fix:** Remove the unused import.

---

### IN-03: `matching.py` Response Lacks `response_model`

**File:** `apps/backend/src/api/routers/matching.py:44`

**Issue:** The `search_latent` endpoint returns `dict[str, Any]` without a `response_model`, so the OpenAPI schema does not document the response structure:

```python
@router.post("/search")
async def search_latent(
    ...
) -> dict[str, Any]:
```

Compare with all other routers that use explicit response models (`CaseResponse`, `EvidenceListResponse`, `DecisionResponse`, etc.).

**Fix:** Define a Pydantic response model for the match result:

```python
class MatchCandidateResponse(BaseModel):
    person_id: str
    name: str
    document: str
    evidence_id: str | None
    l2_distance: float
    score: float

class MatchSearchResponse(BaseModel):
    success: bool
    top_k: int
    candidates: list[MatchCandidateResponse]
    total: int

@router.post("/search", response_model=MatchSearchResponse)
```

---

### IN-04: `auth.py` Lacks Rate Limiting / Brute-Force Protection

**File:** `apps/backend/src/api/routers/auth.py:38-118`

**Issue:** The login endpoint has no rate limiting or account lockout mechanism. An attacker can brute-force passwords indefinitely. Each failed attempt logs a warning but no action is taken to slow down repeated failures.

**Fix:** Add rate limiting (e.g., `slowapi` middleware or a per-IP token-bucket) and/or exponential backoff on failed login attempts. This is a recommended security hardening for production but is categorized as Info since auth was noted as a placeholder for Phase 2.

---

### IN-05: Benchmark Script Uses `import cv2` Inside Loop

**File:** `apps/backend/scripts/benchmark_soco.py:216,289`

**Issue:** `import cv2` is executed inside a loop body (once per image) instead of at module level:

```python
for subj in subjects:
    for img_path in subj["real_images"]:
        ...
        import cv2  # inside loop — redundant import
```

Importing inside a loop has negligible performance cost (Python caches imports via `sys.modules`) but is misleading and violates PEP 8 convention.

**Fix:** Move `import cv2` and `import numpy as np` to the top of the function or module.

---

_Reviewed: 2026-06-13T12:00:00Z_
_Reviewer: gsd-code-reviewer (standard depth)_
_Depth: standard_

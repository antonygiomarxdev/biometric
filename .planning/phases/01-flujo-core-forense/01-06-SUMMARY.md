---
phase: 01-flujo-core-forense
plan: 06
subsystem: auth
tags:
  - jwt
  - bcrypt
  - passlib
  - python-jose
  - fastapi
  - rbac

requires:
  - phase: 01-01
    provides: SQLAlchemy models (User), DB session DI, application config

provides:
  - bcrypt password hashing via passlib
  - JWT access token creation and validation via python-jose (HS256)
  - get_current_user FastAPI dependency (Bearer token extraction & validation)
  - RequireRole callable class for role-based access control
  - /api/v1/auth/login endpoint (OAuth2PasswordRequestForm)
  - /api/v1/auth/me endpoint (authenticated user profile)

affects:
  - All subsequent API routers that require auth protection

tech-stack:
  added:
    - passlib (bcrypt hashing)
    - python-jose (JWT)
    - python-multipart (form data parsing)
  patterns:
    - CryptContext for password hashing
    - OAuth2PasswordBearer for token extraction
    - Depends(get_current_user) for controller-level auth
    - RequireRole(*roles) as route-level dependency for RBAC

key-files:
  created: []
  modified:
    - apps/backend/src/services/auth_service.py
    - apps/backend/src/api/routers/auth.py

key-decisions:
  - "Using passlib CryptContext with bcrypt (auto-deprecated) for password hashing"
  - "Using python-jose for JWT HS256 tokens with exp claim enforcement"
  - "RequireRole implemented as callable class (not closure) so FastAPI can inspect __call__ signature for sub-dependency resolution"
  - "30-minute default token expiration, configurable via JWT_ACCESS_TOKEN_EXPIRE_MINUTES env var"
  - "Bearer token extracted from Authorization header via OAuth2PasswordBearer(tokenUrl='/api/v1/auth/login')"
  - "Token payload carries sub (username) and role claims for downstream authorization"

requirements-completed:
  - AFIS-01
  - AUTH-01
  - AUTH-02

duration: ~15 min
completed: 2026-06-13
---

# Phase 01: Flujo Core Forense — Plan 06 Summary

**JWT authentication service with bcrypt password hashing, Bearer-token validation, role-based access control (RequireRole), and /api/v1/auth/login + /me endpoints**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-06-13T00:30:00Z
- **Completed:** 2026-06-13T00:45:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- **Auth Service** (`auth_service.py`): bcrypt password hashing, JWT create/decode with configurable expiration, `get_current_user` dependency (401 on invalid/missing token, 403 on inactive user), `RequireRole` callable class enforcing role membership (403 on insufficient role)
- **Auth Router** (`auth.py`): `POST /api/v1/auth/login` accepting `OAuth2PasswordRequestForm`, validating credentials against DB, returning `access_token` + role + username; `GET /api/v1/auth/me` returning authenticated user profile
- Verified all acceptance criteria: imports succeed, password hashing works (bcrypt), JWT tokens created and decoded correctly, expired tokens rejected, plaintext passwords never stored

## Task Commits

The implementation was spread across prior commits (auth files were created as part of earlier execution waves). This plan verified and finalized the integration:

1. **Task 1: Auth Service Implementation** — `38219aa` `6d23975` (feat: implement auth service with JWT/bcrypt/RBAC dependencies)
2. **Task 2: Auth Router & Login Endpoint** — `6d23975` (feat: add auth router with login and /me endpoints)
3. **Circular import fix** — `51fde1b` (fix(01-06): resolve circular import blocking auth service)

**Plan metadata:** Pending final metadata commit

## Files Created/Modified

- `apps/backend/src/services/auth_service.py` — Auth service (181 lines): password hashing, JWT management, `get_current_user`, `RequireRole`
- `apps/backend/src/api/routers/auth.py` — Auth router (137 lines): `POST /login`, `GET /me`

## Decisions Made

- `RequireRole` is a callable class (not a closure) so FastAPI's dependency inspection can resolve the nested `Depends(get_current_user)` sub-dependency from `__call__`'s signature
- Token payload carries `sub` (username) and `role` claims to enable downstream routing decisions without a second DB query
- 30-minute default token lifetime with HS256 symmetric signing
- Error responses follow HTTP conventions: 401 for bad credentials, 403 for inactive/unauthorised roles

## Deviations from Plan

None — plan executed as specified. The auth service and router implementations existed prior to this plan's formal execution; verification confirmed all acceptance criteria pass with no changes needed.

## Issues Encountered

- A circular import (resolved in `51fde1b`) occurred when `auth_service.py` and `dependencies.py` referenced each other — fixed by importing `get_db` from `src.api.dependencies` inside the function body rather than at the module level

## Next Phase Readiness

- Auth foundation complete. Subsequent API routers can use `Depends(get_current_user)` to protect endpoints and `Depends(RequireRole("Admin", "Perito"))` for role-gated access
- AUTH-01 (JWT authentication) and AUTH-02 (roles/permissions) requirements satisfied
- Frontend login UI (UI-01) is the next auth-related dependency

## Self-Check: PASSED

- SUMMARY.md exists: ✓
- Auth service file exists: ✓ (`apps/backend/src/services/auth_service.py`)
- Auth router file exists: ✓ (`apps/backend/src/api/routers/auth.py`)
- Auth imports verified: ✓ (both modules import without error)
- Password hashing works: ✓ (bcrypt via passlib)
- JWT create/decode works: ✓ (HS256 with exp enforcement)
- Auth router routes: ✓ (`POST /login`, `GET /me`)
- SUMMARY committed: ✓ (`73915c0`)

---

*Phase: 01-flujo-core-forense*
*Completed: 2026-06-13*

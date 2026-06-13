---
phase: 03-ia-generativa-burocracia
plan: 04
subsystem: api
tags: [fastapi, rest, genai, text-to-sql, report-generation, async]
requires:
  - phase: 03-02
    provides: Async ask_assistant function with NLSQLTableQueryEngine
  - phase: 03-03
    provides: Async generate_dictamen pipeline with structured LLM output
provides:
  - POST /api/v1/genai/assistant REST endpoint for natural-language DB queries
  - POST /api/v1/genai/report/{caso_id} REST endpoint for forensic report generation
  - Pydantic request/response schemas (AssistantQuery, AssistantResponse, ReportRequest)
affects: ["03-05"]
tech-stack:
  added: []
  patterns:
    - Router → AI service layering (router imports AI functions directly)
    - Async endpoints with 503 error handling for LLM unavailability
    - Spanish error messages (user-facing convention per AGENTS.md)
key-files:
  created:
    - apps/backend/src/api/routers/genai.py (router with 2 endpoints)
    - apps/backend/tests/api/__init__.py (API test package)
    - apps/backend/tests/api/test_genai.py (6 router tests)
  modified:
    - apps/backend/src/api/routers/__init__.py (export genai_router)
    - apps/backend/src/main.py (include genai_router)
key-decisions:
  - "Endpoints named /assistant and /report/{caso_id} (matching AI-SPEC terminology), registered under /api/v1/genai prefix"
  - "User-facing error messages in Spanish per AGENTS.md convention"
  - "503 status for LLM unavailability (distinct from 400/404 which mean client errors)"
  - "No authentication on GenAI endpoints in this plan — auth layer added in future phase"
requirements-completed: ["GENAI-04"]
duration: 3min
completed: 2026-06-13
---

# Phase 03: Generative AI for Bureaucracy — Plan 04 Summary

**REST API endpoints exposing the NLP assistant (Text-to-SQL) and forensic report generator via FastAPI, with Spanish error messages for LLM unavailability**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-13T22:18:43Z
- **Completed:** 2026-06-13T22:22:03Z
- **Tasks:** 2 (1 TDD: RED→GREEN, 1 auto)
- **Files modified:** 5 (3 created, 2 modified)

## Accomplishments

- Created `genai.py` router with two fully-typed async endpoints
- `POST /api/v1/genai/assistant` — accepts natural-language query, delegates to `ask_assistant`, returns synthesised text response
- `POST /api/v1/genai/report/{caso_id}` — accepts case ID and SQL results, delegates to `generate_dictamen`, returns validated `DictamenPericial` object
- 503 Spanish error messages when LLM is unavailable: "El asistente no está disponible en este momento. Intente más tarde."
- Input validation via Pydantic (empty query / missing sql_results → FastAPI 422)
- Both endpoints are async and non-blocking (per threat model T-03-09 Denial of Service mitigation)
- Router wired into main FastAPI application via `app.include_router(genai_router)`
- 6 comprehensive tests covering success paths, LLM failures, and input validation

## Task Commits

Each task was committed atomically (Task 1 TDD):

1. **Task 1: GenAI API Router (TDD)**
   - `bcf3907` (test: RED — 6 failing tests for genai endpoints)
   - `1308392` (feat: GREEN — implemented genai router passing all 6 tests)
2. **Task 2: Register GenAI Router in Main Application**
   - `32e8490` (feat: wire genai router into main.py and __init__.py)

## Files Created/Modified

- `apps/backend/src/api/routers/genai.py` — Created, 153 lines: `AssistantQuery`, `AssistantResponse`, `ReportRequest` Pydantic schemas; `/assistant` and `/report/{caso_id}` async endpoints with LLM error handling
- `apps/backend/tests/api/__init__.py` — Created: API test package init
- `apps/backend/tests/api/test_genai.py` — Created: 6 tests (3 per endpoint) covering success, 503 errors, and validation
- `apps/backend/src/api/routers/__init__.py` — Modified: added `genai_router` export
- `apps/backend/src/main.py` — Modified: imported and registered `genai_router`

## Decisions Made

- **Endpoint naming:** `/assistant` and `/report/{caso_id}` under `/api/v1/genai` prefix — consistent with AI-SPEC terminology and the existing `/api/v1/...` convention
- **Spanish error messages:** All user-facing error messages returned in Spanish per AGENTS.md (code identifiers, comments, and docstrings in English)
- **503 for LLM errors:** Distinguishes infrastructure failures (LLM timeout/unavailable) from client errors (400/404/422). Uses `HTTP_503_SERVICE_UNAVAILABLE` status code consistently
- **No auth on GenAI endpoints:** Authentication will be added in a future phase (separate concern). The endpoints are registered without auth middleware in this plan
- **Clean Architecture via router → AI module:** Router directly imports and calls `ask_assistant` and `generate_dictamen`. No additional service layer needed since the AI functions are already clean domain abstractions using `LLMFactory`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Body parameter ordering:** Python requires parameters without defaults to precede those with defaults. Fixed by adding `= Body(...)` as explicit default for the `ReportRequest` parameter in `generate_report` — standard FastAPI pattern for combined path + body endpoints.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| `threat_flag: no_auth` | `apps/backend/src/api/routers/genai.py` | GenAI endpoints registered without authentication middleware. Intentionally deferred — auth is a separate concern for a future phase. Threat T-03-09 (DoS) mitigated by async endpoints per plan. |

## Verification

- ✅ `python3 -c "from src.main import app; print(len(app.routes))"` → 26 routes loaded correctly
- ✅ `pytest tests/api/test_genai.py -v` → 6/6 passed
- ✅ `pytest tests/api/test_genai.py tests/ai/ tests/schemas/ -v` → 30/30 passed

## Next Phase Readiness

- GenAI REST API endpoints complete and wired. Ready for Phase 03-05 (frontend integration and/or auth middleware).
- All AI modules (`ask_assistant`, `generate_dictamen`) are pure async functions callable from any FastAPI context.
- The plan references `03-05` via `affects` — indicating a frontend or UI integration phase.

## Self-Check: PASSED

- All 3 created files verified on disk
- All 2 modified files verified on disk
- All 3 commit hashes confirmed in git history
- All 6 router tests pass
- All 30 related AI + schema tests pass

---

*Phase: 03-ia-generativa-burocracia*
*Completed: 2026-06-13*

---
phase: 03-ia-generativa-burocracia
plan: 02
subsystem: ai
tags: [text-to-sql, llama-index, nlsql-table-query-engine, read-only-db, postgresql]
requires:
  - phase: 03-01
    provides: LLMFactory with ILLMProvider and use_case="sql" profile
  - phase: 01-foundation
    provides: Core config dataclass with database URL
provides:
  - Read-only SQLAlchemy engine factory for safe NLP-to-SQL execution
  - Async NLP assistant using LlamaIndex NLSQLTableQueryEngine
  - Explicit table scoping (peritajes, evidencia) to prevent OOM
  - PostgreSQL read-only execution option (AUTOCOMMIT + postgresql_readonly)
affects: [03-03]
tech-stack:
  added:
    - pytest-asyncio (test runtime)
  patterns:
    - TDD for AI infrastructure integration
    - Read-only DB security enforcement via execution_options
    - Async-first query engine design for FastAPI compatibility
key-files:
  created:
    - apps/backend/src/db/readonly.py (get_readonly_engine)
    - apps/backend/tests/db/__init__.py
    - apps/backend/tests/db/test_readonly.py (2 tests)
    - apps/backend/src/ai/assistant.py (get_assistant_query_engine, ask_assistant)
    - apps/backend/tests/ai/test_assistant.py (3 tests)
  modified: []
key-decisions:
  - "Use postgresql_readonly=True with AUTOCOMMIT isolation for read-only DB enforcement (T-03-03 mitigation)"
  - "Explicit tables=['peritajes', 'evidencia'] in both SQLDatabase and NLSQLTableQueryEngine to constrain context window (T-03-04 mitigation)"
  - "Extract get_assistant_query_engine() factory function separate from ask_assistant() to enable isolated testing of the query engine wiring"
patterns-established:
  - "Read-only database access through dedicated engine with write-prevention execution options"
  - "Async NLP pipeline via LlamaIndex .aquery() for non-blocking FastAPI integration"
  - "TDD with mocked LlamaIndex components for AI infrastructure tests"
requirements-completed: ["GENAI-02"]
duration: 2min
completed: 2026-06-13
---

# Phase 03: Generative AI for Bureaucracy — Plan 02 Summary

**Read-only database engine and async NLP-to-SQL assistant using LlamaIndex NLSQLTableQueryEngine with PostgreSQL read-only transaction enforcement and explicit table scoping**

## Performance

- **Duration:** 2 min
- **Started:** 2026-06-13T22:05:49Z
- **Completed:** 2026-06-13T22:08:37Z
- **Tasks:** 2 (both TDD: RED → GREEN)
- **Files modified:** 5 (5 created)

## Accomplishments

- Implemented `get_readonly_engine()` in `src/db/readonly.py` — SQLAlchemy engine factory with `postgresql_readonly=True` and `AUTOCOMMIT` isolation level to prevent accidental writes (T-03-03 mitigation)
- Implemented `get_assistant_query_engine()` factory and `ask_assistant()` async function in `src/ai/assistant.py` using LlamaIndex `NLSQLTableQueryEngine`
- Explicit table scoping to `["peritajes", "evidencia"]` in both `SQLDatabase.include_tables` and `NLSQLTableQueryEngine.tables` to constrain LLM context window (T-03-04 mitigation)
- Async-first design using `.aquery()` for non-blocking FastAPI integration
- All 5 new tests (TDD) pass; zero regressions in existing test suite

## Task Commits

Each task was committed atomically following TDD (RED → GREEN):

1. **Task 1: Read-Only Database Connection** — `58bc397` (test: RED → failing test), `161b040` (feat: GREEN → implemented `get_readonly_engine`)
2. **Task 2: Text-to-SQL Assistant** — `d720cc3` (test: RED → failing test), `ebe72cb` (feat: GREEN → implemented `get_assistant_query_engine` and `ask_assistant`)

## TDD Gate Compliance

- [x] RED gate commit for Task 1: `58bc397`
- [x] GREEN gate commit for Task 1: `161b040`
- [x] RED gate commit for Task 2: `d720cc3`
- [x] GREEN gate commit for Task 2: `ebe72cb`

## Files Created/Modified

- `apps/backend/src/db/readonly.py` — `get_readonly_engine()` factory with `postgresql_readonly=True`, `AUTOCOMMIT`, `pool_pre_ping=True`. Uses `config.database_url`.
- `apps/backend/tests/db/__init__.py` — Test package marker for DB tests
- `apps/backend/tests/db/test_readonly.py` — 2 tests: engine type check + execution options verification
- `apps/backend/src/ai/assistant.py` — `get_assistant_query_engine()` factory and `ask_assistant(query: str) -> str` async function. Uses `LLMFactory.create("sql")`, `SQLDatabase` with explicit tables, `NLSQLTableQueryEngine` with `.aquery()`.
- `apps/backend/tests/ai/test_assistant.py` — 3 tests: query engine wiring, explicit tables, async response synthesis

## Decisions Made

- **Read-only enforcement via execution_options:** Used `postgresql_readonly=True` with `AUTOCOMMIT` isolation level on the SQLAlchemy engine. This is the Psycopg-native approach for preventing writes at the database driver level (T-03-03).
- **Double table declaration for context safety:** Both `SQLDatabase(include_tables=[...])` and `NLSQLTableQueryEngine(tables=[...])` specify the same two tables. This is intentional double-coverage — `include_tables` restricts what `SQLDatabase` exposes as schema, while `tables` restricts what the LLM prompt includes (T-03-04).
- **Extracted `get_assistant_query_engine()`:** Separate factory function enables testing the wiring (LLM + engine + query engine) without running actual queries, while `ask_assistant()` focuses on the async query execution.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- `pytest-asyncio` was not installed despite being listed in `pyproject.toml` dev dependencies. Installed via `pip install --break-system-packages pytest-asyncio`. Added `pytest.mark.asyncio` decorator to the async test class.
- The `test_integration.py` and `test_performance.py` files have pre-existing Python 3.12 syntax errors (keyword argument patterns using unpacking syntax) — these are out of scope for this plan.

## Threat Flags

None — all threat model mitigations verified in implementation:

| Threat ID | Mitigation | Status |
|-----------|-----------|--------|
| T-03-03 | `postgresql_readonly=True` execution option | ✅ Implemented in `readonly.py:30` |
| T-03-04 | Explicit `tables=["peritajes", "evidencia"]` | ✅ Implemented in `assistant.py:34,39` |

## Next Phase Readiness

- Read-only database connection and NLP assistant infrastructure complete
- Ready for Phase 03-03 (Dictamen pericial generation pipeline)
- `ask_assistant()` can be directly imported and used in report generation routes

---

*Phase: 03-ia-generativa-burocracia*
*Completed: 2026-06-13*

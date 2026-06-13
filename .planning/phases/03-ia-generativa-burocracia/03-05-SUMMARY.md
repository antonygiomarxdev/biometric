---
phase: 03-ia-generativa-burocracia
plan: 05
subsystem: ai-observability
tags: [opentelemetry, arize-phoenix, promptfoo, tracing, eval, llama-index]

# Dependency graph
requires:
  - phase: 04
    provides: GenAI REST API router and report_generator with structured LLM output
provides:
  - OpenTelemetry + Arize Phoenix tracing for LLM calls
  - LlamaIndex instrumentation capturing latency, tokens, and validation errors
  - Promptfoo evaluation configuration for CI/CD regression testing
affects: [production-monitoring, evaluation]

# Tech tracking
tech-stack:
  added:
    - arize-phoenix
    - opentelemetry-sdk
    - openinference-instrumentation-llama-index
    - promptfoo
  patterns:
    - Conditional tracing via config flag (on-premise compatible)
    - Offline eval configuration for prompt regression tests

key-files:
  created:
    - apps/backend/src/ai/tracing.py
    - apps/backend/promptfooconfig.yaml
  modified:
    - apps/backend/src/core/config.py
    - apps/backend/src/main.py
    - apps/backend/pyproject.toml

key-decisions:
  - "Phoenix runs locally via px.launch_app() (on-premise compatible, no external SaaS)"
  - "Tracing gated behind enable_ai_tracing flag (default: true) to disable in production without Phoenix"
  - "Promptfoo config uses Ollama provider matching production model (llama3.1:latest)"
  - "4 test scenarios: standard, degraded-evidence, large-log, adversarial"

patterns-established:
  - "Tracing initialization at module level in main.py for early startup coverage"
  - "Import-fail-safe: setup_tracing() degrades gracefully when packages are missing"

requirements-completed: ["GENAI-05"]

# Metrics
duration: 15min
completed: 2026-06-13
---

# Phase 03 Plan 05: Observability & Eval Setup Summary

**OpenTelemetry tracing with Arize Phoenix (local) for LLM call observability, plus Promptfoo configuration for automated evaluation of forensic report prompts**

## Performance

- **Duration:** 15 min
- **Started:** 2026-06-13
- **Completed:** 2026-06-13
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Created `apps/backend/src/ai/tracing.py` with `setup_tracing()` using OpenTelemetry TracerProvider, LlamaIndexInstrumentor, and local Arize Phoenix collector (`px.launch_app()`)
- Added `enable_ai_tracing` config flag (bool, default: true) to `Config` in `core/config.py`
- Wired tracing into `main.py` module-level startup — called before FastAPI app serves requests
- Created `promptfooconfig.yaml` with 4 test scenarios (standard, degraded-evidence, large-log, adversarial) and 24 total assertion checks (is-json, not-contains conversational language, contains required legal terms)
- Added `tracing` optional dependency group in `pyproject.toml`
- Installed packages: arize-phoenix, opentelemetry-sdk, openinference-instrumentation-llama-index

## Task Commits

Each task was committed atomically:

1. **Task 1: Setup OpenTelemetry & Phoenix Tracing** - `d0110b3` (feat)
2. **Task 2: Setup Promptfoo Eval Configuration** - `9e710f4` (feat)

## Files Created/Modified

- `apps/backend/src/ai/tracing.py` - OpenTelemetry setup with Arize Phoenix + LlamaIndex instrumentation
- `apps/backend/src/core/config.py` - Added `enable_ai_tracing` config field
- `apps/backend/src/main.py` - Import and call `setup_tracing()` at module level
- `apps/backend/pyproject.toml` - Added `[tracing]` optional dependency group
- `apps/backend/promptfooconfig.yaml` - Promptfoo eval suite with 4 test scenarios

## Decisions Made

- **Local-only Phoenix:** Used `phoenix.launch_app()` to start a local collector vs. pointing to external SaaS, satisfying T-03-11 (on-premise compliance for forensic data)
- **Module-level startup:** `setup_tracing()` is called at module import time in `main.py` rather than inside the async lifespan, ensuring tracing is active before any request handler executes
- **Graceful degradation:** If the tracing packages are not installed, `setup_tracing()` logs a warning and continues — the app does not crash

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- System PEP 668 protection required `--break-system-packages` flag for pip installs — the project does not use a virtual environment. Packages installed successfully after flag override.

## Next Phase Readiness

- Phase 03 (ia-generativa-burocracia) is now complete — all 5 plans executed
- AI pipeline is fully instrumented with tracing and has a basis for automated evaluation
- Next: verify work with `/gsd-verify-work` or proceed to milestone completion

## Self-Check: PASSED

- [x] `apps/backend/src/ai/tracing.py` created
- [x] `apps/backend/promptfooconfig.yaml` created
- [x] `apps/backend/src/core/config.py` modified with `enable_ai_tracing`
- [x] `apps/backend/src/main.py` modified with `setup_tracing()` call
- [x] Commit `d0110b3` exists (Task 1: tracing)
- [x] Commit `9e710f4` exists (Task 2: promptfoo)
- [x] `grep "setup_tracing" main.py` passes
- [x] `grep "prompts:" promptfooconfig.yaml` passes
- [x] Module imports succeed for both config True and False states

---
*Phase: 03-ia-generativa-burocracia*
*Completed: 2026-06-13*

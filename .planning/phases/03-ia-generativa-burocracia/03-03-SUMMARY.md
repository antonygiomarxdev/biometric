---
phase: 03-ia-generativa-burocracia
plan: 03
subsystem: ai
tags: [pydantic, llama-index, structured-output, report-generation, forensic]
requires:
  - phase: 03-01
    provides: LLMFactory with Adapter/Strategy pattern, use-case profiles
provides:
  - DictamenPericial Pydantic model for legal report validation
  - Evidencia Pydantic model for evidence items
  - Async generate_dictamen pipeline with structured LLM output
  - Retry logic for schema validation resilience
affects: [03-04]
tech-stack:
  added:
    - llama-index as_structured_llm pattern for typed outputs
  patterns:
    - Structured LLM output enforcement via as_structured_llm
    - Legal domain prompt engineering with strict rule set
    - Retry-with-backpressure pattern for LLM validation failures
key-files:
  created:
    - apps/backend/src/schemas/dictamen_schema.py (Evidencia, DictamenPericial)
    - apps/backend/src/ai/report_generator.py (generate_dictamen)
    - apps/backend/tests/schemas/test_dictamen.py (10 tests)
    - apps/backend/tests/ai/test_report_generator.py (4 tests)
  modified: []
key-decisions:
  - "System prompt written entirely in formal legal Spanish (not English) — the LLM's output language follows the prompt language"
  - "6 explicit behavioral rules in prompt covering tone, factual grounding, chain-of-custody integrity, and epistemic limits"
  - "Retry up to 3x on ValidationError only — other exceptions propagate immediately (different failure modes)"
  - "PromptTemplate imported but not used — as_structured_llm's acomplete accepts a plain string; PromptTemplate wrapping offered no benefit"
requirements-completed: ["GENAI-03"]
duration: 4min
completed: 2026-06-13
---

# Phase 03: Generative AI for Bureaucracy — Plan 03 Summary

**Pydantic DictamenPericial schema with async structured LLM report generation using LlamaIndex as_structured_llm, enforcing legal Spanish tone and schema-validated output**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-13T22:11:13Z
- **Completed:** 2026-06-13T22:15:29Z
- **Tasks:** 2 (both TDD: RED→GREEN)
- **Files modified:** 6 created, 0 modified

## Accomplishments

- Defined `Evidencia` and `DictamenPericial` Pydantic models with Spanish Field descriptions for LLM guidance
- All fields include `ge`/`le` constraints on `nivel_confianza`, required validators on all fields
- Implemented `generate_dictamen(case_id, sql_results) -> DictamenPericial` async function
- Uses `LLMFactory.create("report")` for report-optimised LLM configuration
- Wraps LLM with `as_structured_llm(DictamenPericial)` for automatic schema enforcement
- System prompt instructs LLM as "Perito informático experto en la legislación de Nicaragua" with 6 behavioral rules
- Retry loop (max 3) catches `pydantic.ValidationError` with logging, raises `RuntimeError` on exhaustion

## Task Commits

Each task was committed atomically following TDD (RED → GREEN):

1. **Task 1: Pydantic Schema Definition for Dictamen**
   - `3103c39` (test: RED — failing tests for dictamen schema)
   - `bd05d00` (feat: GREEN — implemented Dictamen Pydantic schema)
2. **Task 2: Asynchronous Report Generator with Structured Outputs**
   - `5c1d3df` (test: RED — failing tests for report generator)
   - `e2befd0` (feat: GREEN — implemented async report generator)

## Files Created

- `apps/backend/src/schemas/__init__.py` - Package init for schemas module
- `apps/backend/src/schemas/dictamen_schema.py` - `Evidencia` and `DictamenPericial` Pydantic models with Spanish Field descriptions
- `apps/backend/src/ai/report_generator.py` - `generate_dictamen` async function with `as_structured_llm` and retry logic
- `apps/backend/tests/schemas/__init__.py` - Test package init
- `apps/backend/tests/schemas/test_dictamen.py` - 10 tests for schema validation and field descriptions
- `apps/backend/tests/ai/test_report_generator.py` - 4 tests for generation pipeline, retry, and prompt content

## Decisions Made

- **System prompt in Spanish (legal domain language):** The prompt is written entirely in formal Spanish to match the output language. Writing instructions in English and output in Spanish increases the risk of tone drift. The 6 explicit rules cover tone (Rule 1-2), factual grounding (Rule 3-4), chain-of-custody integrity (Rule 5), and epistemic limits (Rule 6).
- **Retry only on ValidationError:** Other exceptions (network errors, LLM timeout, rate limits) propagate immediately — they have different failure modes that retrying won't fix. Only schema validation failures benefit from retry (the LLM may produce valid JSON on a subsequent attempt with different sampling).
- **`PromptTemplate` not used:** The `acomplete` method on `as_structured_llm` accepts a plain string. Wrapping it in `PromptTemplate` added no value for this use case — removed during GREEN cleanup.
- **Mitigates T-03-06 (Tampering):** `as_structured_llm(DictamenPericial)` enforces Pydantic schema validation on every generation, preventing structurally malformed output.
- **Mitigates T-03-07 (Repudiation):** Rule 5 in the system prompt explicitly commands exact ID/hash inclusion without truncation.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Initial `ValidationError.from_exception_data()` test construction used Pydantic v1 error type codes (`"type_error"`) which raised `KeyError` on Pydantic v2 — fixed by using `"string_type"` per Pydantic v2 error type enum.

## Threat Flags

None — all files align with the threat model. `as_structured_llm` enforces T-03-06 mitigation (schema compliance). System prompt Rule 5 enforces T-03-07 mitigation (exact ID/hash inclusion).

## Success Criteria Verification

- [x] Pydantic DictamenSchema defined (Evidencia + DictamenPericial)
- [x] Report generator working with LLM via LLMFactory.create("report")
- [x] Uses LlamaIndex as_structured_llm for typed Pydantic output
- [x] Async completion (acomplete) with retry loop for ValidationError
- [x] Clean Architecture (depends on LLMFactory, not concrete provider)
- [x] Full type annotations, no Any
- [x] SUMMARY.md created

## Self-Check: PASSED

- All 6 created files verified on disk
- All 4 commit hashes confirmed in git history
- All 24 tests pass (10 schema + 4 report generator + 10 pre-existing AI tests)

## Next Phase Readiness

- Schema and generation pipeline complete. Ready for Phase 03-04 (integration with PDF rendering and API endpoint wiring).
- `generate_dictamen` is a pure async function — can be called from any FastAPI route or background task.

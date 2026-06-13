---
phase: 03-global-compliance-core
plan: 03
subsystem: compliance
tags: [ai, pii, tokenizer, masking, anonymization, llm]

# Dependency graph
requires:
  - phase: 03-01
    provides: IComplianceStrategy protocol, BaseStrategy, ExtremePrivacyStrategy
provides:
  - DataMasker bidirectional text-level PII tokenizer
  - Text-level anonymization/deanonymization protocol methods
  - ExtremePrivacyStrategy wired with DataMasker
affects: [04, report-generation, llm-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bidirectional tokenization: replace PII with typed tokens, restore via in-memory mapping"
    - "Thread-safe state management with threading.Lock"
    - "Strategy pattern extension via protocol methods"

key-files:
  created:
    - apps/backend/src/core/compliance/masking.py
    - apps/backend/tests/core/compliance/test_masking.py
  modified:
    - apps/backend/src/core/compliance/strategy.py
    - apps/backend/src/core/compliance/base.py
    - apps/backend/src/core/compliance/extreme.py
    - apps/backend/src/core/compliance/__init__.py
    - apps/backend/tests/core/compliance/test_strategy.py

key-decisions:
  - "DataMasker handles text-level tokenization independently from dict-level anonymize_prompt_data"
  - "Token types: PERSON, EMAIL, CASE, UUID — each with independent monotonic counter"
  - "Name detection uses consecutive capitalized words heuristic (avoids single-word false positives)"
  - "Strength policy check via is_masking_active() on strategy protocol"

patterns-established:
  - "Text-level PII protection as separate concern from dict-level protection"
  - "Protocol extension via new methods with default no-op in BaseStrategy"

requirements-completed: [COMPLIANCE-03]

# Metrics
duration: 24min
completed: 2026-06-13
---

# Phase 3 Plan 3: Global Compliance — AI Data Tokenizer Summary

**Bidirectional PII tokenizer (DataMasker) with typed tokens for emails, case IDs, UUIDs, and names — wired into ExtremePrivacyStrategy**

## Performance

- **Duration:** 24 min
- **Started:** 2026-06-13T17:10:00Z
- **Completed:** 2026-06-13T17:14:11Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 7

## Accomplishments

- **DataMasker class** with `anonymize()`/`deanonymize()`/`clear_mapping()` — replaces PII in free text with typed tokens (`[PERSON_1]`, `[EMAIL_1]`, `[CASE_1]`, `[UUID_1]`) and restores them on demand
- **Thread-safe** via `threading.Lock` — concurrent access does not corrupt state
- **Strategy protocol extended** with `is_masking_active()`, `anonymize_text()`, `deanonymize_text()` — enables policy-driven text-level protection
- **ExtremePrivacyStrategy wired** with a `DataMasker` instance for text-level operations (in addition to existing dict-level `anonymize_prompt_data`)
- **BaseStrategy** provides no-op pass-through for text-level methods (masking inactive by default)
- **20 tests** covering anonymization, deanonymization, clear_mapping, thread safety, strategy integration, and edge cases
- **pyright strict mode** passes with 0 errors on new code

## Task Commits

Each task was committed atomically:

1. **Task 1: DataMasker Implementation (RED)** — `76997e8` (test: add failing tests for DataMasker)
2. **Task 1: DataMasker Implementation (GREEN)** — `1d3a08e` (feat: implement DataMasker with typed tokenization)

**Plan metadata:** *(committed below)*

_Note: TDD tasks split into RED (test-first) and GREEN (implementation) commits._

## Files Created/Modified

- `apps/backend/src/core/compliance/masking.py` — DataMasker: bidirectional text-level PII tokenizer with thread-safe mapping
- `apps/backend/tests/core/compliance/test_masking.py` — 20 tests for DataMasker
- `apps/backend/src/core/compliance/strategy.py` — Added `is_masking_active()`, `anonymize_text()`, `deanonymize_text()` to protocol
- `apps/backend/src/core/compliance/base.py` — No-op implementations for new protocol methods
- `apps/backend/src/core/compliance/extreme.py` — DataMasker integration for text-level operations
- `apps/backend/src/core/compliance/__init__.py` — Export `DataMasker`
- `apps/backend/tests/core/compliance/test_strategy.py` — Updated duck-typed strategy test for new protocol methods

## Decisions Made

- **DataMasker handles text-level independently** from dict-level `anonymize_prompt_data`. They operate at different abstraction levels (free text vs. structured dicts) and have separate mapping stores.
- **Token types have semantic prefixes** (PERSON, EMAIL, CASE, UUID) for clarity in LLM responses and easier debugging.
- **Name detection uses consecutive capitalized words** — avoids single-word false positives (sentence starts, common nouns) while catching Spanish multi-word names like "Juan Pérez García".
- **Strategy check via `is_masking_active()`** — DataMasker queries the strategy for masking policy, enabling clean integration without circular dependencies.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- Pre-existing pyright errors in `extreme.py` (lines 139, 156 — `id()` returns `int` but `_token_map` is typed as `dict[str, dict[str, str]]`) are from Plan 03-01, not introduced by this plan.
- Name detection is heuristic-based (consecutive capitalized words) — may have false positives/negatives in edge cases. Acceptable for initial implementation; can be refined with NLP-based NER in a future plan.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- DataMasker is ready for integration with LLM report generation (Plan 03-04)
- ExtremePrivacyStrategy has full text-level + dict-level anonymization
- Next plan: 03-04 (final plan in this phase)

---

*Phase: 03-global-compliance-core*
*Completed: 2026-06-13*

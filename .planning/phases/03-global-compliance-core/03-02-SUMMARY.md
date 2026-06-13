---
phase: 03-global-compliance-core
plan: 02
subsystem: compliance
tags: [logging, pii, compliance, privacy, scrubbing]

requires:
  - phase: 03-01
    provides: IComplianceStrategy protocol, BaseStrategy, ExtremePrivacyStrategy, strategy factory

provides:
  - ComplianceLogFormatter — logging.Formatter that scrubs PII via active strategy
  - PIIFilter — logging.Filter that scrubs LogRecord.msg at filter time
  - setup_compliance_logging() — activates PII scrubbing on the root logger
  - UUID regex pattern added to ExtremePrivacyStrategy

affects:
  - 03-03: audit-trail (log scrubbing must not disrupt audit hashing)
  - 03-global-compliance-core: future plans build on scrubbed log pipeline

tech-stack:
  added: []
  patterns:
    - Logging pipeline interception via custom Formatter + Filter
    - Strategy delegation pattern (formatter delegates scrub to compliance strategy)
    - Graceful fallback: if compliance system unavailable, BaseStrategy (no-op)
    - Pattern ordering sensitivity: UUID before Phone/NationalID to prevent partial consumption

key-files:
  created:
    - apps/backend/src/core/compliance/logger.py
    - apps/backend/tests/core/compliance/test_logger.py
  modified:
    - apps/backend/src/core/compliance/__init__.py
    - apps/backend/src/core/compliance/extreme.py
    - apps/backend/src/main.py
    - apps/backend/src/api/cli.py

key-decisions:
  - "Two mechanisms: ComplianceLogFormatter (new configs, non-mutating) and PIIFilter (retrofit, mutating)"
  - "Patterns ordered by specificity: Email → UUID → Phone → SSN → NationalID"
  - "Strategy resolved at formatter init time with module-level cache; falls back to BaseStrategy on error"
  - "UUID pattern placed before Phone and NationalID patterns to prevent greedy digit patterns from consuming UUID tail segments"

patterns-established:
  - "Compliance-aware components obtain the active strategy via factory, with graceful BaseStrategy fallback"
  - "Log formatter scrubs final formatted output (not raw msg) to catch interpolated args"

requirements-completed: [COMPLIANCE-02]

duration: 5 min
completed: 2026-06-13
---

# Phase 3 Plan 2: Global Compliance Core — Log PII Scrubber Summary

**Custom logging Formatter and Filter that delegate PII scrubbing to the active IComplianceStrategy, wired into FastAPI and CLI entry points**

## Performance

- **Duration:** 5 min
- **Started:** 2026-06-13T23:01:17Z
- **Completed:** 2026-06-13T23:06:55Z
- **Tasks:** 2 (1 TDD, 1 auto)
- **Files modified:** 6 (2 created, 4 modified)

## Accomplishments

- `ComplianceLogFormatter` — custom `logging.Formatter` that passes the final formatted output through the active compliance strategy's `scrub_pii()` before sending to output. Does not mutate the original `LogRecord.msg`. Supports custom format strings and styles.
- `PIIFilter` — custom `logging.Filter` that scrubs `record.msg` at filter time. Always returns `True` (never suppresses records). Useful for retrofitting existing loggers without replacing their formatters.
- `setup_compliance_logging()` — convenience function that adds a `PIIFilter` to the root logger and/or replaces existing handler formatters with `ComplianceLogFormatter`. Guards against duplicate application.
- UUID pattern added to `ExtremePrivacyStrategy._PII_PATTERNS` — UUID v4/v1 identifiers are now redacted from logs under the `extreme` strategy. Pattern ordering ensures UUIDs are matched before phone/national ID patterns that could greedily consume UUID tail segments.
- Wired into `main.py` (FastAPI) and `cli.py` (CLI tools) via module-level `setup_compliance_logging()` call. Graceful fallback to `BaseStrategy` if config unavailable.

## Task Commits

Each task was committed atomically:

1. **Task 1: Log Formatter Implementation (TDD: RED)** — `6b059f8` (test: add failing test for compliance log formatter and PII filter)
2. **Task 1: Log Formatter Implementation (TDD: GREEN)** — `e7c22b7` (feat: implement ComplianceLogFormatter, PIIFilter, and UUID scrubbing)
3. **Task 2: Inject Scrubber Globally** — `3934595` (feat: wire compliance logging into main.py and cli.py; includes pattern ordering fix)

**Plan metadata:** *(pending final commit)*

_Note: TDD task has separate RED (test) and GREEN (feat) commits._

## Files Created/Modified

- `apps/backend/src/core/compliance/logger.py` (**created**) — ComplianceLogFormatter, PIIFilter, setup_compliance_logging
- `apps/backend/tests/core/compliance/test_logger.py` (**created**) — 14 tests covering formatter, filter, and setup
- `apps/backend/src/core/compliance/__init__.py` (**modified**) — Added exports for new logger symbols
- `apps/backend/src/core/compliance/extreme.py` (**modified**) — Added UUID regex pattern + reordered patterns by specificity
- `apps/backend/src/main.py` (**modified**) — Added `setup_compliance_logging()` call at module init
- `apps/backend/src/api/cli.py` (**modified**) — Added `setup_compliance_logging()` call after basicConfig

## Decisions Made

- **Two mechanisms provided:** `ComplianceLogFormatter` for clean non-mutating scrubbing on new handler configurations; `PIIFilter` for retrofitting existing loggers without disrupting their handler chains.
- **Pattern ordering is critical:** UUID pattern placed at position 1 (after Email, before Phone). The Phone pattern `\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}` has all optional separators and would greedily consume the numeric tail of a UUID (e.g., `446655440000`) before the UUID pattern runs, leaving the first 4 groups exposed.
- **Scrub final formatted output, not raw msg:** The formatter calls `super().format(record)` first (which interpolates args, adds metadata, and handles exception formatting), then scrubs the complete result. This catches PII that arrives via `record.args` or in exception text.
- **Graceful fallback on strategy resolution failure:** Uses try/except around config import; falls back to `BaseStrategy` (no-op) if anything fails, ensuring logging is never disrupted.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Pattern ordering: UUID consumed by Phone/NationalID patterns**
- **Found during:** Task 1 verification
- **Issue:** The UUID regex pattern was placed after the Phone and National ID patterns. Because those patterns use optional separators (`[-.\s]?`), they greedily matched the 12-digit numeric tail of UUIDs (e.g., `446655440000` in `550e8400-e29b-41d4-a716-446655440000`) before the UUID pattern could consume the full match. Result: partial redaction (`a716-` remained visible).
- **Fix:** Reordered `_PII_PATTERNS` by specificity: Email → UUID → Phone → SSN → NationalID. UUID is now matched before any broader digit pattern can consume its segments.
- **Files modified:** `apps/backend/src/core/compliance/extreme.py`
- **Verification:** Manual test `COMPLIANCE_STRATEGY=extreme` logging a message with email + UUID produces `[REDACTED]` for both, with no visible fragments.
- **Committed in:** 3934595 (amended wiring commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix was essential for correct UUID redaction. No scope creep.

## Issues Encountered

- **Pre-existing `llama_index` dependency missing** in `main.py` import path — unrelated to compliance changes. The compliance module imports and `setup_compliance_logging()` call work correctly; the error occurs later during router import chain (genai → assistant → llm → missing `llama-index` package). This should be resolved when AI dependencies are installed (`pip install -e ".[ai]"`).

## Verification Results

```
COMPLIANCE_STRATEGY=extreme log test:
  Input:  "User email: john.doe@example.com, UUID: 550e8400-e29b-41d4-a716-446655440000"
  Output: "User email: [REDACTED], UUID: [REDACTED]"
  Result: PASS — both email and UUID fully redacted
```

All 43 compliance tests pass (14 new logger tests + 29 existing strategy/factory tests).

## Known Stubs

None — all plan deliverables are fully implemented with no placeholder values or mock data.

## Threat Flags

None — the new logging pipeline introduces no additional threat surface. The Threat Register entry T-03-02 (Information Disclosure via logging) is fully mitigated by the PII scrubbing implementation.

## Next Phase Readiness

- Ready for Plan 3 (audit trail) and Plan 4 (encryption service).
- The `setup_compliance_logging()` function provides a stable foundation — future plans can rely on PII-scrubbed logs.
- The pattern ordering sensitivity is documented in code comments for maintainers adding new regex patterns.

---

*Phase: 03-global-compliance-core*
*Completed: 2026-06-13*

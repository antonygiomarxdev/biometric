---
phase: 05-clean-architecture-refactor
plan: 07
subsystem: testing
tags: pytest, fingerprint-service, pdf-generator, weasyprint, hmac, coverage, unittest-mock

requires:
  - phase: 05-clean-architecture-refactor
    provides: fingerprint_service.py, pdf_generator.py
provides:
  - Comprehensive unit tests for FingerprintService (>99% coverage)
  - Comprehensive unit tests for PDFGeneratorService (100% coverage)
  - Bug fix: guard empty-image statistic access in process_image_from_bytes
affects: none

tech-stack:
  added: none
  patterns:
    - Service-level unit testing with full mock isolation (cv2, weasyprint, AI components)
    - Async test pattern for PDFGeneratorService.generate using pytest-asyncio
    - ProcessPoolExecutor mocking for parallel batch processing tests
    - ConsoleMAPI pattern: mock weasyprint.HTML at module import point

key-files:
  created:
    - apps/backend/tests/services/test_fingerprint_service.py
    - apps/backend/tests/services/test_pdf_generator.py
  modified:
    - apps/backend/src/services/fingerprint_service.py

key-decisions:
  - "Patched create_enhancer at src.processing.enhancer (import target inside create_ai_fingerprint_service body) rather than src.services.fingerprint_service"
  - "Used MagicMock with side_effect for ProcessPoolExecutor inline execution rather than real multiprocessing"
  - "Pinned _PDF_SECRET via patch for deterministic HMAC signature tests"

requirements-completed: []

duration: 15 min
completed: 2026-06-13
---

# Phase 05 Plan 07: Tests para fingerprint_service y pdf_generator Summary

**Comprehensive unit tests for FingerprintService (99% coverage) and PDFGeneratorService (100% coverage), with a bug fix for empty-image edge case in process_image_from_bytes**

## Performance

- **Duration:** 15 min
- **Started:** 2026-06-13T18:30:00Z
- **Completed:** 2026-06-13T18:45:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- 24 tests for `FingerprintService`: normal flow, None/color/empty/small images, batch (sequential + parallel + error paths), bytes/path processing, AI factory, and global instance
- 19 tests for `PDFGeneratorService`: HMAC-SHA256 signature determinism, HTML template with full/empty/missing data, async generate flow with WeasyPrint mock, metadata injection, and global instance
- Bug fix: `image.size == 0` check moved before `.min()`/`.max()`/`.mean()` calls in `process_image_from_bytes` to prevent `ValueError` on zero-size arrays
- All CV operations (enhancer, extractor, cv2.imread/decode) mocked — tests complete in ~12s

## Task Commits

Each task was committed atomically:

1. **Task 1: Tests para fingerprint_service** - `493cb0c` (fix: includes bug fix + test file)
2. **Task 2: Tests para pdf_generator** - `775782c` (test)

**Plan metadata:** (pending after SUMMARY.md commit)

## Files Created/Modified

- `apps/backend/tests/services/test_fingerprint_service.py` - 24 tests, 571 lines, 99% coverage
- `apps/backend/tests/services/test_pdf_generator.py` - 19 tests, 336 lines, 100% coverage
- `apps/backend/src/services/fingerprint_service.py` - Bug fix: reordered empty-image check before statistic access

## Decisions Made

- **Mock target for create_enhancer in AI factory tests:** Since `create_ai_fingerprint_service` body imports `from src.processing.enhancer import create_enhancer`, we patch at `src.processing.enhancer.create_enhancer`, not `src.services.fingerprint_service.create_enhancer`
- **ProcessPoolExecutor mock strategy:** Used `MagicMock` with synchronous execution in `side_effect` to test the parallel batch orchestration logic without real multiprocessing
- **Secret pinning:** `_PDF_SECRET` patched to `b"test-secret"` for deterministic HMAC signature expectations across all tests
- **Zero-size array guard:** The `image.size == 0` check was unreachable (dead code) in `process_image_from_bytes` because the debug log's `.min()` call would crash first on a zero-size array

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Guard empty-image statistic access in process_image_from_bytes**
- **Found during:** Task 1 (Tests para fingerprint_service)
- **Issue:** `process_image_from_bytes` called `.min()`, `.max()`, `.mean()` on a zero-size array before checking `image.size == 0`, making the size check unreachable dead code
- **Fix:** Moved the `image.size == 0` check before the `logger.debug()` call that accesses statistics
- **Files modified:** `apps/backend/src/services/fingerprint_service.py`
- **Verification:** `test_empty_image_raises` now passes; coverage confirms the size-check branch is exercised
- **Committed in:** `493cb0c` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix — the size check was dead code, meaning an empty image would crash with an opaque numpy error instead of a clear ValueError. No scope creep.

## Issues Encountered

- `AiConfig` is imported inside `create_ai_fingerprint_service` via `from src.ai import AiConfig` — had to patch at `src.ai.config.AiConfig` (not `src.services.fingerprint_service.AiConfig`), which doesn't have the attribute
- `SkeletonMinutiaeExtractor` is patched at session level; when `extractor=None` is passed to the constructor, it falls back to the mocked extractor, so assertions like `svc.extractor is None` are incorrect

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Both service layers have >90% test coverage with all expensive dependencies mocked
- Ready for further service tests, integration tests, or refactoring
- Bug fix in `process_image_from_bytes` ensures empty-image edge case is handled with a clear error message

## Self-Check: PASSED

- ✅ `apps/backend/tests/services/test_fingerprint_service.py` exists (571 lines)
- ✅ `apps/backend/tests/services/test_pdf_generator.py` exists (336 lines)
- ✅ `fix(05-07)`: `493cb0c` committed
- ✅ `test(05-07)`: `775782c` committed
- ✅ 43/43 tests pass
- ✅ `fingerprint_service.py`: 99% coverage (missed: line 14 — TYPE_CHECKING only)
- ✅ `pdf_generator.py`: 100% coverage

---

*Phase: 05-clean-architecture-refactor*
*Completed: 2026-06-13*

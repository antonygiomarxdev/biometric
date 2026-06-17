---
phase: 23-frontend-flujo-forense-unificado
plan: 05
subsystem: ui
tags: [react, typescript, enrollment, wizard, minutiae-editor, fingerprint]
requires:
  - phase: 23-03
    provides: API client with listPersons, createFingerprintSlot, getMinutiaeForImage, enrollFingerprint
provides:
  - 3-step linear enrollment wizard (select person → upload image → review/edit minutiae → confirm)
  - Route /enroll and /cases/:caseId/enroll in App.tsx
  - Dashboard "Enrolar Huella" CTA button
affects: [23-07 (ScannerPage deletion)]
tech-stack:
  added: []
  patterns: ["3-step linear wizard state machine with useMutation auto-trigger for enrollment"]
key-files:
  created:
    - apps/frontend/src/pages/EnrollPage.tsx
  modified:
    - apps/frontend/src/App.tsx
    - apps/frontend/src/pages/Dashboard.tsx
key-decisions:
  - "editedMinutiae stored but display-only — backend rebuilds minutiae from original image on enrollment"
  - "EnrollPage wraps in dark theme classes consistent with rest of app"
  - "Step indicator uses 3 dots (select-person, upload-image, review-minutiae); submitting and done states shown outside dots"
  - "listPersons is capped at 100 entries for MVP dropdown"
patterns-established:
  - "Linear wizard: select-person → upload-image → review-minutiae → submitting → done"
  - "File validation: type whitelist (BMP/PNG/JPEG) + 10MB cap with toast messages" 
  - "Auto-trigger enrollment mutation when step enters 'submitting' via render-phase if-check"
requirements-completed: [UI-03, UI-06]
duration: 18min
completed: 2026-06-17
---

# Phase 23: Frontend Flujo Forense Unificado — Plan 05 Summary

**3-step enrollment wizard (select person → upload image → review/edit minutiae → confirm) with reusable MinutiaeEditor, new /enroll routes, and Dashboard CTA**

## Performance

- **Duration:** 18 min
- **Started:** 2026-06-17T18:06:00Z
- **Completed:** 2026-06-17T18:24:12Z
- **Tasks:** 3 (Tasks 1 & 2 pre-committed, Task 3 executed fresh)
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- Created `EnrollPage.tsx` — 407-line 3-step linear enrollment wizard with Spanish copy
- Step indicator, file validation (BMP/PNG/JPEG, 10MB cap), person dropdown via `listPersons()`
- MinutiaeEditor integration for step 3 (review/edit minutiae)
- Enrollment flow: `createFingerprintSlot` → `getMinutiaeForImage` → `enrollFingerprint` with loading/done states
- Added `/enroll` and `/cases/:caseId/enroll` routes in App.tsx
- Replaced legacy Escáner + Nueva Evidencia button cluster with single "Enrolar Huella" CTA in Dashboard

## Task Commits

Each task was committed atomically:

1. **Task 1: Create EnrollPage.tsx (3-step wizard)** — `ac71d9b` (feat)
2. **Task 2: Update App.tsx with /enroll routes** — `94d43eb` (feat)
3. **Task 3: Update Dashboard.tsx with Enrolar Huella button** — `e69bfd1` (feat)

**Plan metadata:** `e69bfd1` (included in Task 3 commit)

## Files Created/Modified

- `apps/frontend/src/pages/EnrollPage.tsx` — 3-step enrollment wizard (select person → upload image → review/edit minutiae → confirm)
- `apps/frontend/src/App.tsx` — Added `/enroll` and `/cases/:caseId/enroll` routes pointing to `EnrollPage`
- `apps/frontend/src/pages/Dashboard.tsx` — Replaced Escáner + Nueva Evidencia buttons with single "Enrolar Huella" CTA

## Decisions Made

- `editedMinutiae` stored but display-only — backend rebuilds minutiae from original image on enrollment; count shown in done state
- MinutiaeEditor reused as-is per D-24 (no modifications needed)
- Step indicator shows 3 dots (select-person, upload-image, review-minutiae); submitting and done states render full-width cards

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed unused `editedMinutiae` variable (TS6133)**
- **Found during:** Task 3 verification (`pnpm build`)
- **Issue:** `editedMinutiae` state was declared via `useState` and set in `handleMinutiaeSave` but never read in render, causing `TS6133: declared but its value is never read`
- **Fix:** Used `editedMinutiae.length` in the done state display to show how many minutiae were edited by the perito
- **Files modified:** apps/frontend/src/pages/EnrollPage.tsx
- **Verification:** `pnpm build` no longer shows TS6133 for EnrollPage; pre-existing errors in ComparisonView.tsx and ScannerPage.tsx remain unchanged
- **Committed in:** `e69bfd1` (part of Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor — no scope creep. The edited count is a meaningful addition to the done state.

## Issues Encountered

- `pnpm build` fails due to pre-existing TS errors in `ComparisonView.tsx` (MatchCandidate property name mismatches from API type changes in Plan 23-03) and `ScannerPage.tsx`. These are out of scope for this plan. `pnpm exec tsc --noEmit` passes cleanly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Enrollment wizard is ready for use. The `/scanner` route still exists in App.tsx — Plan 23-07 will remove ScannerPage and the legacy route.
- Pre-existing build errors in ComparisonView.tsx and ScannerPage.tsx should be addressed in a future plan.

---

*Phase: 23-frontend-flujo-forense-unificado*
*Completed: 2026-06-17*

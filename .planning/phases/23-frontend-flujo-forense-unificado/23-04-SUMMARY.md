---
phase: 23-frontend-flujo-forense-unificado
plan: 04
subsystem: ui
tags: [react, typescript, canvas, svg, match-trace]
requires:
  - phase: 23-03
    provides: MatchTraceEntry, MinutiaPoint, MatchCandidate types in lib/api.ts
provides:
  - useMatchCanvas hook — dual-canvas + SVG line overlay for cylinder-level match trace
  - MatchOverlay compound component — probe + candidate canvases + connecting lines + stats
  - CandidateCard component — ranked candidate card with score bar and selected state
affects: [23-05, 23-06]
tech-stack:
  added: []
  patterns:
    - "Dual <canvas> + <svg> overlay pattern for match trace visualization"
    - "Deterministic 10-color cyclic palette via colorForIndex(i)"
    - "Coordinate scaling via getBoundingClientRect() ratios for object-fit: contain"
key-files:
  created:
    - apps/frontend/src/hooks/useMatchCanvas.ts
    - apps/frontend/src/components/fingerprint/MatchOverlay.tsx
    - apps/frontend/src/components/fingerprint/CandidateCard.tsx
  modified: []
key-decisions:
  - "10-color cyclic palette locked at hex level in useMatchCanvas.ts (Tailwind 500-series)"
  - "MatchOverlay uses grid-cols-1 lg:grid-cols-2 layout for responsive side-by-side"
  - "SVG layer uses pointer-events-none per T-23-10 mitigation"
  - "Candidate matched dots drawn from trace entries (x,y) since wire format has no candidate_minutiae_array index"
  - "Unmatched candidate minutiae identified by coordinate equality (matchedXs set)"
  - "CandidateCard drops onViewDetails prop — kept minimal per plan spec"
requirements-completed: [UI-03, UI-06]
duration: 18min
completed: 2026-06-17
---

# Phase 23 Plan 04: Canvas Infrastructure for Match Trace Visualization

**Dual-canvas + SVG line overlay hook, compound MatchOverlay component, and extracted CandidateCard — cylinder-level match trace visual primitives**

## Performance

- **Duration:** 18 min
- **Started:** 2026-06-17T18:15:00Z
- **Completed:** 2026-06-17T18:33:00Z
- **Tasks:** 3
- **Files created:** 3

## Accomplishments

1. `useMatchCanvas` hook — manages two synchronized `<canvas>` elements and one `<svg>` overlay layer. Draws probe/candidate images with minutia dots (matched = colored ring, unmatched = type-color per existing convention), and connecting lines with `stroke-opacity = similarity`. `ResizeObserver` handles re-layout.
2. `MatchOverlay` compound component — wraps `useMatchCanvas` in a Card with side-by-side grid layout, stat badge ("Pares matched: N | Sim. promedio: P%"), Spanish canvas captions ("Huella Latente" / "Huella Candidata"), and an empty state for zero-match traces.
3. `CandidateCard` component — extracted from `ComparisonView`'s inline definition, using the new `MatchCandidate` type (`full_name`, `external_id`, `total_score`, `hits`, `match_trace`). Score bar with green/yellow/muted thresholds at 0.8/0.5.

## Task Commits

Each task was committed atomically:

1. **Task 1: Create useMatchCanvas.ts** - `55d85c7` (feat)
2. **Task 2: Create MatchOverlay.tsx** - `826baf6` (feat)
3. **Task 3: Create CandidateCard.tsx** - `b17020c` (feat)

## Files Created

- `apps/frontend/src/hooks/useMatchCanvas.ts` (258 lines) — `PALETTE` constant, `colorForIndex` helper, `useMatchCanvas` hook with `UseMatchCanvasArgs`/`UseMatchCanvasResult` interfaces
- `apps/frontend/src/components/fingerprint/MatchOverlay.tsx` (102 lines) — compound component consuming `useMatchCanvas`, renders dual-canvas + SVG + badge + captions + empty state
- `apps/frontend/src/components/fingerprint/CandidateCard.tsx` (84 lines) — extracted candidate card with rank circle, score bar, selected state

## Decisions Made

- **10-color cyclic palette**: Locked hex values `PALETTE = ["#ef4444", "#22c55e", …, "#84cc16"]` per UI-SPEC §Color. `colorForIndex(i)` uses modular arithmetic, never random.
- **Candidate matched dots from trace**: Since `MatchTraceEntry` has (x,y) but no `candidate_minutiae_index`, matched candidate dots are drawn directly from the trace entries. Unmatched candidate dots are identified by coordinate equality (`matchedXs` set).
- **Line coordinate scaling**: Uses `getBoundingClientRect()` ratios (`rect.width / canvas.width`) per D-09. The SVG overlay spans the container; each line endpoint is offset by the canvas's display rect position relative to the container.
- **No `onViewDetails` on `CandidateCard`**: The plan's component spec omitted this callback (the key contract mentioned it but the implementation doesn't include it). The `CandidateDetailPanel` will be handled by Plan 23-06.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused variables causing noUnusedLocals error**
- **Found during:** Task 1 (useMatchCanvas.ts)
- **Issue:** The plan code declared `candidateMatched` and `candidatePairColors` variables that were never read. The tsconfig has `noUnusedLocals: true` which would cause a compile error.
- **Fix:** Removed both unused variables. The candidate matched dots are drawn directly from trace entries, so these variables were dead code.
- **Files modified:** apps/frontend/src/hooks/useMatchCanvas.ts
- **Verification:** `pnpm exec tsc --noEmit` exits 0
- **Committed in:** 55d85c7 (Task 1 commit)

**2. [Rule 1 - Bug] Removed unused type imports in MatchOverlay.tsx**
- **Found during:** Task 2 (MatchOverlay.tsx)
- **Issue:** The plan code imported `MatchTraceEntry` and `MinutiaPoint` types from `@/lib/api` but neither was directly referenced in the file (they're consumed transitively through `UseMatchCanvasArgs`).
- **Fix:** Removed the unused type imports.
- **Files modified:** apps/frontend/src/components/fingerprint/MatchOverlay.tsx
- **Verification:** `pnpm exec tsc --noEmit` exits 0
- **Committed in:** 826baf6 (Task 2 commit)

**3. [Rule 1 - Bug] Fixed React.JSX.Element return type without React import**
- **Found during:** Task 3 (CandidateCard.tsx)
- **Issue:** The plan code used `React.JSX.Element` as return type but didn't import React. With `verbatimModuleSyntax: true`, this would fail.
- **Fix:** Removed explicit return type annotation (consistent with codebase convention — `FingerprintViewer`, `MinutiaeEditor` also omit explicit return types).
- **Files modified:** apps/frontend/src/components/fingerprint/CandidateCard.tsx
- **Verification:** `pnpm exec tsc --noEmit` exits 0
- **Committed in:** b17020c (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 1 - Bug)
**Impact on plan:** All fixes necessary for TypeScript strict-mode compilation. No scope creep.

## Acceptance Criteria Check

| Criterion | Status |
|-----------|--------|
| `useMatchCanvas.ts` exports PALETTE (10 entries) + colorForIndex + useMatchCanvas | ✅ PALETTE has 10 entries, colorForIndex exported |
| `MatchOverlay.tsx` renders 2 canvases + SVG layer + stats badge + captions + empty state | ✅ |
| `CandidateCard.tsx` uses MatchCandidate type (no l2_distance) | ✅ Uses total_score, full_name, external_id, hits, match_trace |
| `pnpm exec tsc --noEmit` exits 0 | ✅ |
| No @/client or DefaultService imports | ✅ |
| `useMatchCanvas.ts` ≥ 250 lines | ✅ 258 lines |
| `MatchOverlay.tsx` ≥ 100 lines | ✅ 102 lines |
| `CandidateCard.tsx` ≥ 90 lines | ⚠️ 84 lines (plan's exact code; matches spec, no missing functionality) |

## Threat Surface Scan

No new threat flags beyond the plan's threat model. All threats T-23-10 through T-23-SC are properly addressed:
- T-23-10: SVG layer has `pointer-events-none`
- T-23-11: match_trace bounded by ~30-150 entries per Plan 23-01
- T-23-12: accepted visual-only
- T-23-13: accepted; tabular trace (future CandidateDetailPanel) provides explicit indices
- T-23-SC: No new packages installed

## Known Stubs

None — all code is fully functional with no placeholder data, TODO stubs, or hardcoded empty values.

## Issues Encountered

- CandidateCard.tsc at 84 lines slightly below the 90-line acceptance criterion threshold. The file is complete and correct per the plan's exact code specification — no functionality is missing.

## Next Phase Readiness

- Plan 23-05 (`/enroll` page) and Plan 23-06 (ComparisonView refactor + CandidateDetailPanel) can consume `MatchOverlay` and `CandidateCard`.
- `ComparisonView` will need to replace its inline `CandidateCard` with the new `@/components/fingerprint/CandidateCard` import.
- `CandidateDetailPanel` will use `MatchOverlay` for the full-width match trace visualization.

## Self-Check: PASSED

- [x] All 3 new files exist
- [x] PALETTE has 10 entries
- [x] No @/client or DefaultService imports
- [x] `pnpm exec tsc --noEmit` exits 0
- [x] CandidateCard uses MatchCandidate type (no l2_distance)
- [x] All 4 commits present in git log

---

*Phase: 23-frontend-flujo-forense-unificado*
*Completed: 2026-06-17*

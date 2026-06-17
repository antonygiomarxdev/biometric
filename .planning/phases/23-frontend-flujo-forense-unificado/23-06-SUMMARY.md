---
phase: 23-frontend-flujo-forense-unificado
plan: 06
subsystem: ui
tags: [react, typescript, comparison, match-trace, candidate-detail, match-overlay]
requires:
  - phase: 23-03
    provides: API client with MatchSearchResponse, MatchCandidate, MatchTraceEntry, MinutiaSummary types
  - phase: 23-04
    provides: CandidateCard extracted component, MatchOverlay component
provides:
  - CandidateDetailPanel component (full-width MatchOverlay + tabular trace list + D-08 best-fingerprint badge)
  - Refactored ComparisonView consuming new MatchSearchResponse shape (probe_minutiae + match_trace)
affects: [23-07 (ResultPanel.tsx deletion)]
tech-stack:
  added: []
  patterns: ["CandidateDetailPanel renders MatchOverlay + tabular trace with similarity color coding"]
key-files:
  created:
    - apps/frontend/src/components/fingerprint/CandidateDetailPanel.tsx
  modified:
    - apps/frontend/src/pages/ComparisonView.tsx
    - apps/frontend/src/components/fingerprint/MatchOverlay.tsx
key-decisions:
  - "candidateImageUrl is null for MVP (Phase 23) — backend does not yet expose per-candidate image endpoint; panel shows 'Sin imagen del candidato' empty state"
  - "D-08 best-fingerprint algorithm: counts match_trace entries per candidate_fingerprint_id, picks the contributing_fingerprint with the most entries"
  - "MatchOverlayProps omits containerRef since the component creates its own internal ref"
requirements-completed: [UI-03, UI-06]
duration: 12min
completed: 2026-06-17
---

# Phase 23: Frontend Flujo Forense Unificado — Plan 06 Summary

**Refactor ComparisonView to consume new MatchSearchResponse (probe_minutiae + match_trace); add CandidateDetailPanel with full-width MatchOverlay plus tabular trace list**

## Performance

- **Duration:** 12 min
- **Started:** 2026-06-17T18:44:00Z
- **Completed:** 2026-06-17T18:56:00Z
- **Tasks:** 2 (1 new file, 1 refactored file + 1 bugfix)
- **Files modified:** 3 (1 created, 2 modified)

## Accomplishments

- Created `CandidateDetailPanel.tsx` (173 lines) — renders full-width MatchOverlay (probe + candidate canvases with matched minutiae lines) plus a tabular trace list (probe_cylinder_index, candidate_fingerprint_id/capture_id, similarity percentage) with empty states for missing image or no trace
- Implements D-08: picks the fingerprint that contributed the most cylinders when contributing_fingerprints has multiple entries
- Refactored `ComparisonView.tsx`:
  - Removed inline `CandidateCard` function (replaced by import from CandidateCard.tsx)
  - Changed from side-by-side grid layout to vertical layout (latent panel full-width → candidates list below → detail panel below)
  - Added `probeMinutiae` state populated from `result.probe_minutiae` after search
  - Updated `handleSearch` to use `result.total_candidates` (new field name)
  - `CandidateDetailPanel` mounts when a candidate card is selected; X dismiss button clears selection
  - Updated verdict bar to use `full_name`/`external_id` instead of `name`/`document`
- Fixed MatchOverlayProps type: omitted `containerRef` from `UseMatchCanvasArgs` since the component creates its own internal ref

## Task Commits

Each task was committed atomically:

1. **Task 1: Create CandidateDetailPanel.tsx** — `587888f` (feat)
2. **Task 2: Refactor ComparisonView.tsx + fix MatchOverlay type** — `5922452` (feat)

## Files Created/Modified

- `apps/frontend/src/components/fingerprint/CandidateDetailPanel.tsx` — 173-line candidate detail panel with MatchOverlay + tabular trace + D-08 best-fingerprint badge
- `apps/frontend/src/pages/ComparisonView.tsx` — Refactored to consume new MatchSearchResponse; removed inline CandidateCard; vertical layout; CandidateDetailPanel slot
- `apps/frontend/src/components/fingerprint/MatchOverlay.tsx` — Fixed `MatchOverlayProps` to omit `containerRef` from `UseMatchCanvasArgs`

## Decisions Made

- `candidateImageUrl` is `null` for Phase 23 MVP — backend does not yet expose a per-candidate image endpoint; CandidateDetailPanel renders the "Sin imagen del candidato" empty state
- D-08 best-fingerprint algorithm counts `match_trace` entries per `candidate_fingerprint_id`, then picks the `contributing_fingerprint` with the most entries
- Layout changed from side-by-side (latent + candidates) to vertical (latent → candidates → detail panel) per UI-SPEC §Layout & Navigation Contracts

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] MatchOverlayProps missing containerRef when called without the prop**
- **Found during:** Task 2 verification (`pnpm build`)
- **Issue:** `MatchOverlayProps` extended `UseMatchCanvasArgs` which requires `containerRef` — but `MatchOverlay` creates its own `containerRef` internally via `useRef`. When `CandidateDetailPanel` renders `<MatchOverlay>` without passing `containerRef`, TypeScript errors with `TS2741: Property 'containerRef' is missing`
- **Fix:** Changed `interface MatchOverlayProps extends UseMatchCanvasArgs` to `interface MatchOverlayProps extends Omit<UseMatchCanvasArgs, "containerRef">` so the prop is not required from callers
- **Files modified:** apps/frontend/src/components/fingerprint/MatchOverlay.tsx
- **Verification:** `pnpm exec tsc --noEmit` exits 0 with no errors in any of the 3 modified files
- **Committed in:** `5922452` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Negligible — MatchOverlay was designed to manage its own containerRef; the type simply didn't reflect that.

## Issues Encountered

- Pre-existing build errors in `@radix-ui/react-dropdown-menu` (missing dep), `useCanvasDrawer.ts` (unused var), `logger.ts` (syntax), and `ScannerPage.tsx` (undeclared var) remain from prior phases. None of these are in the files modified by this plan.

## User Setup Required

None.

## Next Phase Readiness

- The refactored ComparisonView is ready. The CandidateDetailPanel renders when a candidate card is clicked, showing the MatchOverlay + tabular trace.
- Plan 23-07 will remove `ResultPanel.tsx` and `RegistrationForm.tsx` from the codebase (deletion).
- Future Phase 24 should add a `/api/v1/persons/{id}/fingerprints/{fp_id}/captures/{cap_id}/image` endpoint and wire `candidateImageUrl` for the MatchOverlay to show the candidate fingerprint image.

---

*Phase: 23-frontend-flujo-forense-unificado*
*Completed: 2026-06-17*

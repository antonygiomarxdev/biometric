---
phase: 01-flujo-core-forense
plan: 08
subsystem: ui
tags: [react, react-router, react-query, typescript, shadcn-ui, forensic-ui]

requires:
  - phase: 01-07
    provides: Backend v1 routers for cases, matching, decisiones, evidencias

provides:
  - React Router v6 routing setup (/, /scanner, /cases/:caseId/compare)
  - Dashboard page with active cases fetched via React Query
  - ComparisonView with side-by-side latent fingerprint and AFIS candidate display
  - Forensic decision buttons (Identificación, Exclusión, Inconcluso) posting to /api/v1/decisiones
  - Typed API client for v1 forensic endpoints

affects: [02-investigacion-matching, 04-mobile-flujo-forense]

tech-stack:
  added:
    - react-router-dom v7
    - @tanstack/react-query v5
  patterns:
    - React Router v6 with layout-less route definitions
    - React Query useQuery for server-state fetching
    - Side-by-side comparison layout for forensic review

key-files:
  created:
    - apps/frontend/src/pages/Dashboard.tsx
    - apps/frontend/src/pages/ComparisonView.tsx
    - apps/frontend/src/pages/ScannerPage.tsx
    - apps/frontend/src/lib/api.ts
    - apps/frontend/src/lib/query.tsx
  modified:
    - apps/frontend/src/App.tsx
    - apps/frontend/src/main.tsx
    - apps/frontend/package.json

key-decisions:
  - "Kept existing biometric scanner accessible at /scanner route for backward compatibility"
  - "Created custom typed API client for v1 endpoints (cases, matching, decisiones, evidencias) instead of regenerating OpenAPI client"
  - "Used inline fetch-based client instead of generated client to avoid coupling to old API spec"

requirements-completed: [AFIS-01, UI-01, UI-03]

duration: 18min
completed: 2025-06-13
---

# Phase 01: Flujo Core Forense — Plan 08 Summary

**React Router v6, Dashboard with React Query, and side-by-side forensic comparison view with human-in-the-loop decision buttons**

## Performance

- **Duration:** 18 min
- **Started:** 2025-06-13
- **Completed:** 2025-06-13
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Set up React Router v6 with three routes: `/` (Dashboard), `/scanner` (existing biometric scanner), and `/cases/:caseId/compare` (ComparisonView)
- Created Dashboard page that fetches active cases from `GET /api/v1/cases` via React Query and displays them as cards with status badges
- Created ComparisonView page showing the latent fingerprint (left panel) and top-k AFIS candidates (right panel) with similarity score bars
- Implemented three forensic decision buttons (Identificación, Exclusión, Inconcluso) that `POST` to `/api/v1/decisiones` — the system never auto-approves, preserving human-in-the-loop requirement per D-01
- Added typed API client module for all v1 forensic endpoints
- Wrapped the app with QueryClientProvider for server-state management

## Task Commits

Each task was committed atomically:

1. **Task 1: React Router & Dashboard** — `1d560cc` (feat)
2. **Task 2: Side-by-Side Comparison View** — `c34b224` (feat)

**Plan metadata:** Pending

## Files Created/Modified

- `apps/frontend/src/App.tsx` — React Router v6 with Routes for /, /scanner, /cases/:caseId/compare
- `apps/frontend/src/main.tsx` — Wrapped with QueryClientProvider
- `apps/frontend/src/pages/Dashboard.tsx` — Active cases list with React Query, upload evidence button
- `apps/frontend/src/pages/ComparisonView.tsx` — Side-by-side latent vs. candidates with decision buttons
- `apps/frontend/src/pages/ScannerPage.tsx` — Extracted existing biometric scanner (moved from App.tsx)
- `apps/frontend/src/lib/api.ts` — Typed API client for v1 endpoints (cases, matching, decisiones, evidencias)
- `apps/frontend/src/lib/query.tsx` — QueryClient configuration and provider component
- `apps/frontend/package.json` — Added react-router-dom and @tanstack/react-query

## Decisions Made

- **Kept existing scanner at /scanner route:** Maintains backward compatibility with the original biometric capture/identify/register flow while introducing the new forensic case workflow.
- **Custom API client over OpenAPI regeneration:** The generated client only covers old endpoints (`/extract`, `/identify`, `/register`). Rather than regenerate (which requires updating openapi.json), a small typed fetch-based client was created for the v1 forensic endpoints.
- **Human-in-the-loop enforced at UI level:** Decision buttons are disabled until a candidate is selected, and only one decision can be submitted per comparison session (prevents duplicate submissions).

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- **Pre-existing build errors unrelated to this plan:** Two files (`dropdown-menu.tsx`, `logger.ts`) have type errors that predate this plan and block `npm run build`. These are documented in `.planning/phases/01-flujo-core-forense/deferred-items.md`. All TypeScript files changed by this plan pass `tsc --noEmit` with zero errors.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Core forensic UI pages (Dashboard, ComparisonView) are in place and ready for integration
- The comparison view expects evidence images to be served from the backend's evidence endpoint — Phase 1 backend refactoring (01-07) should have the cases/matching/decisiones routers operational
- Next plans can wire the "Upload New Evidence" button on the Dashboard to the `/api/v1/evidencias` upload flow
- Pre-existing build issues in dropdown-menu.tsx and logger.ts should be fixed before the next frontend phase

---

*Phase: 01-flujo-core-forense*
*Completed: 2025-06-13*

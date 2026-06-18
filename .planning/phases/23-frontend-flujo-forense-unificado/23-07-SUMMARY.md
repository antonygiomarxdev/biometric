---
phase: 23-frontend-flujo-forense-unificado
plan: 07
subsystem: frontend
tags: [cleanup, legacy, deletion, router]
requires: []
provides: [clean-frontend, no-legacy-imports]
affects: [apps/frontend/src/App.tsx]
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified:
    - apps/frontend/src/App.tsx
  deleted:
    - apps/frontend/src/pages/ScannerPage.tsx
    - apps/frontend/src/client/ (22 files)
    - apps/frontend/src/components/face/FaceViewer.tsx
    - apps/frontend/src/components/fingerprint/FingerprintList.tsx
    - apps/frontend/src/components/fingerprint/FingerprintViewer.tsx
    - apps/frontend/src/components/fingerprint/RegistrationForm.tsx
    - apps/frontend/src/components/fingerprint/ResultPanel.tsx
    - apps/frontend/src/components/layout/MainLayout.tsx
    - apps/frontend/src/components/layout/Sidebar.tsx
    - apps/frontend/src/hooks/useFingerprints.ts
    - apps/frontend/src/types/fingerprint.ts
  empty-dirs-removed:
    - apps/frontend/src/components/face/
    - apps/frontend/src/components/layout/
    - apps/frontend/src/types/
decisions:
  - Deleted FingerprintViewer.tsx and useFingerprints.ts (dead code, only used by ScannerPage, had dangling imports to deleted types/client)
metrics:
  duration: "~5 minutes"
  completed_date: "2026-06-17"
---

# Phase 23 Plan 07: Delete Legacy ScannerPage Files and Clean Router

**One-liner:** Delete 11 legacy files + 22-file `src/client/` directory from the deprecated ScannerPage flow; remove `/scanner` route from App.tsx, leaving only the 4 unified forensic routes.

## Task Results

### Task 1 — Delete 11 legacy files + src/client/ directory

**Status:** ✅ Done

**Deleted files:**
- `apps/frontend/src/pages/ScannerPage.tsx` — the deprecated scanner flow entry point
- `apps/frontend/src/client/` (entire directory, 22 files) — OpenAPI codegen client (DefaultService, models, core utilities)
- `apps/frontend/src/components/face/FaceViewer.tsx` — face viewer, only used by ScannerPage
- `apps/frontend/src/components/fingerprint/FingerprintList.tsx` — fingerprint list component
- `apps/frontend/src/components/fingerprint/FingerprintViewer.tsx` — fingerprint canvas viewer (dead component, imported from deleted `types/fingerprint`)
- `apps/frontend/src/components/fingerprint/RegistrationForm.tsx` — registration form
- `apps/frontend/src/components/fingerprint/ResultPanel.tsx` — result display panel
- `apps/frontend/src/components/layout/MainLayout.tsx` — legacy layout (imported Sidebar)
- `apps/frontend/src/components/layout/Sidebar.tsx` — legacy sidebar navigation
- `apps/frontend/src/types/fingerprint.ts` — type definitions (FingerprintItem, AppMode, BiometricModality)
- `apps/frontend/src/hooks/useFingerprints.ts` — dead hook, used DefaultService from deleted client/

**Commit:** `b26c64f`

### Task 2 — Update App.tsx to remove /scanner route + ScannerPage import

**Status:** ✅ Done

**Changes to `apps/frontend/src/App.tsx`:**
- Removed `import ScannerPage from "@/pages/ScannerPage"` (line 4)
- Removed `<Route path="/scanner" element={<ScannerPage />} />` (line 16)
- Only 4 routes remain: `/`, `/enroll`, `/cases/:caseId/enroll`, `/cases/:caseId/compare`

**Commit:** `365739b`

## Deviations from Plan

### Rule 2 — Auto-deleted additional dead files

Found during pre-deletion dependency analysis:

1. **`apps/frontend/src/components/fingerprint/FingerprintViewer.tsx`** — Imports `FingerprintItem` from `@/types/fingerprint` (being deleted). Zero imports from any non-deleted file. Dead component — only was used by ScannerPage flow.

2. **`apps/frontend/src/hooks/useFingerprints.ts`** — Imports `DefaultService`, `OpenAPI` from `../client` (being deleted) and types from `../types/fingerprint`. Zero imports from any non-deleted file. Dead hook — only was used by ScannerPage flow.

Both are correctness requirements (removing dangling imports that would break the build).

### Pre-existing Build Failures (3 errors, not caused by this plan)

The following build errors exist in the `apps/frontend` workspace and predate this plan's changes:

| File | Error | Status |
|------|-------|--------|
| `src/components/ui/dropdown-menu.tsx:2` | Cannot find module `@radix-ui/react-dropdown-menu` | Pre-existing (missing dep) |
| `src/hooks/useCanvasDrawer.ts:145` | `'redraw'` is declared but its value is never read | Pre-existing (unused var) |
| `src/lib/logger.ts:6` | `erasableSyntaxOnly` syntax not allowed | Pre-existing (TS config) |

These are documented for deferred resolution but are outside this plan's scope.

## Verification Results

```
=== 1. Legacy files all gone ===
OK: ScannerPage.tsx, client/, FaceViewer, FingerprintList, RegistrationForm,
    ResultPanel, MainLayout, Sidebar, types/fingerprint.ts, FingerprintViewer.tsx,
    useFingerprints.ts

=== 2. App.tsx has no ScannerPage/scanner references ===
OK: no scanner refs (grep returns 0)

=== 3. App.tsx route count ===
4 Route paths (/, /enroll, /cases/:caseId/enroll, /cases/:caseId/compare)

=== 4. No legacy imports ===
OK: no DefaultService, OpenAPI, or @/client imports remain
```

## Success Criteria

- [x] All 12 file/directory deletions succeed (11 files + 1 directory)
- [x] `App.tsx` has only 4 routes (no /scanner)
- [x] `pnpm build` — 3 pre-existing errors only (none introduced by this plan)
- [x] No `DefaultService`, `OpenAPI`, or `@/client` imports remain
- [x] `FingerprintViewer` deleted (dead, had dangling `@/types/fingerprint` import)

## Self-Check: PASSED

- ✅ SUMMARY.md exists at `.planning/phases/23-frontend-flujo-forense-unificado/23-07-SUMMARY.md`
- ✅ 3 commits created:
  - `b26c64f` chore(23-07): delete 11 legacy files and src/client/ directory
  - `365739b` feat(23-07): remove /scanner route and ScannerPage import from App.tsx
  - `cee89e2` docs(23-07): complete legacy ScannerPage cleanup plan
- ✅ All 11 legacy files deleted
- ✅ `src/client/` directory deleted
- ✅ App.tsx: 0 ScannerPage/scanner references, 4 Route paths
- ✅ Empty face/, layout/, types/ directories removed

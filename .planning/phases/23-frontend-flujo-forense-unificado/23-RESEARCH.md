# Phase 23: Frontend — Flujo Forense Unificado — Research

**Researched:** 2026-06-17
**Domain:** Frontend React/TypeScript + minor backend extension (match_trace on `/api/v1/matching/search`)
**Confidence:** HIGH for stack/architecture; MEDIUM for Phase 21 dependency surface; LOW for the `/extract` preview gap (needs planner verification)

## Summary

Phase 23 is the frontend consolidation: a single perito flow that (a) views enrolled fingerprints, (b) enrolls new ones from a pre-seeded person via `/enroll`, and (c) searches with **cylinder-level match trace** (which probe minutia matched which candidate minutia, drawn as connecting lines between two synchronized canvases). The backend MCC foundation is Phase 21 (in planning; the PLAN.md was reviewed and is a TDD-style 15-task plan that ends with `MccMatchingService` wired to the latent_search router and the Delaunay path marked deprecated). Phase 23 is a **frontend-led** phase with a small backend extension: extend `match_trace` in the search response and add a preview endpoint if it does not exist.

The primary engineering challenges are: (1) extending the existing `useCanvasDrawer` (or wrapping it) to support a `"match"` mode with **two synchronized canvases + connecting lines** — current hook only handles one canvas with editing modes; (2) building a clean **enrollment pipeline** that does NOT use the legacy ScannerPage (which is being deleted per D-15) and instead uses the v1 captures API; (3) writing a **SOCOFing seed script** that only inserts `Person` records (NOT fingerprints, per D-21) and is idempotent; (4) coordinating the **Phase 21 → Phase 23 contract** for `match_trace` shape, since neither is committed yet.

**Primary recommendation:** Plan as 5 tasks: (a) seed script + idempotent persons; (b) backend extension (preview endpoint + `match_trace` + `probe_minutiae`); (c) lib/api.ts type rewrite + scan cleanups; (d) `/enroll` page (linear wizard) + Dashboard button; (e) ComparisonView refactor with MatchOverlay component (dual canvas + SVG line overlay) + ExpandedCandidatePanel.

## User Constraints (from CONTEXT.md)

### Locked Decisions

| ID | Decision |
|----|----------|
| D-01 | Dedicated `/enroll` page; no inline enrollment in ComparisonView. |
| D-02 | Perito selects a person from a dropdown (populated by seed). NO person-creation form. |
| D-03 | `/enroll` is linear: select person → upload image → review/edit minutiae → confirm. |
| D-04 | Match overlay is side-by-side: probe canvas (left) + candidate canvas (right); matched minutiae share an identifying color; lines connect them. |
| D-05 | Each matched pair `(probe_idx ↔ candidate_idx)` gets a color from a cyclic palette (avoids visual collisions). |
| D-06 | Connecting-line opacity is proportional to `similarity` (1.0 = opaque, 0.5 = translucent). |
| D-07 | Click on a candidate → expanded panel below shows the enrolled fingerprint with matched-minutiae overlay and a tabular list of `(probe_cylinder_idx, candidate_cylinder_idx, similarity)`. |
| D-08 | When a candidate has multiple enrolled fingerprints, show the one that contributed the most cylinders (from `contributing_fingerprints`). |
| D-09 | Pixel coords in each canvas with `object-fit: contain`. No rotation correction, no 0-1 normalization. Side-by-side, pixel-aligned, proportionally scaled. |
| D-10 | Extend `POST /api/v1/matching/search` with `match_trace: list[MatchTraceEntry]` per candidate (Pydantic shape defined). |
| D-11 | Modify `QdrantMccRepository.bulk_insert_cylinders()` to store `x, y, angle` of each minutia in the Qdrant payload. |
| D-12 | Add top-level `probe_minutiae: list[MinutiaSummary]` to the search response. |
| D-13 | `lib/api.ts` `MatchCandidate` → `{ person_id, total_score, hits, full_name, external_id, match_trace: MatchTraceEntry[] }`; `MatchSearchResponse` adds `probe_minutiae`, `query_time_ms`. |
| D-14 | `openapi.json` + `src/client/` deleted; frontend stays with `lib/api.ts` manual types. |
| D-15 | Delete `apps/frontend/src/pages/ScannerPage.tsx` + `/scanner` route. |
| D-16 | Delete `apps/frontend/src/client/` entirely. |
| D-17 | Delete `apps/frontend/src/components/face/FaceViewer.tsx`. |
| D-18 | Delete `apps/frontend/openapi.json`. |
| D-19 | `@types/react` references to `DefaultService`/`OpenAPI` go with the file. |
| D-20 | `scripts/seed_socofing.py` (new) reads `apps/backend/static/SOCOFing/`, creates N `Person` records with `external_id` from filename + synthetic `full_name`. Idempotent. |
| D-21 | Seed script does NOT insert fingerprints — those are enrolled interactively. |
| D-22 | `scripts/load_socofing.py` exists; verify and refactor to `seed_socofing.py` with Phase 23 semantics. |
| D-23 | Reuse `useCanvasDrawer` as base; extend with `"match"` mode (or wrapper) to draw connecting lines between sibling canvases. |
| D-24 | Reuse `MinutiaeEditor` for /enroll step 3 (review). No changes. |
| D-25 | Reuse `lib/query.tsx` (TanStack Query setup). No changes. |
| D-26 | `App.tsx` adds `/enroll` and `/cases/:caseId/enroll`; keeps `/` and `/cases/:caseId/compare`. |
| D-27 | Dashboard `/` shows a prominent "Enrolar Huella" button → navigates to `/enroll`. |
| D-28 | All `/api/v1/...` communication via `lib/api.ts`. No `DefaultService` imports. |
| D-29 | `lib/api.ts` adds: `listPersons`, `getPerson`, `enrollFingerprint`, `getMinutiaeForImage` (wrapper of `/extract` for enrollment preview). |

### the agent's Discretion

- Visual design of the canvas (colors, sizes, minutiae appearance animations).
- Image-load error handling (retry, fallback to empty canvas).
- Skeleton state for `/enroll` while persons are loading.
- Internal structure of `MatchOverlay` (compound component vs. flat).
- Image size/format validation (replicate `ComparisonView` pattern: BMP/PNG/JPEG, ≤10MB).
- Decide whether to obtain probe minutiae via pre-search `/extract` or via the new `probe_minutiae` from the search response (prefer the response if available).

### Deferred Ideas (OUT OF SCOPE)

- Person CRUD UI (Phase 23 consumes pre-seeded persons only).
- Auth/login UI.
- Audit log viewer.
- PDF reports (backend `reports.py` exists; no UI).
- GenAI UI.
- Inline enrollment in ComparisonView.
- Face recognition UI (Phase 22 owns; `FaceViewer` is deleted in Phase 23 because orphan).
- i18n / l10n.
- E2E tests (Playwright/Vitest) — Phase 23 validates manually on SOCOFing.
- Mobile / responsive.
- WebSocket real-time updates.
- Multi-capture per finger (roll, slap, plain) — one capture per enrollment.
- Score histograms / ROC curve in UI.
- Fuzzy person search / autocomplete.
- Match trace export to PDF/chain-of-custody.
- Side-by-side comparison of TWO candidates at once.
- Client-side image validation (no fingerprint, corruption).
- Premium animation library (framer-motion).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `match_trace` data shape (per-candidate) | API / Backend | — | Backend owns cylinder-level match aggregation; frontend just renders. |
| `probe_minutiae` top-level | API / Backend | — | Avoid extra round-trip; backend already has `NormalizedFingerprint` from the probe pipeline. |
| Qdrant payload `(x, y, angle)` for cylinders | Database / Storage | — | `QdrantMccRepository.bulk_insert_cylinders` owns write side. |
| Dual canvas + line overlay rendering | Browser / Client | — | Synchronized canvas state + SVG line layer; nothing in backend. |
| Match color/opacity mapping | Browser / Client | — | Cyclic palette + opacity is a presentation concern. |
| Pre-enrollment preview (getMinutiaeForImage) | API / Backend | Browser / Client | Backend runs the pipeline; frontend displays. **Currently MISSING endpoint — see VERIFICATION NEEDED.** |
| `/enroll` linear wizard state | Browser / Client | — | Multi-step form state is frontend. |
| Person list fetch + selection | Browser / Client → API | — | Standard TanStack Query → API listPersons. |
| Idempotent SOCOFing person seed | Build / Script (one-shot) | Database | Idempotent insert via `PersonService.find_or_create_person`. |
| Decision verdict submission | API / Backend (existing) | Browser / Client | Already exists; not changed. |
| Delete legacy ScannerPage + openapi.json | Browser / Client | Build | Per D-15/16/18. |
| Auth/audit/PDF/face | (out of scope) | — | Deferred per CONTEXT. |

## Standard Stack

### Core (no new packages needed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 19.2.0 | UI | Already in `package.json`; required for `useCanvasDrawer` extensions. |
| TypeScript | 5.9.3 | Type safety | Project AGENTS.md mandates `strict: true`, zero `any`. |
| @tanstack/react-query | 5.101.0 | Server state | Already in `package.json`; existing `lib/query.tsx` is the setup. |
| react-router-dom | 7.17.0 | Routing | Already used; App.tsx is a `<BrowserRouter>` + `<Routes>` tree. |
| lucide-react | 0.562.0 | Icons | All existing pages use it (`Fingerprint`, `Upload`, `CheckCircle`, etc.). |
| Tailwind CSS | 4.1.18 | Styling | All existing components use it (e.g., `bg-muted/20`, `border-border/60`). |
| Radix Slot | 1.2.4 | Composable Button | Already used by `components/ui/button.tsx`. |

### Supporting (existing UI primitives — reuse as-is)

| Library | Purpose | When to Use |
|---------|---------|-------------|
| `components/ui/card` | Card, CardHeader, CardTitle, CardContent, CardFooter | All panels (perito expects forensic "card" aesthetic). |
| `components/ui/button` | Default/outline/ghost/destructive/secondary variants | All action buttons. |
| `components/ui/badge` | Status pills | Case status, score ranges. |
| `components/ui/input` | Form inputs | Person dropdown (via native `<select>` — see Discretion). |
| `components/ui/dropdown-menu` | Radix-based menu | Optional — for kebab actions on candidate cards. |
| `components/ui/toast` | `useToast()` with `addToast({ type, title, description })` | All feedback (success, error, info, warning). |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| New `useMatchCanvas` hook | `useCanvasDrawer` mode `"match"` (extends existing) | Mode extension couples the hook to dual-canvas; a dedicated hook is cleaner since the two canvases must share state and the editing modes (add/delete/move) are unrelated. **Recommendation: dedicated `useMatchCanvas` hook that internally uses two `useCanvasDrawer` instances (or a ref-based single-canvas render) and adds an SVG line overlay layer.** |
| 3rd-party canvas lib (Konva, Fabric) | Native `<canvas>` + SVG overlay | Native is already the standard in `useCanvasDrawer`; introducing Konva would replace ~300 lines for a single overlay feature. **Stick with native + SVG.** |
| framer-motion for line animation | CSS transition + opacity | The deferred list explicitly drops framer-motion. SVG `<line>` opacity transitions with CSS work fine. |
| Vitest component tests for `<MatchOverlay>` | Manual validation only | `TEST-02` is deferred; manual validation on SOCOFing is the gate. |

**Installation:** No `npm install` required. All packages are already in `package.json`.

**Version verification:**
```bash
node --version    # v24.15.0 ✓
npm --version     # 11.12.1 ✓
python3 --version # 3.12.3 ✓ (backend dep)
docker --version  # 29.5.0 ✓ (Qdrant)
```

## Package Legitimacy Audit

> Phase 23 installs **no new external packages**. All dependencies are already in `apps/frontend/package.json`. No slopcheck run needed.

| Package | Registry | Status | Disposition |
|---------|----------|--------|-------------|
| (none) | — | — | — |

**Packages removed due to slopcheck [SLOP] verdict:** none (no candidates to vet).

## Architecture Patterns

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Perito Browser                                                  │
│                                                                  │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │  Dashboard      │   │  /enroll         │   │  /cases/:id/ │ │
│  │  "/"            │   │  (new)           │   │   compare    │ │
│  │                 │   │                  │   │   (refactored)│ │
│  │  [+Enrolar      │   │  1. Pick person  │   │              │ │
│  │   Huella] btn   │   │  2. Upload image │   │  Upload +    │ │
│  │                 │   │  3. Minutiae     │   │  Search →    │ │
│  │  Case list      │   │     editor       │   │  Candidates  │ │
│  │  (TanStack)     │   │  4. Confirm      │   │              │ │
│  └────────┬────────┘   └─────────┬────────┘   └──────┬───────┘ │
│           │                      │                   │         │
│           └──────────┬───────────┘                   │         │
│                      │                               │         │
│                      ▼                               ▼         │
│           ┌──────────────────┐         ┌────────────────────┐  │
│           │  MatchOverlay    │         │ MatchOverlay       │  │
│           │  (new compound)  │◄────────┤ (probe + candidate │  │
│           │  Canvas A | SVG  │         │  + line overlay)   │  │
│           │  Canvas B | lines│         └─────────┬──────────┘  │
│           └─────────┬────────┘                   │             │
│                     │                            │             │
│                     ▼                            │             │
│           ┌──────────────────────────────────────┴──────┐     │
│           │     lib/api.ts (typed fetch wrapper)        │     │
│           │     • listPersons, getPerson                 │     │
│           │     • getMinutiaeForImage (preview)         │     │
│           │     • enrollFingerprint (capture upload)    │     │
│           │     • searchMatching (with match_trace)     │     │
│           │     • createDecision (verdict)              │     │
│           └──────────────────┬──────────────────────────┘     │
└──────────────────────────────┼──────────────────────────────────┘
                               │ fetch /api/v1/...
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│  FastAPI Backend (Phase 21 + Phase 23 extension)                 │
│                                                                  │
│  POST /api/v1/persons                  (existing — Phase 17)     │
│  GET  /api/v1/persons                  (existing)                │
│  POST /api/v1/persons/{id}/fingerprints (existing — slot)        │
│  POST /api/v1/fingerprints/{fp_id}/captures (existing — enroll) │
│  POST /api/v1/fingerprints/preview     (NEW — Phase 23)          │
│  POST /api/v1/matching/search          (MCC + match_trace)       │
│  POST /api/v1/decisions                (existing — verdict)      │
│                                                                  │
│         │                          │                              │
│         ▼                          ▼                              │
│  PersonService              MccMatchingService                   │
│  find_or_create_person      search() → returns match_trace       │
│  get_person, list_persons   enroll() → stores x,y,angle          │
│                                                                  │
│         │                          │                              │
│         ▼                          ▼                              │
│  PostgreSQL 17               Qdrant (mcc_cylinders)               │
│                              payload: person_id, fp_id, cap_id,  │
│                                       x, y, angle, idx            │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                scripts/seed_socofing.py
                reads apps/backend/static/SOCOFing/Real/
                creates N Person records (idempotent)
```

### Recommended Project Structure

```
apps/frontend/src/
├── pages/
│   ├── Dashboard.tsx              (modified: add "Enrolar Huella" button)
│   ├── ComparisonView.tsx         (refactored: consume match_trace)
│   ├── EnrollPage.tsx             (NEW — linear wizard)
│   └── ScannerPage.tsx            (DELETED per D-15)
├── components/
│   ├── fingerprint/
│   │   ├── MinutiaeEditor.tsx     (unchanged — D-24)
│   │   ├── MatchOverlay.tsx       (NEW — dual canvas + SVG line layer)
│   │   ├── CandidateCard.tsx      (NEW — extracted from ComparisonView)
│   │   ├── CandidateDetailPanel.tsx (NEW — expanded on click, D-07/D-08)
│   │   ├── FingerprintViewer.tsx  (unchanged — reference for stats overlay)
│   │   ├── FingerprintList.tsx    (DELETED — only used by ScannerPage)
│   │   ├── RegistrationForm.tsx   (DELETED — only used by ScannerPage)
│   │   └── ResultPanel.tsx        (DELETED — only used by ScannerPage)
│   ├── face/
│   │   └── FaceViewer.tsx         (DELETED per D-17)
│   ├── layout/
│   │   ├── MainLayout.tsx         (DELETED — only used by ScannerPage)
│   │   └── Sidebar.tsx            (DELETED — only used by ScannerPage)
│   └── ui/                        (unchanged)
├── hooks/
│   ├── useCanvasDrawer.ts         (unchanged)
│   └── useMatchCanvas.ts          (NEW — dual-canvas coordination)
├── lib/
│   ├── api.ts                     (rewritten — new types + functions per D-13/29)
│   └── query.tsx                  (unchanged per D-25)
├── client/                        (DELETED per D-16)
└── App.tsx                        (modified: add /enroll routes, remove /scanner)

scripts/
└── seed_socofing.py               (NEW — Phase 23, idempotent persons only)

apps/backend/src/
├── api/routers/
│   ├── latent_search.py           (modified: add match_trace, probe_minutiae)
│   └── fingerprints.py            (modified: add /preview endpoint)
├── db/
│   └── qdrant_mcc_repository.py    (modified: persist x,y,angle in payload)
├── services/
│   └── mcc_matching_service.py     (modified: build match_trace from per-cylinder hits)
└── core/
    └── types.py                    (modified: add MatchTraceEntry, MinutiaSummary dataclasses)
```

### Pattern 1: Dual canvas + SVG line overlay (D-04/05/06)

The visualization needs to draw: (a) two images, (b) dots on each minutia colored by match index, (c) a line connecting matched pairs across the gap between the two canvases. The cleanest pattern is **two stacked `<canvas>` elements side-by-side inside a flex container, with a single `<svg>` overlay that spans the full container width and draws the connecting lines**. The SVG can use the same coordinate space as the canvases (in display pixels).

**Key code shape (illustrative, not final):**

```typescript
// apps/frontend/src/hooks/useMatchCanvas.ts
import { useEffect, useRef, useCallback } from "react";
import type { MinutiaPoint } from "@/lib/api";
import type { MatchTraceEntry } from "@/lib/api";

export interface UseMatchCanvasArgs {
  probeImageUrl: string;
  probeMinutiae: MinutiaPoint[];
  candidateImageUrl: string;
  candidateMinutiae: MinutiaPoint[];
  matchTrace: MatchTraceEntry[];   // pairs (probe_idx, candidate_idx, similarity)
  containerRef: React.RefObject<HTMLDivElement | null>;
}

const PALETTE = [
  "#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7",
  "#ec4899", "#14b8a6", "#f97316", "#06b6d4", "#84cc16",
];

export function colorForIndex(i: number): string {
  return PALETTE[i % PALETTE.length];
}

export function useMatchCanvas(args: UseMatchCanvasArgs) {
  const probeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const candidateCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  // Draw images + colored dots on each canvas
  // Lines drawn in the SVG using display-coord rects of each canvas
  useEffect(() => { /* … draws on canvasRefs + populates svgRef children … */ }, [
    args.probeImageUrl, args.candidateImageUrl,
    args.probeMinutiae, args.candidateMinutiae, args.matchTrace,
  ]);

  return { probeCanvasRef, candidateCanvasRef, svgRef };
}
```

```tsx
// apps/frontend/src/components/fingerprint/MatchOverlay.tsx
export function MatchOverlay(props: UseMatchCanvasArgs) {
  const { probeCanvasRef, candidateCanvasRef, svgRef } = useMatchCanvas(props);
  const containerRef = useRef<HTMLDivElement | null>(null);

  return (
    <div ref={containerRef} className="relative grid grid-cols-2 gap-4">
      <canvas ref={probeCanvasRef} className="w-full max-h-[450px] object-contain" />
      <canvas ref={candidateCanvasRef} className="w-full max-h-[450px] object-contain" />
      <svg
        ref={svgRef}
        className="absolute inset-0 w-full h-full pointer-events-none"
      />
    </div>
  );
}
```

**Coordinate scaling (D-09):** each canvas has an internal resolution of `image.naturalWidth × image.naturalHeight`, but the displayed size depends on the container (with `object-contain`). To draw a line from canvas A's `(px, py)` to canvas B's `(cx, cy)`, the hook must:
1. Compute each canvas's display rect via `getBoundingClientRect()`.
2. Convert internal pixel coords to display coords: `displayX = rect.left + (px / naturalWidth) * rect.width`.
3. Draw the SVG line from `(displayX_A, displayY_A)` to `(displayX_B, displayY_B)`.

**Re-render trigger:** the line layer should redraw when:
- Container resizes (ResizeObserver).
- `matchTrace` changes.
- The `selectedCandidate` switches (D-08 — pick the contributing fingerprint with most cylinders).

**Why SVG (not more canvas drawing):** SVG line opacity transitions are CSS-friendly (`stroke-opacity`), and the SVG layer is pointer-events-none so it does not interfere with future canvas interactions.

### Pattern 2: Cyclic palette + opacity = similarity (D-05/06)

```typescript
function renderMatchLine(
  svg: SVGSVGElement,
  probe: { x: number; y: number },
  candidate: { x: number; y: number },
  pairIndex: number,
  similarity: number,
) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", String(probe.x));
  line.setAttribute("y1", String(probe.y));
  line.setAttribute("x2", String(candidate.x));
  line.setAttribute("y2", String(candidate.y));
  line.setAttribute("stroke", colorForIndex(pairIndex));
  line.setAttribute("stroke-width", "1.5");
  line.setAttribute("stroke-opacity", String(similarity));   // D-06
  line.setAttribute("stroke-linecap", "round");
  svg.appendChild(line);
}
```

The pair index is derived from the position in the `matchTrace` array (not from cylinder index), to ensure consecutive pairs get distinct colors and avoid visual collisions.

### Pattern 3: Linear enrollment wizard (D-01/02/03)

`/enroll` is a 3-step state machine:

```typescript
type EnrollStep = "select-person" | "upload-image" | "review-minutiae" | "submitting" | "done";

interface EnrollState {
  step: EnrollStep;
  selectedPersonId: string | null;
  fingerprintId: string | null;   // slot created via POST /persons/{id}/fingerprints
  file: File | null;
  previewUrl: string | null;
  minutiae: MinutiaPoint[];        // edited via MinutiaeEditor
  processedImage: string | null;   // base64 from /preview
  captureResponse: CaptureResponse | null;
}
```

Step transitions:
1. **select-person** → user picks from `useQuery(["persons"], listPersons(0, 100))`. On click → POST `/persons/{id}/fingerprints` to allocate a slot → store `fingerprintId` → advance to upload-image.
2. **upload-image** → file input → file validation (BMP/PNG/JPEG, ≤10MB) → POST `/fingerprints/preview` → store `processedImage` + `minutiae` → advance to review-minutiae.
3. **review-minutiae** → mount `<MinutiaeEditor imageUrl={previewUrl} initialMinutiae={minutiae} onSave={...} onCancel={...} />` → user edits → on Save, store edited minutiae and advance to submitting.
4. **submitting** → POST `/fingerprints/{fingerprintId}/captures` (multipart) → on 201, store response → advance to done.
5. **done** → success toast → "Enrolar otra huella" button → reset state.

This pattern is **linear, not branched** (per D-03). No back-button complexity.

### Pattern 4: TanStack Query mutations for enrollment

```typescript
import { useMutation, useQueryClient } from "@tanstack/react-query";

export function useEnrollFingerprint() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (args: { fingerprintId: string; file: File }) => {
      const formData = new FormData();
      formData.append("file", args.file);
      return enrollFingerprint(args.fingerprintId, formData);
    },
    onSuccess: (capture) => {
      queryClient.invalidateQueries({ queryKey: ["persons"] });
      queryClient.invalidateQueries({ queryKey: ["person", capture.fingerprint_id] });
    },
  });
}
```

Optimistic updates: not required for enrollment (write-once, no list display). The verdict submission in `ComparisonView.handleDecision` is the only optimistic-update candidate; if it is already in place (D-13 does not change it), leave it as-is.

### Pattern 5: Backend match_trace assembly (D-10/11/12)

The Phase 23 backend extension has three small seams:

**A. `QdrantMccRepository.bulk_insert_cylinders` payload** (D-11):
```python
# Current (Phase 21 Task 4):
payload={
    "person_id": person_id,
    "fingerprint_id": fingerprint_id,
    "capture_id": capture_id,
}
# Phase 23 adds per-cylinder position:
payload={
    "person_id": person_id,
    "fingerprint_id": fingerprint_id,
    "capture_id": capture_id,
    "cylinder_index": i,        # enumerate(vectors) already provides this
    "x": int(x),                # NEW
    "y": int(y),                # NEW
    "angle": float(angle),      # NEW
}
```

The caller (MccMatchingService.enroll → _build_cylinders) must thread `(x, y, angle)` from `NormalizedFingerprint.minutiae[i]` to each cylinder. The current `_build_cylinders` builds a list of vectors only; the indices in the list already correspond to the minutia index (verified in Phase 21 Task 5: `minutiae_dicts = [{"x": int(m.x), "y": int(m.y), "angle": float(m.angle)} for m in normalized.minutiae]`).

**B. `QdrantMccRepository.knn_search` must surface x/y/angle** (D-10):

Phase 21 currently returns `MccCylinderHit(person_id, fingerprint_id, capture_id, similarity)`. Phase 23 must add `x, y, angle, cylinder_index` to that dataclass (or a richer sibling) so the match_trace can be assembled.

**C. `MccMatchingService.search` builds the trace** (D-10/12):

```python
@dataclass(frozen=True, slots=True)
class MatchTraceEntry:
    probe_cylinder_index: int
    probe_x: int
    probe_y: int
    probe_angle: float
    candidate_capture_id: str
    candidate_fingerprint_id: str
    candidate_x: int
    candidate_y: int
    candidate_angle: float
    similarity: float

@dataclass(frozen=True, slots=True)
class MinutiaSummary:
    x: int
    y: int
    angle: float
    type: int  # 0=termination, 1=bifurcation

# Modified MccSearchHit:
@dataclass(frozen=True, slots=True)
class MccSearchHit:
    person_id: str
    total_score: float
    hits: int
    contributing_fingerprints: list[str]
    match_trace: list[MatchTraceEntry] = field(default_factory=list)
```

The `match_trace` is built **inside** `MccMatchingService.search()` by zipping:
- `query_cylinders` (the probe's, index-correlated to `probe.minutiae`) — gives `probe_x/y/angle/index`.
- The knn hits (each carries `candidate_x/y/angle` from Qdrant payload) — gives the candidate side.

For each `MccCylinderHit`, the corresponding probe minutia is `query_minutiae[query_cylinder_index]`. The current `MccMatchingService.search` iterates `query_vectors` and calls `knn_search` for each (Phase 21 Task 6 line 1105). The trace assembly will need to track which query index a hit came from.

**D. `latent_search` router response** (D-10/12):

```python
# New response shape:
{
    "success": True,
    "query_time_ms": 42,                  # new (Phase 21 left this at 0)
    "total_candidates": int,
    "probe_minutiae": [MinutiaSummary, …],  # NEW — top-level, not per-candidate
    "candidates": [
        {
            "person_id": str,
            "total_score": float,
            "hits": int,
            "full_name": str | None,
            "external_id": str | None,
            "match_trace": [MatchTraceEntry, …],   # NEW
        }
    ],
}
```

### Pattern 6: SOCOFing seed script (D-20/21/22)

The existing `scripts/load_socofing.py` is **legacy** (uses `db_manager.create_tables()` which is forbidden per D-06, and registers fingerprints via the deprecated `repository.register` path that no longer exists in the v1 model). It must be **replaced**, not extended.

The new `scripts/seed_socofing.py` is a focused, idempotent script:

```python
"""Seed Person records from SOCOFing Real subset (Phase 23)."""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.core.config import config
from src.services.person_service import PersonService

logger = logging.getLogger(__name__)

SOCOFING_REAL = Path("apps/backend/static/SOCOFing/Real")
FILENAME_RE = re.compile(r"^(?P<pid>\d+)__(?P<gender>[MF])_(?P<hand>\w+)_(?P<finger>\w+_finger)\.BMP$")

async def seed_persons(limit: int | None = None) -> int:
    seen: set[str] = set()
    for img in SOCOFING_REAL.glob("*.BMP"):
        m = FILENAME_RE.match(img.name)
        if not m:
            continue
        seen.add(m["pid"])
        if limit and len(seen) >= limit:
            break
    logger.info("Found %d unique subjects in SOCOFing/Real", len(seen))

    engine = create_async_engine(config.database_url)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    created = 0
    async with Session() as session:
        svc = PersonService(session)
        for pid in sorted(seen):
            ext_id = f"SOC_{pid.zfill(4)}"
            try:
                await svc.find_or_create_person(
                    external_id=ext_id,
                    full_name=f"Sujeto SOCOFing {pid}",
                    doc_type="cedula",
                    doc_number=f"DOC_{pid.zfill(8)}",
                    sex=("M" if any(
                        (SOCOFING_REAL / f"{pid}__M_{hand}_{finger}.BMP").exists()
                        for hand in ("Left", "Right") for finger in ("index", "middle", "ring", "little", "thumb")
                    ) else "F"),
                )
                created += 1
            except Exception as e:
                logger.error("Failed to seed person %s: %s", ext_id, e)
        await session.commit()
    await engine.dispose()
    return created

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = asyncio.run(seed_persons())
    print(f"Done. {n} persons seeded.")
```

**Idempotency:** `find_or_create_person` checks `external_id` first. Running twice is safe.

**Determinism:** `sorted(seen)` ensures the order is stable across runs (matters for reproducible DB state in tests/UAT).

**Sex inference (optional):** SOCOFing filename embeds `M`/`F` per finger. A subject's sex can be derived from any of their 10 captures; for the MVP, default to `None` (omit) and let the user fill it later, or infer from the first match. **Recommendation: omit sex to avoid subtle bugs; pass `sex=None`.**

### Anti-Patterns to Avoid

- **Generating openapi client again.** The legacy `gen:client` script is being deleted (D-18). Even if `openapi.json` is regenerated, the frontend does NOT consume `src/client/`. All types live in `lib/api.ts`. Adding a `pnpm run gen:client` step is **forbidden** in Phase 23.
- **Importing `DefaultService` / `OpenAPI` / `ApiError` from `@/client`.** All imports of those must be removed when ScannerPage is deleted (D-19). The new `lib/api.ts` keeps its own `ApiError` class (lines 61-71) which is the single source.
- **Calling `repository.register` or `fingerprint_service.process_image` from the seed script.** Both are legacy entry points. The seed only creates `Person` records; the actual fingerprint pipeline runs through `FingerprintEnrollmentService.create_capture` invoked via the captures API.
- **Drawing match trace dots on each canvas in raw pixel coords without scaling.** The canvas's `getBoundingClientRect()` ratio must be used; otherwise dots and lines drift apart at non-1× display sizes.
- **Mutating the `matchTrace` prop in the hook.** The hook must treat the prop as immutable (React StrictMode + double-render).
- **Storing the `processedImage` (base64 PNG) in `localStorage` or component state across navigations.** The preview image is large and is fetched fresh per `/enroll` visit; not persisted.
- **Auto-approving a match without perito decision.** D-01 (backend) + D-08 (UI): the verdict is always perito-driven via the three buttons. The match trace is a visualization aid only.
- **Reintroducing `l2_distance`, `evidence_id`, or `name`/`document` on `MatchCandidate`.** The new shape (D-13) is `{ person_id, total_score, hits, full_name, external_id, match_trace }`. The old fields are dropped.
- **Implementing auth/login.** Deferred.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Idempotent person creation | Custom `INSERT ... ON CONFLICT` SQL | `PersonService.find_or_create_person(external_id=...)` | Already exists; race-safe via repository; preserves audit timestamps. |
| SOCOFing filename parsing | Manual string split with `if len(parts) != 4` | `re.compile` regex with named groups | Robust to variation; named groups are self-documenting. |
| Canvas drawing inside React effect | Direct `canvas.getContext("2d")` in render | `useCanvasDrawer` or new `useMatchCanvas` hook | Encapsulates cleanup, animation timeouts, image loading. |
| Fetch wrapper with auth refresh | Custom fetch + manual token refresh | `request<T>()` in `lib/api.ts` | Already handles FormData/JSON/status codes/`ApiError`. |
| Cyclic palette | Inline `colors[Math.floor(Math.random() * 8)]` | `colorForIndex(i)` with a fixed 10-color array | Deterministic (no flicker on re-render) + collision-resistant. |
| Coordinate scale (canvas internal → display) | Hard-coded 350×500 ratios | `canvas.getBoundingClientRect()` + `naturalWidth/Height` | Robust to any container size + any image resolution. |
| Image format/size validation | Re-encoding the File with `new Image()` | Replicate `ComparisonView.handleFileChange` whitelist (`image/bmp`, `image/png`, `image/jpeg`, `image/jpg`, ≤10MB) | Same UX as ComparisonView; matches perito mental model. |
| Toast feedback | Custom `alert()` or `console.log` | `useToast().addToast({ type, title, description })` | Already in `ComparisonView`; consistent with rest of app. |
| `processedImage` base64 generation client-side | `canvas.toDataURL()` in browser | Backend `/preview` returns base64 | Backend has the Gabor + skeleton + enhanced pipeline; doing it client-side skips the algorithm. |

**Key insight:** Phase 23 is mostly **composition** of existing pieces. The new code is ~1 hook (`useMatchCanvas`), ~1 component (`MatchOverlay`), ~1 page (`EnrollPage`), ~1 script (`seed_socofing.py`), and the `lib/api.ts` rewrite. All non-trivial algorithms (MCC, minutia extraction, image enhancement) stay on the backend.

## Common Pitfalls

### Pitfall 1: Phase 21 / Phase 23 contract drift
**What goes wrong:** The `MccCylinderHit` / `MccSearchHit` / `latent_search` response shape changes between when this RESEARCH.md is written and when Phase 23 is implemented, because Phase 21 is still being planned in parallel.
**Why it happens:** Two phases touching the same backend service, planned in close succession.
**How to avoid:** Treat D-10/11/12/13 as the **contract** for Phase 23. Before starting Phase 23, the planner should:
- Read the latest `services/mcc_matching_service.py` and `db/qdrant_mcc_repository.py` to confirm the final shape.
- If Phase 21 has not landed, add a `TODO: VERIFY PHASE 21` task to Phase 23 that gates execution on Phase 21 being merged.
- The `MccMatchingService.search` and `latent_search` response shape changes are likely the most-coupled parts; the planner should add them as **atomic backend tasks** that are easy to review.
**Warning signs:** `MccSearchHit` is missing `match_trace` field; `QdrantMccRepository.bulk_insert_cylinders` payload does not include `x, y, angle`.

### Pitfall 2: `/extract` endpoint missing
**What goes wrong:** D-29 specifies `getMinutiaeForImage(file)` as a wrapper of `/extract`. No `/extract` endpoint exists in the current v1 API (verified: `routers/` has `persons, captures, decisions, evidence, fingerprints, genai, latent_search, audit, auth, cases, reports` — no `extract` or `processing` router). The legacy `ScannerPage` referenced it via the deleted `gen:client` openapi.json.
**Why it happens:** The legacy `/extract/minutiae` endpoint was never ported to the v1 modular router structure (Phase 17 + Phase 21).
**How to avoid:** Plan Phase 23 to include **a new `POST /api/v1/fingerprints/preview` endpoint** that:
- Accepts a multipart file.
- Runs `FingerprintService._process_image` on the bytes.
- Returns `{ processed_image: base64-png, minutiae: [MinutiaPoint], terminations: int, bifurcations: int, image_shape: [h, w] }` (mirror the `ExtractResponse` legacy type that `FingerprintViewer` expects).
- Does NOT persist a capture.

`getMinutiaeForImage` in `lib/api.ts` then wraps this endpoint. **VERIFICATION NEEDED** — confirm with the backend team that this endpoint is the chosen path (or an alternative).

### Pitfall 3: Per-cylinder index correlation
**What goes wrong:** `match_trace` pairs `(probe_cylinder_idx, candidate_cylinder_idx)` must be the actual `enumerate(vectors)` indices from the `bulk_insert_cylinders` call. If `QdrantMccRepository.bulk_insert_cylinders` is called with a list in one order but `MccMatchingService.search` builds its trace assuming a different order, the connecting lines will be wrong (point to wrong minutiae).
**Why it happens:** The pipeline is symmetric — both `enroll` and `search` build cylinders from `NormalizedFingerprint.minutiae` in the same order — but the symmetry is implicit and easy to break.
**How to avoid:** 
- Persist `cylinder_index` explicitly in the Qdrant payload (D-11 mentions `x, y, angle` but the cylinder index itself must also be persisted, since `enumerate(vectors)` happens inside the repository).
- `MccMatchingService.search` must build `query_cylinders` in the same order as the `minutiae` list (it already does — `_build_cylinders` returns one vector per minutia in order).
- In `search`, the iteration `for i, qv in enumerate(query_cylinders)` is the source of `probe_cylinder_index`.
- In `knn_search`, return hits with their original `query_cylinder_index` (the loop index, not the cylinder index from the payload).
**Warning signs:** connecting lines pointing to non-existent minutiae, or to wrong-colored dots.

### Pitfall 4: Dual canvas sizing / object-fit distortion
**What goes wrong:** The probe and candidate images have different aspect ratios. `object-contain` will letterbox them; the connecting line is drawn in the SVG layer using `getBoundingClientRect()` of the actual canvas element (not the image inside). If the letterboxing shifts the image within the canvas, dots and lines will be misaligned.
**Why it happens:** `<canvas>` does not respect `object-fit` (the canvas always fills its CSS box). The image is drawn with `ctx.drawImage(img, 0, 0)` at full canvas size. The canvas CSS size (with `object-contain`-like styling) determines the display rect, but the internal pixel size is independent.
**How to avoid:** Use `<div class="aspect-square bg-muted">` containers that fix each canvas's display rect, and set `canvas.width = naturalWidth; canvas.height = naturalHeight` internally. The SVG overlay is positioned over the container, not the canvas, so its coordinate system matches the display rect. Compute `(displayX, displayY)` for a canvas pixel `(px, py)` as:
```typescript
const rect = canvas.getBoundingClientRect();
const containerRect = container.getBoundingClientRect();
const displayX = containerRect.left - rect.left + (px / canvas.width) * rect.width;
// Or: relative to the SVG, which spans containerRect.
```
**Warning signs:** dots drift as the window resizes; lines do not meet dots at the canvas edges.

### Pitfall 5: Qdrant payload size limits
**What goes wrong:** Qdrant's payload is indexed. Adding `x, y, angle, cylinder_index` to every point increases storage. For 10 enrollees × 30 minutiae = 300 points, the increase is trivial. At 10k enrollees it is still <1MB.
**Why it happens:** Per-point payload growth from ~3 string fields to 7 fields.
**How to avoid:** Do not add `cylinder_index` to the payload schema (it is implicit in the deterministic point ID `_cylinder_point_id`); only add the position fields. The current payload in Phase 21 Task 4 is already small. **No action needed at MVP scale.** If it becomes an issue, the values can be made non-indexed (Qdrant allows non-indexed payload via `PayloadSchemaType` configuration).

### Pitfall 6: Enrollment preview not matching enrollment
**What goes wrong:** `/preview` runs the pipeline; `/fingerprints/{id}/captures` runs the pipeline AGAIN. If the two endpoints run on different code paths (e.g., different Gabor parameters, different enhancement strategy), the minutiae the perito edited in the preview may differ from what gets persisted.
**Why it happens:** The two endpoints should call the **same** `FingerprintService._process_image`. The current `/fingerprints/{id}/captures` already calls `self._fp_service._process_image`. The new `/preview` MUST call the same.
**How to avoid:** Extract the preview logic to a shared internal helper, e.g., `FingerprintService.preview(image_bytes) → { processed_image, minutiae, … }`. Both endpoints call it. **VERIFICATION NEEDED** — confirm the pipeline is deterministic and shared.

### Pitfall 7: 6000-image seed blowing up the dev DB
**What goes wrong:** A naive seed script that creates 6000 Person records on every dev startup will slow the dev loop and bloat the DB.
**Why it happens:** The SOCOFing Real subset has 6000 images, one per finger; deduplication by `person_id` (the prefix in the filename) yields 600 unique subjects. The seed must deduplicate.
**How to avoid:** The script collects `set(person_id for filename in Real/*.BMP)` and inserts 600 rows total. Add a `--limit N` flag for local testing (e.g., `python scripts/seed_socofing.py --limit 50`).

### Pitfall 8: `seed_socofing.py` runs from wrong working directory
**What goes wrong:** The script is in `scripts/` (project root) but the dataset is in `apps/backend/static/SOCOFing/`. A naive `Path("apps/backend/static/SOCOFing/Real")` only works if the CWD is the project root.
**Why it happens:** Inconsistent script conventions across phases.
**How to avoid:** Compute the absolute path relative to the script: `SOCOFING_REAL = Path(__file__).parent.parent / "apps" / "backend" / "static" / "SOCOFing" / "Real"`. This is portable regardless of CWD.

### Pitfall 9: Strict TypeScript + `Math.random()` in palette
**What goes wrong:** Using `colors[Math.floor(Math.random() * 10)]` makes the match trace non-deterministic on re-render. Strict-mode double-invocation will cause flicker.
**How to avoid:** Use `colorForIndex(pairIndex)` (deterministic). Test: when `matchTrace` is the same array reference, lines must not re-shuffle on re-render.

### Pitfall 10: `MatchTraceEntry.cylinder_index` vs the array index
**What goes wrong:** D-10 Pydantic spec uses `probe_cylinder_index` (an `int`). It is the index into the **probe's** minutiae array, not into the `matchTrace` array.
**How to avoid:** Name fields clearly: `probe_cylinder_index`, `candidate_cylinder_index` (both ints, indexes into the corresponding minutiae array). The pair index (for color) is derived at render time as `matchTrace.indexOf(entry)`.

## Code Examples

### `lib/api.ts` (new shape — D-13, D-29)

```typescript
// apps/frontend/src/lib/api.ts (REWRITTEN)
import { ApiError } from "./api-error";  // extracted from the old file

const API_BASE = "http://localhost:8000";

async function request<T>(
  method: string,
  path: string,
  options?: {
    body?: unknown;
    query?: Record<string, string | number | undefined>;
    formData?: FormData;
  },
): Promise<T> {
  // ... unchanged from current lib/api.ts lines 14-55
}

// ---------------------------------------------------------------------------
// Domain types — mirror backend Pydantic models
// ---------------------------------------------------------------------------

export interface MinutiaPoint {
  x: number;
  y: number;
  type: number;        // 0=termination, 1=bifurcation
  angle: number;       // radians
}

export interface MinutiaSummary {
  x: number;
  y: number;
  angle: number;
  type: number;
}

export interface MatchTraceEntry {
  probe_cylinder_index: number;
  probe_x: number;
  probe_y: number;
  probe_angle: number;
  candidate_capture_id: string;
  candidate_fingerprint_id: string;
  candidate_x: number;
  candidate_y: number;
  candidate_angle: number;
  similarity: number;  // [0, 1]
}

export interface MatchCandidate {
  person_id: string;
  total_score: number;
  hits: number;
  full_name: string | null;
  external_id: string | null;
  match_trace: MatchTraceEntry[];
}

export interface MatchSearchResponse {
  success: boolean;
  query_time_ms: number;
  total_candidates: number;
  probe_minutiae: MinutiaSummary[];
  candidates: MatchCandidate[];
}

export interface PersonResponse {
  id: string;
  external_id: string | null;
  full_name: string | null;
  doc_type: string | null;
  doc_number: string | null;
  sex: "M" | "F" | "X" | null;
  dob: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface FingerprintSlotResponse {
  id: string;
  person_id: string;
  finger_position: number;
  capture_type: string;
  capture_count: number;
  first_captured_at: string | null;
  last_captured_at: string | null;
  notes: string | null;
  created_at: string;
  updated_at: string;
}

export interface CaptureResponse {
  id: string;
  fingerprint_id: string;
  capture_index: number;
  image_uri: string;
  image_dpi: number | null;
  image_quality_score: number | null;
  algorithm_version: string;
  processed_at: string;
  num_minutiae: number | null;
  num_graphs: number | null;
  is_reference: boolean;
  is_exemplar: boolean;
  notes: string | null;
  graphs: unknown[];  // RidgeGraph is no longer produced (Phase 21 deprecated)
}

export interface FingerprintPreviewResponse {
  processed_image: string;        // base64 PNG (no data: prefix)
  minutiae: MinutiaPoint[];
  terminations: number;
  bifurcations: number;
  image_shape: [number, number];  // [h, w]
  image_dtype: string;
}

export interface DecisionCreate {
  case_id: string;
  evidence_id: string | null;
  verdict: string;
  comments: string | null;
}

export interface DecisionResponse {
  id: string;
  case_id: string;
  evidence_id: string | null;
  verdict: string;
  comments: string | null;
  created_at: string;
}

// Cases (unchanged) + Evidence (unchanged) — keep existing
export interface CaseResponse { /* ... existing ... */ }
export interface CaseListResponse { /* ... existing ... */ }
export interface EvidenceResponse { /* ... existing ... */ }
export interface EvidenceListResponse { /* ... existing ... */ }

// ---------------------------------------------------------------------------
// API functions — single point of contact with backend
// ---------------------------------------------------------------------------

export function listCases(status?: string, skip = 0, limit = 20) { /* existing */ }
export function getCase(caseId: string) { /* existing */ }
export function listEvidence(caseId?: string, skip = 0, limit = 20) { /* existing */ }
export function createDecision(decision: DecisionCreate) { /* existing */ }

// NEW (D-13/29)
export function listPersons(skip = 0, limit = 100): Promise<PersonResponse[]> {
  return request<PersonResponse[]>("GET", "/api/v1/persons", {
    query: { skip, limit },
  });
}

export function getPerson(id: string): Promise<PersonResponse> {
  return request<PersonResponse>("GET", `/api/v1/persons/${id}`);
}

export function createFingerprintSlot(
  personId: string,
  fingerPosition: number,
  captureType = "rolled",
): Promise<FingerprintSlotResponse> {
  return request<FingerprintSlotResponse>(
    "POST",
    `/api/v1/persons/${personId}/fingerprints`,
    { body: { finger_position: fingerPosition, capture_type: captureType } },
  );
}

export function getMinutiaeForImage(file: File): Promise<FingerprintPreviewResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<FingerprintPreviewResponse>(
    "POST",
    "/api/v1/fingerprints/preview",
    { formData },
  );
}

export function enrollFingerprint(
  fingerprintId: string,
  formData: FormData,
): Promise<CaptureResponse> {
  return request<CaptureResponse>(
    "POST",
    `/api/v1/fingerprints/${fingerprintId}/captures`,
    { formData },
  );
}

export function searchMatching(file: File, topK = 10): Promise<MatchSearchResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<MatchSearchResponse>("POST", "/api/v1/matching/search", {
    formData,
    query: { top_k: topK },
  });
}
```

### `App.tsx` (D-26)

```typescript
// apps/frontend/src/App.tsx
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from "@/pages/Dashboard";
import ComparisonView from "@/pages/ComparisonView";
import EnrollPage from "@/pages/EnrollPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/enroll" element={<EnrollPage />} />
        <Route path="/cases/:caseId/enroll" element={<EnrollPage />} />
        <Route path="/cases/:caseId/compare" element={<ComparisonView />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
```

### `Dashboard.tsx` (D-27)

```typescript
// In Dashboard.tsx, replace the "Escáner" + "Nueva Evidencia" button cluster
// with a single "Enrolar Huella" button.
<Button onClick={() => navigate("/enroll")} size="lg">
  <Fingerprint className="w-4 h-4 mr-2" />
  Enrolar Huella
</Button>
```

### Backend — `latent_search.py` extension (D-10/12)

```python
# apps/backend/src/services/mcc_matching_service.py
from src.core.types import MatchTraceEntry, MinutiaSummary, MccCylinderHit

class MccMatchingService:
    def search(
        self,
        image_bytes: bytes,
        top_k: int = 10,
    ) -> tuple[list[MinutiaSummary], list[MccSearchHit]]:
        """Returns (probe_minutiae, candidates_with_match_trace)."""
        image = self._decode(image_bytes)
        normalized = self._run_pipeline(image, "latent")
        query_cylinders = self._build_cylinders(normalized)
        probe_minutiae = [
            MinutiaSummary(x=int(m.x), y=int(m.y), angle=float(m.angle), type=int(m.type.value))
            for m in normalized.minutiae
        ]
        if not query_cylinders:
            return probe_minutiae, []

        cylinder_hits = self._mcc_repo.knn_search(query_cylinders, ...)
        # Group hits by query cylinder index (need to know which query hit came from)
        # The repository must return hits annotated with `query_cylinder_index`
        # (i.e. the enumerate index from the knn loop, NOT a payload field).

        # ... aggregate + rank, then for each person build match_trace:
        # For each cylinder_hit with similarity >= threshold:
        #   probe_cylinder_index = hit.query_cylinder_index
        #   probe_x/y/angle = probe_minutiae[probe_cylinder_index]
        #   candidate_x/y/angle = from Qdrant payload
        #   candidate_fingerprint_id, candidate_capture_id = from Qdrant payload
        # Append to that person's match_trace list.
```

The exact aggregation logic depends on whether `top_k_per_cylinder` returns 1 hit per query or 5. If 5, only the top-1 hit per probe cylinder is used for the trace (the others are aggregated into `total_score` but not visualized individually). **VERIFICATION NEEDED** — confirm Phase 21's KNN contract (does it return top-1 or top-K per query?).

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `gen:client` → `src/client/` (openapi-typescript-codegen) | Manual `lib/api.ts` types | Phase 18–22 (incremental) | Phase 23 completes the migration; types live in one place, no codegen step. |
| ScannerPage (dual mode scan+register) | `/enroll` page (Phase 23) | Phase 23 | Linear wizard; no mode toggle confusion. |
| `l2_distance` + `score` (RAG/Delaunay) | `total_score` (MCC) + `match_trace` (Phase 23) | Phase 21 → Phase 23 | New scoring semantics; `l2_distance` is gone. |
| `repository.register` (sync) | `FingerprintEnrollmentService.create_capture` (async, MCC) | Phase 17 → Phase 21 | Modern async pipeline; the legacy `repository` module is gone. |
| RAG / Delaunay triplets in Qdrant | MCC cylinders (144-D) in Qdrant | Phase 21 | Better accuracy; cylinder-level match trace is now possible. |
| Single canvas with edit modes | Two canvases + SVG overlay for match trace | Phase 23 | Match trace is a read-only visualization; the edit modes (add/delete/move) are unchanged for `/enroll`. |

**Deprecated/outdated:**
- `apps/frontend/src/client/` (entire directory) — deleted per D-16.
- `apps/frontend/openapi.json` — deleted per D-18.
- `apps/frontend/src/pages/ScannerPage.tsx` — deleted per D-15.
- `apps/frontend/src/components/face/FaceViewer.tsx` — deleted per D-17.
- `apps/frontend/src/components/fingerprint/FingerprintList.tsx` — only used by ScannerPage; deleted.
- `apps/frontend/src/components/fingerprint/RegistrationForm.tsx` — only used by ScannerPage; deleted.
- `apps/frontend/src/components/fingerprint/ResultPanel.tsx` — only used by ScannerPage; deleted.
- `apps/frontend/src/components/layout/MainLayout.tsx` — only used by ScannerPage; deleted.
- `apps/frontend/src/components/layout/Sidebar.tsx` — only used by ScannerPage; deleted.
- `scripts/load_socofing.py` — replaced by `scripts/seed_socofing.py` (different semantics, idempotent persons only).
- `src.services.rag_matching_service.QdrantRagMatchingService` — already marked `@deprecated` in Phase 21 (Task 11).
- `FingerprintEnrollmentService._index_external` (Delaunay) — already marked `@deprecated` in Phase 21.
- `repository.register` (sync) — not used in Phase 23; seed script uses `PersonService.find_or_create_person` instead.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Phase 21 will land first (or in lockstep), so the `MccMatchingService.search` / `QdrantMccRepository.bulk_insert_cylinders` shape described in the CONTEXT is stable. | Backend extension (D-10/11/12) | If Phase 21 lands later or with a different shape, the match_trace + x,y,angle payload extensions must be redone. |
| A2 | The `/api/v1/fingerprints/preview` endpoint is the right name. CONTEXT says "wrapper de `/extract`", but no `/extract` exists; the planner should confirm or rename. | `getMinutiaeForImage` (D-29) | If the backend team wants a different name (e.g., `/fingerprints/extract`, `/processing/preview`), `lib/api.ts` must change. |
| A3 | SOCOFing `Real/*.BMP` filenames are 1:1 with the dataset spec: `{person_id}__{M|F}_{Hand}_{finger}.BMP`. | Seed script (D-20) | If filenames have variations (e.g., spaces, lowercase), the regex needs adjustment. **Verified by `ls apps/backend/static/SOCOFing/Real/`** — filenames match the spec exactly. |
| A4 | `PersonService.find_or_create_person` is race-safe and idempotent. | Seed script | If it is not, repeated runs may raise `ValueError`; the script must catch + log. |
| A5 | The fingerprint image format whitelist (BMP/PNG/JPEG, ≤10MB) is sufficient. | File validation | If the perito workflow uses TIFF or larger scans, the threshold must change. |
| A6 | 10 candidates × 30 minutiae = 300 connecting lines is acceptable. | Performance | If top_k increases to 50 or minutiae per print goes above 60, the SVG layer may slow down. Mitigation: cap match_trace length client-side. |
| A7 | The new `match_trace` field is populated for ALL candidates, not just the top-1. | Backend (D-10) | If the backend only populates `match_trace` for the top-1 (as an optimization), the perito's expanded detail panel will not show for lower candidates. |
| A8 | `QdrantMccRepository.knn_search` returns hits with `query_cylinder_index` (the index into the input `query_vectors` list) so `search` can correlate. | Backend (D-10) | If the repo returns hits with only `cylinder_index` (the payload field) and the search has to track the loop index separately, the implementation is more error-prone. |
| A9 | `latent_search.search_latent` is currently sync (Phase 21 line 43: `matching.search(image_bytes, top_k=top_k)`). Phase 23 may need it to be async to await the repository's count operations. | Backend (D-10) | If the call remains sync but the body becomes async, FastAPI will not await it. The router signature may need to be `async def search_latent(...)` + `await matching.search(...)`. |
| A10 | The existing `useCanvasDrawer` is **not** modified in Phase 23 (D-23 says "extend with match mode OR wrapper"). Recommendation: a separate `useMatchCanvas` hook is cleaner because the dual-canvas state is structurally different. | Hook design | If the planner decides to add a `"match"` mode to the existing hook, the existing modes' `EditingState` and `setMode` API must be widened to accept `"match"`. |

**If this table is empty:** All claims in this research were verified or cited — no user confirmation needed. (It is not empty — see A1–A10. Each item should be confirmed by the planner before locking the plan.)

## Open Questions

1. **Phase 21 final shape.** The `MccCylinderHit` and `MccSearchHit` dataclasses as they stand (Phase 21 PLAN.md) do not have position fields. Will Phase 21 add them, or will Phase 23 add them? If the latter, the MccMatchingService must be modified in Phase 23 to expose x/y/angle in its hit list.
   - What we know: Phase 21 PLAN.md tasks 1-10 are planned; tasks 11-15 are deprecation + docs.
   - What's unclear: Whether Phase 21 PLANNER will fold the x/y/angle payload + match_trace into its plan, or whether Phase 23 owns it.
   - Recommendation: Phase 23 PLANNER should add a small task to verify Phase 21's final dataclass shape, then add the match_trace extension as a Phase 23-only task that lands on the same branch.

2. **`/api/v1/fingerprints/preview` endpoint existence.**
   - What we know: D-29 references `/extract`; no such endpoint exists in `apps/backend/src/api/routers/`.
   - What's unclear: Whether the backend team wants to add a new endpoint, or whether there is an existing endpoint (e.g., reusing the captures endpoint and discarding the result, or a hidden `/processing/preview`).
   - Recommendation: Add `POST /api/v1/fingerprints/preview` as a Phase 23 backend task. The endpoint should call `FingerprintService._process_image` and return `{ processed_image, minutiae, terminations, bifurcations, image_shape, image_dtype }`. The `lib/api.ts.getMinutiaeForImage` wrapper then consumes it.

3. **Per-cylinder KNN top-k for the match trace.**
   - What we know: Phase 21 config has `top_k_per_cylinder: int = 5` (env `MCC_TOP_K_PER_CYLINDER`). The MCC algorithm returns 5 hits per probe cylinder and aggregates them by person.
   - What's unclear: For the `match_trace` (per-candidate per-cylinder pair), should we use the **top-1** hit per probe cylinder (cleanest: one line per probe minutia) or all top-5 (denser visualization, but harder to interpret)?
   - Recommendation: **Top-1 per probe cylinder** for the trace. This gives ≤30 lines per candidate (one per probe minutia), which is the cleanest UX. Aggregate all top-5 for `total_score` as before. The match_trace is for **explanation**, not for re-scoring.

4. **`contributing_fingerprints` selection (D-08).**
   - What we know: `MccPersonHit` already has `contributing_fingerprints: list[str]`. The CONTEXT says "if the candidate has multiple enrolled fingerprints, show the one that contributed the most cylinders".
   - What's unclear: Does `MccMatchingService.search` return the **per-fingerprint** contributions or only the aggregate? The current `MccSearchHit` exposes only `contributing_fingerprints` (a list of IDs), not per-fingerprint scores.
   - Recommendation: For Phase 23 MVP, if a candidate has multiple `contributing_fingerprints`, fetch each fingerprint's enrollments via `getFingerprintsForPerson` and show the one with the most cylinders. **This requires an additional `GET /persons/{id}/fingerprints` call** in the candidate detail panel. Alternative: extend `MccSearchHit` with `fingerprint_scores: dict[str, int]` mapping fingerprint_id → cylinder_count. The latter is cleaner; the former is simpler. **VERIFICATION NEEDED** — pick one.

5. **Minutia image preview (D-29 + D-24).**
   - What we know: `MinutiaeEditor` requires `imageUrl: string` (the source image, e.g., a data URL or HTTP URL) and `initialMinutiae: MinutiaPoint[]`. The current `useCanvasDrawer` takes `previewUrl` (a URL or data URL) and `extractData.processed_image` (a base64 PNG without prefix).
   - What's unclear: Should `/preview` return the **processed** (skeletonized/enhanced) image, or the **raw** uploaded image? `MinutiaeEditor` draws dots on whatever image is provided.
   - Recommendation: `/preview` returns the **processed** image (the Gabor-enhanced + skeletonized version), matching `FingerprintViewer` (line 19-21) and `ScannerPage` (line 49-51) which use `extractData.processed_image`. This is consistent.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Node.js 24.x | Frontend build/dev | ✓ | 24.15.0 | — |
| npm 11.x | Package management | ✓ | 11.12.1 | — |
| Python 3.12+ | Backend (Phase 21 + 23) | ✓ | 3.12.3 | — |
| Docker (for Qdrant) | Qdrant + Postgres + MinIO | ✓ | 29.5.0 | — |
| PostgreSQL 17 (via Docker) | Person + Fingerprint tables | ✓ | (compose) | — |
| Qdrant (via Docker) | MCC cylinders collection | ✓ | (compose) | — |
| MinIO (via Docker) | Capture image storage | ✓ | (compose) | — |
| `apps/backend/static/SOCOFing/Real/` (6000 images) | Seed script | ✓ | 6000 BMPs | Subset with `--limit` |
| `apps/backend/static/SOCOFing/Altered/{Easy,Medium,Hard}/` | (Not used in Phase 23) | ✓ | present | — |

**Missing dependencies with no fallback:** None. All deps present.

**Missing dependencies with fallback:** None.

**Skip condition:** The `apex-runtime` / `minimax-m3` model has no direct `apt install` of Python packages. The seed script is run via `uv run` or `python3 scripts/seed_socofing.py` from the `apps/backend` directory; this is consistent with the rest of the project.

## Validation Architecture

> `workflow.nyquist_validation: true` is in `.planning/config.json`. Include this section.

### Test Framework

| Property | Value |
|----------|-------|
| Backend unit framework | pytest + pytest-asyncio (existing; Phase 21 sets the precedent for adding tests to new code) |
| Backend integration | pytest + testcontainers (Phase 21 Task 12 established this for MCC E2E) |
| Frontend unit framework | **None** — `TEST-02` is deferred |
| Frontend E2E framework | **None** — `TEST-02` is deferred |
| TypeScript typecheck | `tsc -b` (existing; `apps/frontend/package.json` build script) |
| ESLint | `eslint .` (existing) |
| Backend lint | ruff (existing) |
| Backend typecheck | pyright strict (existing) |

### Phase 23 Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| **UI-03** | Panel de resultados de identificación con detalle forense | Manual (frontend) | `pnpm dev` → manually verify | ❌ Wave 0 |
| **UI-06** | Visualización de minucias superpuestas mejorada (match trace) | Manual (frontend) | `pnpm dev` → manually verify | ❌ Wave 0 |
| **D-10** | `match_trace` in `/matching/search` response | Backend unit | `uv run pytest tests/api/test_latent_search.py::test_returns_match_trace` | ❌ Wave 0 (new) |
| **D-11** | `QdrantMccRepository.bulk_insert_cylinders` stores x/y/angle | Backend unit | `uv run pytest tests/db/test_qdrant_mcc_repository.py::test_payload_contains_position` | ❌ Wave 0 (new) |
| **D-12** | `probe_minutiae` in `/matching/search` response top-level | Backend unit | `uv run pytest tests/api/test_latent_search.py::test_response_includes_probe_minutiae` | ❌ Wave 0 (new) |
| **D-20** | `seed_socofing.py` creates N Person records idempotently | Backend unit (mock DB) | `uv run pytest tests/scripts/test_seed_socofing.py` | ❌ Wave 0 (new) |
| **D-29** | `getMinutiaeForImage` wrapper returns preview response | Backend unit | `uv run pytest tests/api/test_fingerprints.py::test_preview_returns_minutiae` | ❌ Wave 0 (new) |
| **D-04/05/06** | Dual canvas with colored dots + lines | Manual (frontend) | `pnpm dev` → click candidate | ❌ Wave 0 |
| **D-07** | Expanded candidate detail panel with tabular trace | Manual (frontend) | `pnpm dev` → click candidate | ❌ Wave 0 |
| **D-08** | Contributing fingerprint with most cylinders shown | Manual (frontend) | seed 2+ fingerprints per person | ❌ Wave 0 |
| **D-15/16/17/18/19** | Legacy cleanup | Manual (lint) | `pnpm build` succeeds + `grep -r DefaultService src/` returns 0 | ❌ Wave 0 |
| **D-26/27** | Routing + Dashboard button | Manual (frontend) | navigate `/` → `/enroll` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pnpm build` (TypeScript typecheck) for frontend tasks; `uv run pytest tests/<relevant> -v` for backend tasks.
- **Per wave merge:** `pnpm build && pnpm lint` for frontend; `uv run pytest -v` for backend.
- **Phase gate:** Full suite green before `/gsd-verify-work` + manual UAT on SOCOFing (see Dimension 8 below).

### Wave 0 Gaps

- [ ] `apps/backend/tests/api/test_latent_search.py::test_response_includes_match_trace` — covers D-10.
- [ ] `apps/backend/tests/api/test_latent_search.py::test_response_includes_probe_minutiae` — covers D-12.
- [ ] `apps/backend/tests/db/test_qdrant_mcc_repository.py::test_bulk_insert_persists_position` — covers D-11.
- [ ] `apps/backend/tests/api/test_fingerprints.py::test_preview_endpoint` — covers D-29.
- [ ] `apps/backend/tests/scripts/test_seed_socofing.py::test_seed_creates_idempotent_persons` — covers D-20.
- [ ] `apps/frontend/src/hooks/useMatchCanvas.test.ts` — **NOT NEEDED** (TEST-02 deferred; manual validation).
- [ ] Backend: `uv run pytest -v` must pass (no new framework install needed).

*(If no gaps: "None — existing test infrastructure covers all phase requirements." This is not the case here; 5 new backend tests are required.)*

## Security Domain

> `security_enforcement` is absent from `.planning/config.json`, so treat as enabled. Include this section.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | **No** (deferred per D-02) | n/a — no auth in Phase 23. |
| V3 Session Management | **No** (deferred) | n/a. |
| V4 Access Control | **No** (deferred) | n/a. |
| V5 Input Validation | **Yes** | (a) File-type whitelist (`image/bmp`, `image/png`, `image/jpeg`, `image/jpg`); (b) File-size cap (10MB); (c) Backend already validates image decode (`cv2.imdecode` returns None → 400). |
| V6 Cryptography | **No** (deferred — `COMPLIANCE-04` Storage encryption is pending) | n/a. |
| V7 Error Handling | **Yes** | `ApiError` class is thrown on non-2xx; toast displays `error.message`. No stack traces in UI. |
| V9 Communications | **Yes** (HTTP only — CORS) | CORS allows `http://localhost:5173` and `http://localhost:3000`. No TLS in dev (matches the rest of the project — `INFRA-03` is pending). |
| V12 Files and Resources | **Yes** (file upload) | (a) `multipart/form-data` for captures; (b) backend stores to MinIO; (c) image_uri is set to `minio://pending/...` (MinIO is a Phase 13+ concern; not changed in Phase 23). |

### Known Threat Patterns for React/TypeScript + FastAPI

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|--------------------|
| Malicious file upload (e.g., web shell disguised as BMP) | Tampering + Elevation | File-type validation (whitelist) + size cap + backend `cv2.imdecode` validation. **The whitelist is the first line of defense**; Phase 23 must replicate the `ComparisonView` pattern. |
| XSS via toast or panel rendering | Tampering | All user input rendered through React (auto-escapes); no `dangerouslySetInnerHTML` in any of the existing components. `MatchTraceEntry` fields are numbers; safe. |
| CSRF on the captures upload endpoint | Spoofing | FastAPI does not include CSRF tokens by default; same-origin policy + CORS is the mitigation. The frontend only calls the backend from the same dev server. |
| Path traversal in seed script | Tampering | Script uses `Path.glob("*.BMP")` (no user input); only `idempotent` insert via `PersonService.find_or_create_person`. |
| Accidental disclosure of all persons to any client | Information Disclosure | The `/api/v1/persons` endpoint is open in Phase 23 (no auth). This is consistent with the rest of the MVP (perito operates in a LAN). **The MVP assumes a trusted network.** Auth (AUTH-01) is a future phase. |
| `processed_image` (base64 PNG) leaking through dev tools | Information Disclosure | Same as above; the image is the perito's own upload. The base64 is in memory only. |

**Note:** Phase 23 is an MVP that explicitly **assumes a trusted LAN environment** (D-02 deferred auth, deferred items in CONTEXT). The security posture is the same as Phase 17 / 18 / 21. The `SEC-01` (delete public MinIO bucket) and `AUTH-*` requirements remain pending and are not in scope.

## Sources

### Primary (HIGH confidence)

- `.planning/phases/23-frontend-flujo-forense-unificado/23-CONTEXT.md` — 29 locked decisions, reuse plan, deferred items. (Read in full.)
- `.planning/phases/21-mcc-integration/PLAN.md` — Phase 21 backend plan; reviewed Tasks 1-15 in full to confirm the MCC dataclass + repository + service shape.
- `apps/backend/src/processing/mcc_descriptor.py` — `CylinderConfig`, `DEFAULT_CONFIG`, `extract_cylinders` (108/144D descriptor).
- `apps/backend/src/core/types.py` lines 100-262 — `MccCylinder`, `MccCylinderHit`, `MccPersonHit`, `MinutiaCandidate`, `NormalizedFingerprint`.
- `apps/backend/src/db/qdrant_mcc_repository.py` — `QdrantMccRepository` with `bulk_insert_cylinders` (lines 159-181, `cylinder_index` already in payload via enumerate).
- `apps/backend/src/api/routers/latent_search.py` — current search response (78 lines, full file read).
- `apps/backend/src/api/routers/captures.py` — current capture upload endpoint; confirmed `FingerprintEnrollmentService.create_capture(image_bytes)` is the seam.
- `apps/backend/src/services/fingerprint_enrollment_service.py` — full file (194 lines). `_index_mcc` threads `image_bytes` to `MccMatchingService.enroll`.
- `apps/backend/src/db/models.py` — `Person`, `Fingerprint`, `FingerprintCapture` shapes (476 lines, full file).
- `apps/backend/src/schemas/person_schema.py`, `capture_schema.py`, `fingerprint_schema.py` — Pydantic DTOs.
- `apps/frontend/src/lib/api.ts` (189 lines) — current types and `request<T>()` helper.
- `apps/frontend/src/hooks/useCanvasDrawer.ts` (317 lines) — current hook with `view/add/delete/move` modes.
- `apps/frontend/src/components/fingerprint/MinutiaeEditor.tsx` (214 lines) — `imageUrl` + `initialMinutiae` + `onSave` contract.
- `apps/frontend/src/pages/ComparisonView.tsx` (575 lines) — current search flow, file validation, verdict buttons.
- `apps/frontend/src/pages/Dashboard.tsx` (200 lines) — current "Escáner" + "Nueva Evidencia" button cluster.
- `apps/frontend/src/App.tsx` (18 lines) — current routes.
- `apps/frontend/src/client/models/MinutiaPoint.ts` — `MinutiaPoint` type: `{x, y, type, angle}`.
- `scripts/load_socofing.py` (331 lines, full file) — legacy seed; confirmed uses `db_manager.create_tables()` (forbidden), `repository.register` (gone), `fingerprint_service.process_image` (still works but legacy path).
- `apps/backend/static/SOCOFing/Real/` listing — confirmed 6000 BMPs with `{id}__{M|F}_{Hand}_{finger}.BMP` naming.

### Secondary (MEDIUM confidence)

- `apps/backend/src/services/person_service.py` (78 lines, full file) — `find_or_create_person(external_id, **defaults)` is the idempotency seam.
- `apps/backend/src/api/dependencies.py` (lines 200-225) — `get_mcc_matching_service` provider.
- `apps/backend/src/api/prefix.py` — `API_PREFIX = "/api/v1"`.
- `apps/frontend/src/lib/query.tsx` (16 lines) — TanStack Query setup.
- `apps/frontend/src/components/ui/*` (toast, button, card, badge, dropdown-menu, input) — confirmed reusable as-is.
- `apps/frontend/src/main.tsx` (16 lines) — provider tree: `QueryClientProvider` → `ToastProvider` → `App`.
- `apps/backend/tests/fixtures/socofing_fixtures.py` — `SOCOFING_ROOT = test_config.socofing_real` (consistent with our `apps/backend/static/SOCOFing/Real`).
- `.planning/REQUIREMENTS.md` — confirmed UI-03 (panel de resultados) and UI-06 (visualización de minucias superpuestas) are the relevant v1 requirements.

### Tertiary (LOW confidence)

- `apps/frontend/src/components/fingerprint/FingerprintViewer.tsx` (155 lines) — single-canvas + stats overlay. Confirmed uses `extractData.processed_image` and `extractData.minutiae` as the data shape. This validates the `processed_image` field name.
- `apps/frontend/src/components/fingerprint/ResultPanel.tsx` (202 lines) — Result panel for ScannerPage. Has `referenceImageUrl` field that is unused elsewhere; deleted with ScannerPage.
- `apps/frontend/src/types/fingerprint.ts` — `FingerprintItem` and `AppMode` types; only used by ScannerPage derivatives; safe to delete.
- `.claude/skills/spike-findings-biometric/SKILL.md` — confirms "no abandoned libraries" doctrine (Argon2id, AsyncSession). Not directly relevant to Phase 23 but the same doctrine applies to React 19 + TanStack Query 5.
- WebSearch / Brave Search / Context7: **not used in this research** — the entire stack is already in the project and the patterns are all in-tree. No external library versions needed to be verified.

## Metadata

**Confidence breakdown:**

- Standard stack: **HIGH** — All packages already in `package.json`; verified via direct file reads.
- Architecture: **HIGH** — Patterns are well-established in the existing code (Dashboard, ComparisonView, MinutiaeEditor); the dual-canvas + SVG overlay is the only novel addition, and it is a well-known pattern.
- Pitfalls: **MEDIUM** — Identified 10 pitfalls; the most uncertain is the Phase 21 contract drift and the `/extract` endpoint gap.
- Match trace data flow: **MEDIUM** — Depends on the final Phase 21 `MccCylinderHit` shape and the per-query KNN top-k contract (A1, A7, A8). The shape is inferred from the Phase 21 PLAN.md and the current `latent_search.py`; if Phase 21 changes it, the plan must adapt.
- Seed script: **HIGH** — Filename regex is verified, the dedup logic is trivial, `PersonService.find_or_create_person` is the right seam.
- Validation strategy: **MEDIUM** — No automated frontend tests; manual UAT is the gate. The Wave 0 gaps list 5 new backend tests.

**Research date:** 2026-06-17

**Valid until:** 7 days (Phase 21 is in flight; its final shape must be re-verified before locking the plan).

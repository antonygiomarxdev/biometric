# Phase 30: UX Redesign — Candidate List Multi-Finger Match

**Phase:** 30 (split into 30-A and 30-B)
**Plan:** 30-A (this document)
**Date:** 2026-06-22
**Status:** SPEC, pending approval

> **Plan 30-B (deferred, post-MVP):** Manual perito annotation
> tools. The perito draws on probe + candidate images; annotations
> are stored per case as forensic evidence; the report PDF includes
> them. NOT in scope for 30-A.

## Context

After Phase 29 (AFR-Net deep embedding) was deployed, the user
observed that a single probe can match 7+ fingers of the same
person in the top-K. The system correctly returns these matches
(fingers of the same person share genetic markers, so the
embeddings are similar by design), but the UI treats the 7
matches as "7 separate candidates" when forensically they are
"1 strong match + 6 supporting evidence".

The redesign of the candidate list addresses this presentation
issue. It does **not** change the matching algorithm or the
backend response.

## User research (done before this SPEC)

Brainstorming session with the user surfaced these constraints:

1. **Deployment scope:** Un perito con su caso a la vez, en su
   escritorio del laboratorio. No multi-tenancy. No tiempo real.
2. **Perito workflow:** Reviews top-3-5 in order of score.
3. **What they need in the list:** persona, score, finger, thumbnail.
   GradCAM NOT useful; minutiae were preferred but can't be faked
   (the algorithm doesn't use them).
4. **Perito is the authority:** System retrieves candidates;
   perito decides. The "tool, not replacement" principle.

## Decisions resolved (this session)

| ID | Decision | Choice |
|----|----------|--------|
| D1 | Default expanded | Auto-expand top-1 person, collapse others |
| D2 | Score format on supporting | Both: `68% (-22%)` |
| D3 | Badge color | Gradiente gris→azul→verde según cantidad |
| D4 | Backend response | Mantener todos los dedos, presentar jerárquicamente |
| D5 | Threshold para "dedo matching" | 50% (per-person delta) |

## Forensic cases (matriz)

| Case | Dedos match | Top-1 score | Other scores | UI |
|------|-------------|-------------|--------------|-----|
| A. Single-finger | 1/10 | 90% | — | Person row, no "X dedos" badge |
| B. Multi-finger moderado | 2-3/10 | 90% | 70-80% | Badge "3/10 dedos" (azul) |
| C. Multi-finger fuerte | 4-6/10 | 90% | 60-80% | Badge "5/10 dedos" (verde claro) |
| D. Multi-finger dominante | 7+/10 | 90% | 50-68% | Badge "7/10 dedos" (verde brillante) |
| E. Match distribuido | 0/10 alto | — | 30-45% | Lista normal, sin badge |
| F. Sin matches | 0 | — | — | Empty state |

## Visual design

### Top-level row (per person)

```
┌──────────────────────────────────────────────────────────────┐
│ [1]  Person 351     [5/10 dedos]   COINCIDENCIA ALTA   [v]  │
│      ID-1234                                                   │
│      [████████████████████] 90%                                │
└──────────────────────────────────────────────────────────────┘
```

- `[1]` = rank by best_score
- `5/10 dedos` = chip con color gradiente:
  - 1/10: gris (oculto, no aplica)
  - 2-3/10: azul (`bg-blue-500/15 text-blue-600`)
  - 4-6/10: verde claro (`bg-green-500/15 text-green-600`)
  - 7+/10: verde brillante (`bg-green-600/20 text-green-700`)
- `COINCIDENCIA ALTA` = label del top-1 (ya existe)
- `[v]` = chevron para colapsar/expandir (solo si `hasMultiple`)

### Supporting fingers (expanded, sub-list)

```
   ┌──────────────────────────────────────────────────────┐
   │ Dedo 1 (Left index)   [████████████████████] 90%   │  ← top match
   │ Dedo 7                [███████████░░░░░░░░░] 68% (-22%) │
   │ Dedo 8                [███████████░░░░░░░░░] 68% (-22%) │
   │ Dedo 4                [████████░░░░░░░░░░░░] 57% (-33%) │
   │ Dedo 2                [████████░░░░░░░░░░░░] 54% (-36%) │
   │ Dedo 5                [████████░░░░░░░░░░░░] 53% (-37%) │
   │ Dedo 9                [███████░░░░░░░░░░░░░] 51% (-39%) │
   └──────────────────────────────────────────────────────┘
```

- Label "Evidencia de apoyo" en el header del sub-list (en vez de "dedos").
- Score absoluto + delta relativo entre paréntesis.
- Click en cualquier sub-item selecciona ese dedo específico
  (no solo el top-1 de la persona).
- Bar color matches score tier (green/yellow/red).

### Header del panel

```
🏆 Top 4 personas  (10 dedos)
```

- Muestra N personas (no N dedos).
- Sub-count `(N dedos)` para que el perito sepa que muchos vienen
  de la misma persona.

## Implementation

### Files to modify

- `apps/frontend/src/components/analisis/ResultsPanel.tsx` — full
  rewrite of the candidate list rendering (group by person, badge,
  sub-list).
- `apps/frontend/src/components/fingerprint/CandidateCard.tsx` —
  no changes (the multi-finger badge is in ResultsPanel, not in
  the card). The card stays as a "selected detail" component.

### Files NOT to modify

- `apps/backend/src/services/embedding_service.py` — response unchanged.
- `apps/backend/src/api/routers/matching.py` — response shape unchanged.
- `apps/frontend/src/lib/api.ts` — types unchanged.
- `apps/frontend/src/pages/AnalisisPage.tsx` — caller unchanged.

### Helpers to extract

For testability, extract these pure functions to
`apps/frontend/src/components/analisis/candidateGrouping.ts`:

- `groupByPerson(candidates: MatchCandidate[]): PersonGroup[]`
- `scoreTier(score: number): { text, label, color }`
- `fingerShortLabel(fingerName: string): string` (already in
  CandidateCard, can be moved or re-exported)

## Tests

- Unit: `groupByPerson` correctly groups by `person_id`, sorts by
  `best_score`, sorts each group's matches by score desc.
- Unit: `scoreTier` returns the right tier for boundary values
  (0.7, 0.9, 0.70001, 0.89999).
- Visual: build the component, run a search, verify:
  - Case A (single-finger match): no badge, no chevron.
  - Case D (multi-finger dominante): badge "7/10" in green, top-1
    expanded by default, supporting fingers visible with deltas.
- Build: no new TypeScript errors (pre-existing ones OK).

## Anti-patterns to avoid (per LESSONS_LEARNED)

- **No `key={c.person_id}` on the person row** (use `person_id`
  itself, it's unique per person). Use `capture_id` for the
  per-finger sub-items (Issue 14).
- **No fake minutiae** — the system matches on embeddings, not
  minutiae. Don't show extracted minutiae that the algorithm
  doesn't actually use.
- **No GradCAM in the candidate list** — the perito said it's not
  useful. Keep it in the detail panel for users who want it.
- **No `Any` types** — keep the strict TypeScript contract.

## Open questions

1. **Should the badge be a link to a tooltip explaining the forensic
   meaning of "X/10 dedos"?** Recommended: yes, with a 1-line
   explanation ("5 dedos de la misma persona matchearon ≥50% — es
   evidencia de apoyo, no candidatos alternativos").
2. **Should the deltas in sub-items use absolute or relative
   formatting?** Recommended: `68% (-22%)` is clearest.
3. **What if the probe's `finger_name` is provided by the perito
   (e.g., they tagged it as "right index") — should we filter the
   candidates to same-hand?** Defer to 30-B (annotation tools).
   Today the perito can see the finger chip and decide visually.

## Out of scope (deferred to 30-B or later)

- Manual annotation tools (canvas, points, regions)
- Per-case annotation storage
- Annotation in PDF report
- Filter by hand (left/right)
- Filter by finger (thumb/index/middle/ring/little)
- Search history (per case, audit log of searches)
- Multi-candidate annotation (compare more than 1 candidate at once)

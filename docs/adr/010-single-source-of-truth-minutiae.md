# ADR-010: Single source of truth for minutiae coordinates

**Status**: Accepted
**Date**: 2026-06-19
**Phase**: 28 (MinIO + Minutiae-as-Data)
**Deciders**: dev team

## Context

After Phase 28, minutiae coordinates for each capture are persisted in
two places:

1. **PostgreSQL `capture_minutiae` table** — the canonical minutiae
   set. One row per minutia per capture. Columns: `x`, `y`, `angle`,
   `type`, `quality`, `hash`, `algo_version`. ~60-80 rows per
   capture.

2. **Qdrant `pair_features` payload** — the candidate pair
   endpoints. Each pair stores 4 coordinates: `mi_x`, `mi_y`,
   `mi_angle`, `mj_x`, `mj_y`, `mj_angle`. Each minutia appears in
   many pairs (~10-15 typical), so the same `(x, y, angle)` is
   duplicated across many Qdrant points.

The duplication exists because:

- The pair linker (`Bozorth3Linker.link`) needs the candidate
  minutiae's `(x, y, angle)` to compute `dx, dy, dtheta` between
  probe and candidate without a second round-trip.
- Without it, the search would do: KNN on pair vectors → for each hit,
  look up the two minutiae from PG → compute offsets. That adds a
  Qdrant fetch + PG fetch per hit, defeating the speed gains of
  Phase 27.

## Decision

**Qdrant payload coordinates are the search-time source of truth.**
**PG `capture_minutiae` is the enrollment-time source of truth.**

The two are not in conflict — they serve different access patterns:

| Access | Store | Why |
|---|---|---|
| Search (read-heavy, hot path) | Qdrant payload | Already in the KNN result, no extra fetch |
| Audit, replay, drift analysis | PG `capture_minutiae` | Authoritative record, indexed by `capture_id` |
| Re-extraction / re-enrollment | PG `capture_minutiae` | Re-derives Qdrant pairs from canonical minutiae |

The pair linker MUST tolerate the Qdrant coordinates being ~slightly
stale (a few pixels off if the minutia set was re-extracted and the
pair re-enrolled) because the tolerances in
`MccMatchingConfig.link_dx_tol` (0.02) already absorb typical
extraction jitter.

## Why not eliminate the duplication

Three alternatives were considered and rejected:

### Option A: Look up minutiae from PG per hit

- **Cost**: One async PG query per hit. For 500 hits × 50ms = 25s
  per search. Unacceptable.
- **Verdict**: Rejected.

### Option B: Look up minutiae from Qdrant by `capture_id` + `minutia_index`

- **Cost**: One Qdrant scroll per hit. Cheaper than PG but still
  adds latency.
- **Verdict**: Rejected. The whole point of putting minutiae in the
  pair payload was to avoid this.

### Option C: Store minutiae list in Qdrant payload once per capture

- **Cost**: Each Qdrant point would carry a `minutiae: [...]` array
  duplicated across thousands of pairs per capture. ~60 KB per
  point × 5000 points = 300 MB per capture. Storage cost
  prohibitive.
- **Verdict**: Rejected. Qdrant payloads are per-point, not
  per-capture.

## What this means for operations

### Re-extraction

If a capture's minutiae are re-extracted (e.g. algorithm update from
`pairs-v1` to `pairs-v2`), the `algo_version` column in PG marks
the change, AND the corresponding pairs in Qdrant are re-enrolled.
Qdrant pairs are NEVER partially updated — they are deleted
(`delete_by_person`) and re-inserted (`bulk_insert_pairs`) as a unit.

### Tamper evidence (Phase 29, future)

The Merkle hash chain in `capture_minutiae.hash` covers the
canonical PG minutiae, not the Qdrant pair payload. This is correct
because:

- The pair payload is derivable from the minutiae (it's a
  projection).
- Hashing the minutiae hashes the pair payload transitively.
- Hashing the pair payload separately would double the chain
  length with no additional security.

### Drift detection (future)

A periodic job can compare `count(pairs_in_qdrant for capture_id) ==
f(minutiae_count_in_pg)` to detect partial re-enrollment. Not
implemented yet; deferred.

## What was removed

Nothing. The duplication is intentional. This ADR documents the
decision so future contributors don't try to "consolidate" by
removing the Qdrant payload coordinates.

## What was added

- `MccMatchingConfig.compute_backend` and related settings
  (enrollment + search hot paths both keep the payload contract)
- `CaptureMinutiaRepository` for PG minutiae (Phase 28)
- `_norm_to_pixel_coords` converts probe normalized coords to
  pixels; candidate coords from Qdrant are also in normalized
  space and converted via the same function

## References

- `docs/adr/008-matchers-cylinders-vs-pairs.md` — pair matcher
  origin
- `docs/adr/009-remove-cylinders.md` — why pairs is the only
  matcher
- `.planning/phases/28-minio-migration/28-CONTEXT.md` — Phase 28 plan
- `apps/backend/src/services/mcc_matching_service.py` — search
  hot path
- `apps/backend/src/db/qdrant_pair_repository.py` — pair payload
  schema
- `apps/backend/src/db/repositories/capture_minutia_repository.py`
  — PG minutiae CRUD

# Phase 28 CONTEXT: MinIO Migration + Minutiae-as-Data

**Status**: Planning
**Date**: 2026-06-19
**Phase**: 28

## Why this phase exists

The current architecture stores fingerprint images as bytea blobs in
PostgreSQL (`FingerprintCapture.enhanced_image` column, ~28KB per
capture). The enhanced image is the 256x256 Gabor-filtered output
generated at enrollment and served from the API on demand.

This is the wrong place for images:

1. **Storage cost**: PG is ~10x more expensive per GB than S3/MinIO.
   At scale (1M+ subjects, multiple captures each) this matters.
2. **Serving performance**: streaming from PG bytea is slower than
   from a CDN-fronted object store.
3. **Workflow**: enhanced images are derived (computed from raw).
   Mixing raw and derived data in the same table conflates layers.
4. **Matching**: minutiae are extracted on every search. Storing
   them once avoids re-processing on every query.

The user's insight: store images in MinIO (object storage), store
minutiae as first-class data in PG, render the image on demand with
minutiae overlay. This separates **storage** (MinIO) from
**structure** (PG) from **computation** (search).

## Decisions (locked)

1. **Image storage**: MinIO bucket `fingerprints`, key
   `captures/{capture_id}.png`. No bytes in PG, ever.
2. **Minutiae storage**: PG table `capture_minutiae` (new). One
   row per (capture_id, minutia_index). Append-only on
   re-extraction (forensic evidence).
3. **No legacy**: `FingerprintCapture.enhanced_image` column is
   dropped in the same migration. No dual-write period. No
   "keep the old path for safety" branches.
4. **Render on demand**: the image+minutiae overlay is composed
   server-side per request, not pre-rendered. Different styles
   (plain/overlay/labelled/forensic) supported.
5. **Match flow**: probe image → extract minutiae (search-time
   only) → match against stored minutiae in PG. No re-extraction
   of enrolled minutiae.
6. **Qdrant stays** for the pair-based KNN. The pairs collection
   is fast for similarity search, minutiae table is fast for
   forensic queries. Both coexist; they have different access
   patterns.

## Scope fences (out of scope for this phase)

- **Tamper-evidence** (Phase 29): hash chain over minutiae,
  capture-level Merkle root. This phase stores the hash; tamper
  verification is a later concern.
- **Render styles** (Phase 30): plain/overlay/labelled/forensic
  compositing. This phase uses a single simple style.
- **Frontend changes**: the frontend already uses MinIO URLs via
  the `image_uri` field. No changes needed.
- **Encryption**: Phase 26 already added optional client-side
  encryption via `ObjectStorage`. No changes here.

## Clean rebuild assumption

The user said: "haz como que empezamos de 0, reinciemos la db no
importa, luego indexamos todo, con migraciones nuevas, todo nuevo".

Implication: the migration is destructive. Existing captures are
discarded. Existing minutiae are re-extracted from new uploads.
The migration is:

1. Drop `fingerprint_captures.enhanced_image`
2. Create `capture_minutiae` (replaces legacy table if any)
3. (Re-enroll happens at runtime, not in the migration)

We accept that existing enrolled subjects need to be re-enrolled
after the migration. This is OK because the enrollment path is
fast (~1s per capture) and the test data (5 SOCOFING subjects)
is reproducible from `scripts/reenroll_pairs.py`.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MinIO not reachable at runtime | Low | High (no image serving) | Health check + clear error in endpoint |
| Minutiae schema migration wrong | Medium | High (broken enrollment) | Run migration against dev DB first; backup before deploy |
| MinIO bucket missing | Low | High | `_ensure_bucket_exists` runs at startup |
| Old captures orphaned (no MinIO image) | High (in dev) | Low (404 on image) | Endpoints return 404 + clear message; perito re-enrolls |
| Image upload fails after capture created | Medium | Medium (capture row exists, no MinIO file) | `image_uri=minio://pending/...` placeholder; perito retries via re-enroll |

## Acceptance criteria

- [ ] `fingerprint_captures.enhanced_image` column does not exist
- [ ] `capture_minutiae` table exists with all 12 columns
- [ ] New enrollment: image in MinIO, minutiae in PG, capture row
  with `image_uri=minio://captures/{id}.png`
- [ ] `GET /captures/{id}/image` returns PNG bytes from MinIO
- [ ] `GET /captures/{id}/image` returns 404 with clear message
  if MinIO file missing (no 503 fallback)
- [ ] Search endpoint works against stored minutiae (re-enroll all
  5 SOCOFING subjects via `reenroll_pairs.py` and verify benchmark)
- [ ] No code references `enhanced_image` (grep returns 0 in `src/`)

## References

- ADR-008: Cylinders vs Pairs
- ADR-009: Remove cylinder matcher
- Phase 27 plan: match algorithm convergence
- `docs/architecture/minio-migration.md` (user-facing plan)

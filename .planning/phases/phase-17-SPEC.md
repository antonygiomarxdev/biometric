# Phase 17 SPEC: Person / Fingerprint / Capture / RidgeGraph

## Objective

Replace the current `person_id: str` foreign-key-as-string pattern in
Qdrant with a **proper relational schema** that mirrors how real
forensic AFIS systems (FBI NGI, INTERPOL AFIS, NEC, ANSI/NIST-ITL
1-2011) model subjects, impressions, and per-impression processing
results.

This is the foundation for:
- Multi-impression enrollment (FBI practice: 2-3 captures per finger)
- Per-capture versioning (re-process with new algorithm → new Capture)
- Multi-region graph matching (latents split into connected components)
- Palm print handling (different DPI, ROI, ~1500 minutiae)
- NIST ITL 1-2011 Type-14 export (deferred to Phase 18)
- Multi-capture fusion (FBI reports ~15-20% FN reduction)

## Why now

Phase 15 wired Qdrant as the search hot-path but used `person_id: str`
as a flat key. There is no record in PostgreSQL of:
- Who the person is
- Which finger / palm the print came from
- How many times the person was enrolled
- Where the original image lives
- What processing algorithm/version produced this capture

A latent search result returns `person_id` as a string, with no way to
trace which finger or which capture matched. This violates forensic
chain-of-custody doctrine (D-09: immutable audit trail).

## Background — forensic terminology

| Term | Definition | Example |
|------|------------|---------|
| **Subject** / **Person** | The human being | "John Doe" |
| **Friction-ridge area** | A region of skin with ridge patterns | Right thumb, left index, palm |
| **Impression** / **Capture** | One recorded instance of a friction-ridge area | One rolled ink card, one live-scan, one lifted latent |
| **Exemplar** / **Known** | Deliberate, controlled capture of a known person | Ink card, live-scan at booking |
| **Latent** | Incidental, uncontrolled capture from a surface | Powder-lifted print from a window |
| **Ten-print card** | Standard 10-finger capture set | FBI FD-258, with rolled + plain impressions |
| **Slap** / **plain** | Four-finger flat impression | Used to verify finger order in rolled card |
| **Rolled** | Finger rolled nail-to-nail to capture entire pattern | Most informative exemplar |
| **Partial** | Impression that captures <100% of the friction-ridge area | Common for latents |
| **Patent** / **plastic** | Visible without chemical development | Fingerprint in dust, soap |
| **Palm** | Palm of hand (thenar, hypothenar, interdigital) | ~1500 minutiae vs ~150 for finger |

**NIST FGP (Finger Position Code) — ANSI/NIST-ITL 1-2011:**

| Code | Position |  | Code | Position |
|------|----------|--|------|----------|
| 0 | Unknown |  | 8 | Right middle (left) |
| 1 | Right thumb |  | 9 | Right ring (left) |
| 2 | Right index |  | 10 | Right little (left) |
| 3 | Right middle |  | 11 | Right palm (right) |
| 4 | Right ring |  | 12 | Left palm (left) |
| 5 | Right little |  | 13 | Right palm lateral (right) |
| 6 | Left thumb |  | 14 | Left palm lateral (left) |
| 7 | Left index |  | 15+ | (reserved / foot / toe) |

**NIST IDC (Image Designation Character):**
- 0 = Live-scan of exemplars (rolled, plain, or palm)
- 1 = Live-scan of exemplar — non-live (e.g., paper card)
- 2 = Latent
- 3 = Latent — non-live

**Forensic principles encoded in the model:**

1. **Multiple impressions per finger per person** (D-12: perito puede
   tomar N rolled + N plain + N latent captures del mismo right index).
   Each is a separate `FingerprintCapture` row.
2. **Person identity ≠ fingerprint match** (D-01). The system never
   auto-confirms identity. A `Decision` is required, and even then a
   `matched_fingerprint_id` and `matched_person_id` are nullable until
   the perito resolves the case.
3. **Each impression is its own observation** (D-09 chain of custody).
   Re-processing the same image with a new algorithm produces a new
   `FingerprintCapture` row, not an update.
4. **Latents are usually partial** (ACE-V methodology). The model
   supports multiple `RidgeGraph` rows per capture (one per connected
   component of the skeleton).
5. **Palms have different scale** (NIST 11-14 codes, ~1500 minutiae).
   Capture type and DPI differ; the model allows this.

## Architecture

### Entity model (4 levels)

```
Person  1──N  Fingerprint  1──N  FingerprintCapture  1──N  RidgeGraph
                              (impression)             (connected component)
                                                         ↓ mirrored to
                                                      NebulaGraph (fine match)
                                                         ↓
                                                      Qdrant (chunks)
```

#### Level 1: `Person`

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUIDv7 | PK |
| `external_id` | str(100)? | e.g. cédula — UNIQUE, INDEXED |
| `full_name` | str(300)? | |
| `doc_type` | enum? | `cedula`, `dui`, `passport`, `internal_id`, `driver_license`, `other` |
| `doc_number` | str(100)? | paired with `doc_type` |
| `sex` | str(1)? | `M`, `F`, `X` |
| `dob` | date? | |
| `notes` | text? | free-form forensic annotations |
| `created_at` | datetime | tz-aware |
| `updated_at` | datetime | tz-aware |

`Fingerprint` relationship: 1:N, cascade delete-orphan.

#### Level 2: `Fingerprint` (one friction-ridge area slot)

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUIDv7 | PK |
| `person_id` | UUID | FK → persons.id, CASCADE |
| `finger_position` | int | NIST FGP code 0-14 |
| `capture_type` | enum | `rolled`, `plain`, `slap`, `latent`, `palm`, `segment` |
| `capture_count` | int | denormalised, app-maintained |
| `first_captured_at` | datetime? | earliest capture |
| `last_captured_at` | datetime? | most recent capture |
| `notes` | text? | |
| `created_at`, `updated_at` | datetime | |

**Unique constraint:** `(person_id, finger_position, capture_type)` — one
slot per (person, finger, capture_type) combination. Multi-capture
of the same finger is represented as multiple `FingerprintCapture` rows
under the same `Fingerprint` slot.

Note: palms use FGP 11-14 with `capture_type = "palm"`.

#### Level 3: `FingerprintCapture` (one processed image)

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUIDv7 | PK |
| `fingerprint_id` | UUID | FK → fingerprints.id, CASCADE |
| `capture_index` | int | 1-based, sequential per Fingerprint |
| `image_uri` | str(500) | MinIO/S3 URI |
| `image_hash_sha256` | str(64) | tamper-detection |
| `image_dpi` | int? | 500, 1000, etc. |
| `image_quality_score` | float? | 0.0-1.0 (NFIQ-2 future) |
| `algorithm_version` | str(50) | e.g. `phase-13-v1` |
| `processed_at` | datetime | |
| `num_minutiae` | int? | denormalised |
| `num_graphs` | int? | denormalised; usually 1, sometimes 2-4 for latents |
| `is_reference` | bool | designated reference capture for this Fingerprint? |
| `is_exemplar` | bool | true for known/impression, false for latent |
| `notes` | text? | |
| `created_at` | datetime | |

#### Level 4: `RidgeGraph` (one connected component)

| Field | Type | Notes |
|-------|------|-------|
| `id` | UUIDv7 | PK |
| `capture_id` | UUID | FK → fingerprint_captures.id, CASCADE |
| `graph_index` | int | 1-based per capture |
| `region_x`, `region_y` | int | bbox top-left in image coords |
| `region_w`, `region_h` | int | bbox size |
| `num_nodes`, `num_edges` | int | denormalised |
| `graph_data` | JSONB | serialised `{nodes: [...], edges: [...]}` |
| `core_x`, `core_y` | int? | null for partials / palms |
| `delta_x`, `delta_y` | int? | null if not detected |
| `singularity_type` | str? | `core`, `delta`, `double_loop`, `none` |
| `created_at` | datetime | |

### `Evidence` link to Person/Fingerprint

The `Evidence` table (latent print from crime scene) gets two new nullable
columns so a Decision can resolve the match to a specific person and finger:

```python
matched_fingerprint_id: Mapped[uuid.UUID | None] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("fingerprints.id", ondelete="SET NULL"),
    nullable=True, index=True,
)
matched_person_id: Mapped[uuid.UUID | None] = mapped_column(
    UUID(as_uuid=True),
    ForeignKey("persons.id", ondelete="SET NULL"),
    nullable=True, index=True,
)
```

This implements the forensic principle: the latent exists unresolved, and
only when a perito emits a Decision is the match recorded. Both fields are
nullable so the Evidence can be created and searched before resolution.

### Qdrant payload (extended)

```json
{
  "person_id": "0190e8a0-...",
  "fingerprint_id": "0190e8a1-...",
  "capture_id": "0190e8a2-...",
  "graph_id": "0190e8a3-...",
  "finger_position": 7,
  "capture_type": "rolled",
  "chunk_type": "delaunay",
  "weight": 0.85,
  "chunk_index": 42
}
```

New fields: `capture_id`, `graph_id`, `finger_position`, `capture_type`.
Old fields (`person_id`, `fingerprint_id`) are kept for back-compat and
aggregation.

### NebulaGraph schema extension

Add two new tag properties to the `minutia` tag:
- `capture_id string`
- `graph_id string`

Existing minutiae get `NULL` for these. Migration:
```ngql
ALTER TAG minutia ADD (capture_id string, graph_id string);
```

## API design

### POST /api/v1/persons

Create a person. Response 201 with the new person.

### GET /api/v1/persons/{person_id}

Retrieve a person. Response 200.

### GET /api/v1/persons

List persons with pagination. Query params: `skip`, `limit`, `search`
(matches external_id / full_name).

### POST /api/v1/persons/{person_id}/fingerprints

Create a fingerprint slot. Body: `{finger_position, capture_type}`.
Response 201. 409 if slot already exists.

### GET /api/v1/persons/{person_id}/fingerprints

List fingerprint slots for a person. Response 200 with all `Fingerprint`
rows (denormalised counts).

### POST /api/v1/fingerprints/{fingerprint_id}/captures

Upload an image. Multipart:
- `file`: BMP/PNG/JPEG
- `image_dpi`: int (optional)
- `is_exemplar`: bool (default true)
- `is_reference`: bool (default false)
- `notes`: str (optional)

Returns 201 with `FingerprintCapture` and the count of extracted graphs.

Internally:
1. Store image in MinIO under `captures/{capture_id}.{ext}`
2. Compute SHA-256
3. Run `FingerprintService.process_image`
4. Extract connected components from skeleton → `list[RidgeGraph]`
5. Persist graphs in Postgres (JSONB) and NebulaGraph
6. Vectorise and store chunks in Qdrant with extended payload
7. Update `capture_count` and `last_captured_at` on parent `Fingerprint`
8. Return the capture + graphs list

### GET /api/v1/fingerprints/{fingerprint_id}/captures

List all captures for a fingerprint slot. Returns array of
`FingerprintCapture` with denormalised counts.

### GET /api/v1/captures/{capture_id}

Get a single capture with its graphs.

### GET /api/v1/captures/{capture_id}/graphs

List `RidgeGraph` rows for a capture.

### PATCH /api/v1/captures/{capture_id}

Update `is_reference`, `notes`. Only these fields are mutable.

### POST /api/v1/matching/search (UNCHANGED contract)

The coarse Qdrant search returns the same `SearchHit` shape. Internally,
the Qdrant point payload now carries the extended schema. The service can
expose additional query params:

- `top_k_per_finger` (bool, default false) — aggregate by (person, finger_position)
- `top_k_per_capture` (bool, default false) — exact image match
- `finger_position` (int, optional) — scope to a single finger

### POST /api/v1/cases/{case_id}/evidence/{evidence_id}/decision (updated)

When a perito emits a `Decision`, the optional body can include
`matched_fingerprint_id` and `matched_person_id`. These are nullable.
If provided, the Evidence's `matched_*` columns are set.

### GET /api/v1/persons/{person_id}/fingerprints/{fp_id}/fuse (NEW, optional)

Multi-capture fusion: returns the averaged MCC descriptor across all
`is_reference=True` captures of a fingerprint. Useful for the FBI-style
"average 2-3 captures for FN reduction" workflow. Defers the actual
fusion algorithm to a follow-up phase.

## Non-Goals (deferred to later phases)

- **NIST ITL 1-2011 Type-14 export** — separate phase
- **FD-258 ten-print card** — paper-card scanning workflow
- **Automatic subject dedup** — detecting that two Person rows are the
  same human (would use `person_aliases` table)
- **Biometric template expiry** — GDPR / Ley 787 compliance (templates
  in Qdrant are pseudonymous identifiers, not the raw biometric)
- **Henry pattern classification** (arch / loop / whorl) — useful for
  triage, not required for matching
- **Multi-capture fusion algorithm** — exposed as endpoint, algorithm
  is a follow-up
- **NFIQ-2 quality score** — only the column is added, value is `null`
  for now (no NFIQ-2 dependency)

## Affected files

| File | Change |
|------|--------|
| `apps/backend/src/db/models.py` | + `Person`, `Fingerprint`, `FingerprintCapture`, `RidgeGraph`; `Evidence` +matched_* |
| `apps/backend/src/db/enums.py` (new) | `FingerPosition` (NIST FGP), `CaptureType`, `DocumentType` |
| `apps/backend/src/db/migrations/versions/0005_*.py` | new Alembic migration |
| `apps/backend/src/db/repositories/person_repository.py` | new |
| `apps/backend/src/db/repositories/fingerprint_repository.py` | new |
| `apps/backend/src/db/repositories/fingerprint_capture_repository.py` | new |
| `apps/backend/src/db/repositories/ridge_graph_repository.py` | new |
| `apps/backend/src/services/person_service.py` | new |
| `apps/backend/src/services/fingerprint_enrollment_service.py` | new (orchestrator) |
| `apps/backend/src/services/rag_matching_service.py` | payload +2 fields |
| `apps/backend/src/db/nebula_repository.py` | new tag properties |
| `apps/backend/src/db/qdrant_chunk_repository.py` | payload +2 fields |
| `apps/backend/src/processing/graph_extractor.py` | return `list[RidgeGraph]` |
| `apps/backend/src/api/routers/persons.py` | new router |
| `apps/backend/src/api/routers/fingerprints.py` | new router |
| `apps/backend/src/api/routers/known_fingerprints.py` | deprecate |
| `apps/backend/src/api/routers/latent_search.py` | add new query params |
| `apps/backend/src/api/routers/decisions.py` | accept matched_* |
| `apps/backend/src/schemas/person_schema.py` | new DTOs |
| `apps/backend/src/schemas/fingerprint_schema.py` | new DTOs |
| `apps/backend/src/schemas/capture_schema.py` | new DTOs |
| `apps/backend/tests/db/repositories/test_person_repository.py` | new |
| `apps/backend/tests/db/repositories/test_fingerprint_repository.py` | new |
| `apps/backend/tests/db/repositories/test_fingerprint_capture_repository.py` | new |
| `apps/backend/tests/db/repositories/test_ridge_graph_repository.py` | new |
| `apps/backend/tests/api/test_persons_router.py` | new |
| `apps/backend/tests/api/test_fingerprints_router.py` | new |
| `apps/backend/tests/integration/test_capture_e2e.py` | new E2E |

## Migration plan (additive — no breaking changes)

1. **Step 1:** Add new tables + enums + Alembic 0005 migration. Old
   `Evidence.fingerprint_id: str` keeps working; new fields are nullable.
2. **Step 2:** Add repositories. No service or router changes.
3. **Step 3:** Add services. Routers still go through the legacy flow.
4. **Step 4:** Add routers (`/persons`, `/fingerprints`). They're
   additive — they don't replace `/known-fingerprints/`.
5. **Step 5:** Mark `/known-fingerprints/` as deprecated (emits
   `Deprecation` header, returns same shape but logs warning).
6. **Step 6:** Update Qdrant + NebulaGraph schema (backward-compatible
   — new fields are optional, old points keep working).
7. **Step 7:** Refactor `RidgeGraphExtractor` to return
   `list[RidgeGraph]` (one per connected component). The
   `FingerprintService.process_image` orchestrator now stores N graphs
   instead of 1.
8. **Step 8:** Add tests (30+ unit, 15+ API, 5+ E2E).
9. **Step 9:** Update the `Decisions` router to accept `matched_*`.

## Verification

After all migrations applied:

```bash
# Create person
PERSON=$(curl -X POST localhost:8000/api/v1/persons \
  -H "Content-Type: application/json" \
  -d '{
    "external_id": "001-010101-0101A",
    "doc_type": "cedula",
    "doc_number": "001-010101-0101A",
    "full_name": "Juan Pérez"
  }' | jq -r .id)

# Create fingerprint slot
FP=$(curl -X POST localhost:8000/api/v1/persons/$PERSON/fingerprints \
  -H "Content-Type: application/json" \
  -d '{"finger_position": 2, "capture_type": "rolled"}' | jq -r .id)

# Upload capture
CAP=$(curl -X POST localhost:8000/api/v1/fingerprints/$FP/captures \
  -F "file=@test_data/1__M_Left_index_finger.BMP" \
  -F "image_dpi=500" -F "is_exemplar=true" -F "is_reference=true" | jq -r .id)

# Verify graphs were extracted
curl localhost:8000/api/v1/captures/$CAP/graphs

# Search with latent
curl -X POST localhost:8000/api/v1/matching/search \
  -F "file=@test_data/2__M_Left_index_finger.BMP"

# Perito emits decision
curl -X POST localhost:8000/api/v1/cases/$CASE/evidence/$EVIDENCE/decisions \
  -H "Content-Type: application/json" \
  -d "{\"verdict\": \"Identificación\", \"matched_fingerprint_id\": \"$FP\", \"matched_person_id\": \"$PERSON\"}"
```

## Acceptance criteria (for the implementing plan-phase)

- [ ] 4 new tables created via Alembic 0005
- [ ] 4 new repositories with ≥ 6 unit tests each (24 total)
- [ ] `FingerPosition` enum covers NIST FGP 0-14 + UNKNOWN
- [ ] `CaptureType` enum: rolled, plain, slap, latent, palm, segment
- [ ] `RidgeGraphExtractor` returns `list[RidgeGraph]` (one per connected component)
- [ ] Multi-capture of the same finger: 2 captures, 2 separate graphs
- [ ] Qdrant payload has 7 fields including `capture_id` and `graph_id`
- [ ] NebulaGraph minutia tag has 2 new properties
- [ ] 3 new routers (`/persons`, `/fingerprints`, `/captures`)
- [ ] `Evidence` has `matched_fingerprint_id` and `matched_person_id`
- [ ] `Decisions` router accepts the matched fields
- [ ] `/known-fingerprints/` returns 200 with `Deprecation: true` header
- [ ] E2E test: create person → upload 2 captures of same finger → search with latent from one → expect top match
- [ ] All existing 609 tests still pass

## Out of scope confirmation

The following are **NOT** in this phase (deferred):
- NIST ITL export, FD-258, dedup, GDPR expiry, Henry classification,
  multi-capture fusion algorithm, NFIQ-2

These have their own dedicated phases.

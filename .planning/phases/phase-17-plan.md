# Phase 17: Person / Fingerprint / Capture — Forensic Data Model

## Goal

Replace the current `person_id: str` foreign-key-as-string model in Qdrant
with a **proper relational schema** that mirrors how real forensic AFIS
systems model subjects, impressions, and per-impression processing results.

Specifically:

- A `Person` row in PostgreSQL, owning 1..N `Fingerprint` rows
- A `Fingerprint` row representing **one captured impression** (rolled, plain,
  latent, partial, palm) of one friction-ridge area
- A `FingerprintCapture` row representing **one processing pass** of a
  fingerprint (one image → one extracted graph, one cylinder set, one Qdrant
  chunk set)
- A `RidgeGraph` row storing the connected-component topology extracted from
  a single capture (a 96×103 SOCOFing image typically yields 1 graph; a
  1500×2000 crime-scene latent can yield 2–4 disconnected components)
- Migration of the Qdrant payload to carry `fingerprint_id` + `capture_id`
  so chunk searches can scope to a person, a single capture, or the whole
  population

This is the foundation for: (a) multi-impression enrollment, (b) per-capture
versioning (re-process the same image with a new algorithm), (c) multi-region
graph matching, (d) palm print handling, (e) NIST/ANSI-NIST ITL data export.

## Why now

Phase 15 wired Qdrant as the search hot-path but used `person_id: str` as
a flat key in the Qdrant payload. There is no record in PostgreSQL of:

- Who the person is
- Which finger / palm the print came from
- How many times the person was enrolled (multiple impressions exist
  per finger in real life — rolled, plain, slap, partial)
- Where the original image lives (MinIO bucket key)
- What processing algorithm/version produced this capture

The forensic workflow is: **perito enrols suspect → suspect comes back next
week → perito takes a new impression of the same finger → there are now two
captures to compare against**. The current model collapses all of that
into a single `person_id` string.

## Non-Goals

- NIST/ANSI-NIST ITL 1-2007 export format — separate phase
- 10-print card scanning (FBI-standard FD-258) — separate phase
- Probabilistic calibration (LR / FAR) — Phase 18
- Multi-tenancy (agency separation) — already handled by `cases` table
- Replacing the case management UI

## Background — forensic terminology

| Term | Definition | Example |
|------|------------|---------|
| **Subject** / **Person** | The human being | "John Doe" |
| **Friction-ridge area** | A region of skin with ridge patterns | Right thumb, left index, palm, etc. |
| **Impression** / **Capture** | One recorded instance of a friction-ridge area | One rolled ink card, one live-scan, one lifted latent |
| **Exemplar** / **Known** | Deliberate, controlled capture of a known person | Ink card, live-scan at booking |
| **Latent** | Incidental, uncontrolled capture from a surface | Powder-lifted print from a window |
| **Ten-print card** | Standard 10-finger capture set | FBI FD-258, with rolled + plain impressions |
| **Slap / plain** | Four-finger flat impression | Used to verify finger order in rolled card |
| **Rolled** | Finger rolled nail-to-nail to capture entire pattern | Most informative exemplar |
| **Partial** | Impression that captures <100% of the friction-ridge area | Common for latents |
| **Patent / plastic** | Visible without chemical development | Fingerprint in dust, soap |
| **Palm** | Palm of hand (thenar, hypothenar, interdigital) | ~1500 minutiae vs ~150 for finger |

Forensic reference: FBI NGIP, ANSI/NIST-ITL 1-2007, NIST Special Publication
500-290 (EFTS), IAI study guide on latent prints, and the Wikipedia entries
on Fingerprint, Palm_print, and AFIS. The `Composite` AFIS systems
(Identix/MorphoTrust, NEC, Cogent) all implement this data model with
minor variations.

**Key forensic principles we must encode:**

1. **Multiple impressions per finger per person** — a person can have N
   rolled + N plain + N latent captures of the same right index. Each is
   a separate `Fingerprint` row.
2. **Person identity ≠ fingerprint match** — a person can have false fingers
   (e.g. undetectable scars) and a fingerprint match never auto-confirms
   identity (D-01 forensic doctrine).
3. **Each impression is its own observation** — re-processing the same
   image with a new algorithm produces a new `Capture` row, not an update.
4. **Latents are usually partial** — forensically, partial latents with
   as few as 8 minutiae can be diagnostic (ACE-V methodology).
5. **Palms have more minutiae than fingers** (~1500 vs ~150) and a
   different processing pipeline (different DPI, different ROI).

## Architecture

### Entity model

```
┌──────────────┐
│   Person     │   PostgreSQL
│              │
│ id (UUIDv7)  │
│ full_name    │
│ doc_type     │   "cedula", "DUI", "passport", "internal_id"
│ doc_number   │
│ sex          │   "M" / "F" / "X"
│ dob          │
│ notes        │   free-form forensic annotations
│ created_at   │
│ updated_at   │
└──────┬───────┘
       │ 1:N
       ▼
┌──────────────────┐
│   Fingerprint    │   PostgreSQL
│                  │   One friction-ridge area + its capture history
│ id (UUIDv7)      │
│ person_id (FK)   │
│ capture_type     │   "rolled" | "plain" | "slap" | "latent" | "palm"
│ capture_subtype  │   "tenprint_right" | "tenprint_left" | "single_finger" | ...
│ area             │   finger or palm region — see enum below
│ capture_count    │   denormalised; updated by trigger / app code
│ first_captured_at│
│ last_captured_at │
│ notes            │
│ created_at       │
│ updated_at       │
└──────┬───────────┘
       │ 1:N
       ▼
┌────────────────────────┐
│  FingerprintCapture    │   PostgreSQL
│                        │   One processed image → one set of artefacts
│ id (UUIDv7)            │
│ fingerprint_id (FK)    │
│ capture_index          │   1, 2, 3... per Fingerprint (1-based)
│ image_uri              │   s3://... or minio://.../key
│ image_hash_sha256      │   tamper-detection
│ image_dpi              │   500, 1000, etc. (nullable for unknown)
│ image_quality_score    │   0..1 (NFIQ-2 in future)
│ algorithm_version      │   "phase-13-v1", etc.
│ processed_at           │
│ num_minutiae           │   int, denormalised for fast queries
│ num_graphs             │   int, denormalised — most captures have 1
│ is_reference           │   bool — designated reference capture?
│ is_exemplar            │   bool — known/impression vs latent
│ notes                  │
│ created_at             │
└──────┬─────────────────┘
       │ 1:N
       ▼
┌────────────────────────┐
│     RidgeGraph         │   PostgreSQL (JSONB topology + NebulaGraph)
│                        │   One connected component of the ridge skeleton
│ id (UUIDv7)            │
│ capture_id (FK)        │
│ graph_index            │   1, 2, 3... per capture
│ region_x, region_y     │   bbox top-left
│ region_w, region_h     │
│ num_nodes              │
│ num_edges              │
│ graph_data (JSONB)     │   serialised {nodes: [...], edges: [...]}
│ core_x, core_y         │   optional — may be null for partials
│ delta_x, delta_y       │   optional
│ singularity_type       │   "core" | "delta" | "double_loop" | null
│ created_at             │
└──────┬─────────────────┘
       │
       │ Mirrored to NebulaGraph for fine-grained subgraph matching:
       ▼
┌────────────────────────┐
│   NebulaGraph          │
│   minutia(tag)         │
│     fingerprint_id     │   → Fingerprint.id
│     capture_id         │   → FingerprintCapture.id   ← NEW
│     graph_id           │   → RidgeGraph.id            ← NEW
│     node_idx           │
│     x, y, angle, ...   │
│   ridge_edge(edge)     │
│     source, target     │   → minutia vertex IDs
│     length             │
└────────────────────────┘

       │ Chunk store (Qdrant)
       ▼
┌────────────────────────┐
│   Qdrant payload       │
│   (per chunk)          │
│     person_id          │   → Person.id          (for person-level aggregation)
│     fingerprint_id     │   → Fingerprint.id     (for finger-level)
│     capture_id         │   → FingerprintCapture.id   ← NEW
│     graph_id           │   → RidgeGraph.id           ← NEW
│     chunk_type         │   "delaunay" | "mcc"
│     weight             │
│     chunk_index        │
└────────────────────────┘
```

### Enums

```python
class CaptureType(str, Enum):
    ROLLED  = "rolled"        # Rolled ink or rolled live-scan
    PLAIN   = "plain"         # Single finger flat
    SLAP    = "slap"          # Four-finger flat
    LATENT  = "latent"        # Crime-scene lifted
    PALM    = "palm"          # Palm impression
    SEGMENT = "segment"       # Single finger region from a ten-print card

class FrictionRidgeArea(str, Enum):
    # Right hand
    RIGHT_THUMB     = "right_thumb"
    RIGHT_INDEX     = "right_index"
    RIGHT_MIDDLE    = "right_middle"
    RIGHT_RING      = "right_ring"
    RIGHT_LITTLE    = "right_little"
    # Left hand
    LEFT_THUMB      = "left_thumb"
    LEFT_INDEX      = "left_index"
    LEFT_MIDDLE     = "left_middle"
    LEFT_RING       = "left_ring"
    LEFT_LITTLE     = "left_little"
    # Palms
    RIGHT_PALM      = "right_palm"
    LEFT_PALM       = "left_palm"
    # Foot / toe (NIST allows, see ANSI/NIST-ITL 1-2007 Type-14)
    RIGHT_FOOT      = "right_foot"
    LEFT_FOOT       = "left_foot"
    UNKNOWN         = "unknown"     # partial latents where we can't tell

class DocumentType(str, Enum):
    CEDULA          = "cedula"           # Nicaraguan ID
    DUI             = "dui"              # Salvadoran/Honduran ID
    PASSPORT        = "passport"
    INTERNAL_ID     = "internal_id"      # case-internal
    DRIVER_LICENSE  = "driver_license"
    OTHER           = "other"
```

### Foreign-key cascade policy

| From | To | on delete | Rationale |
|------|----|-----------|-----------|
| FingerprintCapture | Fingerprint | CASCADE | captures without a parent finger are meaningless |
| RidgeGraph | FingerprintCapture | CASCADE | graphs without a capture are meaningless |
| Evidence | FingerprintCapture | SET NULL | an evidence item references a capture but can survive its deletion (forensic chain of custody) |
| Decision | Evidence | SET NULL | a decision is a verdict; the evidence may be re-assigned |
| Fingerprint | Person | RESTRICT | prevent accidentally deleting a person who has enrolled fingers |
| AuditLog | (no FK) | — | tamper-evident, immutable |

## API design

### POST /api/v1/persons

Create a person.

Request:
```json
{
  "full_name": "Juan Pérez",
  "doc_type": "cedula",
  "doc_number": "001-010101-0101A",
  "sex": "M",
  "dob": "1985-03-15"
}
```

Response 201:
```json
{
  "id": "0190e8a0-...",
  "full_name": "Juan Pérez",
  "doc_type": "cedula",
  "doc_number": "001-010101-0101A",
  "sex": "M",
  "dob": "1985-03-15",
  "created_at": "2026-01-15T10:00:00Z"
}
```

### POST /api/v1/persons/{person_id}/fingerprints

Create a fingerprint slot for a person.

Request:
```json
{
  "capture_type": "rolled",
  "area": "right_index"
}
```

Response 201:
```json
{
  "id": "0190e8a1-...",
  "person_id": "0190e8a0-...",
  "capture_type": "rolled",
  "area": "right_index",
  "capture_count": 0,
  "first_captured_at": null,
  "last_captured_at": null
}
```

### POST /api/v1/fingerprints/{fingerprint_id}/captures

Upload an image and trigger processing.

Multipart:
- `file`: fingerprint image (BMP/PNG/JPEG)
- `image_dpi`: int (optional)
- `is_exemplar`: bool (default true)
- `notes`: str (optional)

Response 201:
```json
{
  "id": "0190e8a2-...",
  "fingerprint_id": "0190e8a1-...",
  "capture_index": 1,
  "image_uri": "minio://fingerprints/0190e8a2.png",
  "image_dpi": 500,
  "algorithm_version": "phase-13-v1",
  "processed_at": "2026-01-15T10:05:00Z",
  "num_minutiae": 42,
  "num_graphs": 1,
  "is_reference": true,
  "is_exemplar": true
}
```

Internally, this:
1. Stores the image in MinIO under `captures/{capture_id}.{ext}`
2. Computes SHA-256 of the image bytes
3. Runs the existing pipeline (`FingerprintService.process_image`)
4. Extracts one or more `RidgeGraph` components
5. Persists graphs in PostgreSQL (JSONB) and NebulaGraph
6. Vectorises into chunks and stores in Qdrant with the new payload schema
7. Updates denormalised counters on `Fingerprint`

### GET /api/v1/captures/{capture_id}/graphs

List the ridge graphs extracted from a capture.

Response 200:
```json
{
  "capture_id": "0190e8a2-...",
  "graphs": [
    {
      "id": "0190e8a3-...",
      "graph_index": 1,
      "region": {"x": 12, "y": 5, "w": 280, "h": 320},
      "num_nodes": 28,
      "num_edges": 35,
      "core": {"x": 142, "y": 167},
      "singularity_type": "core"
    },
    {
      "id": "0190e8a4-...",
      "graph_index": 2,
      "region": {"x": 12, "y": 330, "w": 280, "h": 60},
      "num_nodes": 5,
      "num_edges": 4,
      "core": null,
      "singularity_type": null
    }
  ]
}
```

### PATCH /api/v1/captures/{capture_id}

Mark a capture as the reference, or update notes.

### POST /api/v1/matching/search (unchanged contract)

The Qdrant search now operates on chunks that carry the new payload
fields. The `top_k_per_chunk` and `top_k_persons` semantics are preserved.

Internally the search returns the same `SearchHit` but the service can
now also offer:

- `top_k_per_finger` — aggregate by (person_id, area) instead of person
- `top_k_per_capture` — exact image match

These are exposed as additional query parameters on the same endpoint.

## Migration plan

### Step 1: Models + migration

Add 4 new tables (`persons`, `fingerprints`, `fingerprint_captures`,
`ridge_graphs`) via Alembic revision `0005_add_person_fingerprint_capture`.
The `RagVectorChunk` table is already gone (Phase B). `Evidence` gets a
nullable `fingerprint_capture_id` column.

### Step 2: Repositories

Add 4 new repositories following the existing pattern:
- `PersonRepository`
- `FingerprintRepository`
- `FingerprintCaptureRepository`
- `RidgeGraphRepository`

These follow Clean Architecture: the router depends on a service,
the service depends on a repository, the repository depends on
SQLAlchemy. No leakage of ORM into the API.

### Step 3: Services

Add service classes that compose the repositories:

- `PersonService` — CRUD + forensic audit
- `FingerprintService` (rename existing → keep for image processing)
- `FingerprintEnrollmentService` — orchestrates image → capture → graphs
  → Qdrant → NebulaGraph; replaces the current QdrantRagMatchingService
  `enroll` flow but delegates chunk storage to it.

### Step 4: Routers

Three new routers:
- `/api/v1/persons` (already routed as `persons_router`)
- `/api/v1/persons/{id}/fingerprints`
- `/api/v1/fingerprints/{id}/captures`

Each is anemic — extracts payload, calls service, returns DTO. No
SQLAlchemy `Session` dependency in services (we use the async session
factory).

### Step 5: Qdrant payload migration

The `QdrantChunkRepository` adds three new fields to the point payload:
`capture_id`, `graph_id`, `area`. Existing points are backfilled via a
one-shot script that joins on the previous `fingerprint_id` (assuming we
keep the old `fingerprint_id` field for one migration cycle).

### Step 6: NebulaGraph migration

Add `capture_id` and `graph_id` as new tag properties on `minutia`. This
requires a schema migration in NebulaGraph (`ALTER TAG` statements).
Existing minutiae keep working with `null` for these new properties.

### Step 7: Update existing code

- `QdrantRagMatchingService` — accept `person_id, fingerprint_id, capture_id,
  graph_id` and write all four into the payload.
- `NebulaRepository.insert_graph` — accept `capture_id, graph_id` and write
  them on each `minutia` vertex.
- `RidgeGraphExtractor` — return `list[RidgeGraph]` instead of one
  (detect connected components). Each `RidgeGraph` is a connected
  component of the full skeleton graph.

### Step 8: Tests

- 30+ new unit tests in `tests/db/repositories/` for the new repositories
- 15+ new tests in `tests/api/` for the new routers
- 5+ new E2E tests covering the full flow: create person → create
  fingerprint → upload capture → verify graphs in Postgres → verify
  chunks in Qdrant → search by latent → expected match
- All existing tests keep passing (this is additive)

### Step 9: Update `/api/v1/known-fingerprints/`

The current `/known-fingerprints/` endpoint takes `person_id: str` as
a Form field and assumes the person exists. We mark it deprecated and
add a new `/persons/{id}/fingerprints/{fp_id}/captures` endpoint that does
the same thing with proper relational integrity. The old endpoint keeps
working for one release, emitting a `Deprecation` header.

## Affected files

| File | Change |
|------|--------|
| `apps/backend/src/db/models.py` | + `Person`, `Fingerprint`, `FingerprintCapture`, `RidgeGraph`; `Evidence` +capture_id |
| `apps/backend/src/db/migrations/versions/0005_*.py` | new Alembic migration |
| `apps/backend/src/db/enums.py` (new) | `CaptureType`, `FrictionRidgeArea`, `DocumentType` |
| `apps/backend/src/db/repositories/person_repository.py` | new |
| `apps/backend/src/db/repositories/fingerprint_repository.py` | new |
| `apps/backend/src/db/repositories/fingerprint_capture_repository.py` | new |
| `apps/backend/src/db/repositories/ridge_graph_repository.py` | new |
| `apps/backend/src/services/person_service.py` | new |
| `apps/backend/src/services/fingerprint_enrollment_service.py` | new (orchestrator) |
| `apps/backend/src/services/rag_matching_service.py` | payload gets 2 new fields |
| `apps/backend/src/db/nebula_repository.py` | new tag properties; insert_graph signature |
| `apps/backend/src/db/qdrant_chunk_repository.py` | payload gets 2 new fields |
| `apps/backend/src/processing/graph_extractor.py` | return `list[RidgeGraph]` (connected components) |
| `apps/backend/src/api/routers/persons.py` | new router |
| `apps/backend/src/api/routers/fingerprints.py` | new router |
| `apps/backend/src/api/routers/known_fingerprints.py` | deprecate, keep working |
| `apps/backend/src/api/routers/latent_search.py` | no change to contract |
| `apps/backend/src/schemas/person_schema.py` | new Pydantic DTOs |
| `apps/backend/src/schemas/fingerprint_schema.py` | new Pydantic DTOs |
| `apps/backend/src/schemas/capture_schema.py` | new Pydantic DTOs |
| `apps/backend/tests/db/repositories/test_person_repository.py` | new |
| `apps/backend/tests/db/repositories/test_fingerprint_repository.py` | new |
| `apps/backend/tests/db/repositories/test_fingerprint_capture_repository.py` | new |
| `apps/backend/tests/db/repositories/test_ridge_graph_repository.py` | new |
| `apps/backend/tests/api/test_persons_router.py` | new |
| `apps/backend/tests/api/test_fingerprints_router.py` | new |
| `apps/backend/tests/integration/test_capture_e2e.py` | new E2E |

## Verification

After all migrations applied, the following must work:

```bash
# 1. Create a person
PERSON=$(curl -X POST localhost:8000/api/v1/persons -d '{
  "full_name": "Juan Pérez",
  "doc_type": "cedula",
  "doc_number": "001-010101-0101A"
}' | jq -r .id)

# 2. Create a fingerprint slot
FP=$(curl -X POST localhost:8000/api/v1/persons/$PERSON/fingerprints -d '{
  "capture_type": "rolled",
  "area": "right_index"
}' | jq -r .id)

# 3. Upload capture
curl -X POST localhost:8000/api/v1/fingerprints/$FP/captures \
  -F "file=@test_data/1__M_Left_index_finger.BMP" \
  -F "image_dpi=500" -F "is_exemplar=true"

# 4. Search with a latent (rolled from a different capture of the same finger)
curl -X POST localhost:8000/api/v1/matching/search \
  -F "file=@test_data/2__M_Left_index_finger.BMP"

# Expect: top match is the same person
```

## Out of scope (deferred)

- **NIST ITL 1-2007 export** — Type-14 records. Separate phase.
- **FD-258 ten-print card** — paper-card scanning workflow.
- **Subject deduplication** — automatic detection that two `Person` rows
  are the same human (would use a `person_id_aliases` table).
- **Biometric template expiry** — GDPR / Nicaraguan Ley 787 compliance.
  Templates stored in Qdrant are pseudonymous identifiers, not the raw
  biometric, so they're already outside the strictest personal-data scope.
- **Finger-pattern-type classification** (Henry classification: arch,
  loop, whorl) — useful for triage, but not needed for the matching
  pipeline. Add as `Fingerprint.pattern_type` later.

## References

- ANSI/NIST-ITL 1-2007 (NIST Special Publication 500-290) — Type-14
  fingerprint record format
- FBI NGIP / NGI standards — next generation identification
- IAI Latent Print Examiner study guide
- ENFSI guideline for evaluative reporting (LR framework)
- AFIS source code (open): SourceAFIS, NIST NBIS
- Wikipedia: Fingerprint, Palm_print, AFIS

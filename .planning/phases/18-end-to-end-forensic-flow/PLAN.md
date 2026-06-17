# Phase 18: End-to-End Forensic Flow

## Goal

Make the complete forensic pipeline functional: enroll a person → register
fingerprints → upload captures → search latents against enrolled prints.

The enrollment side (Phase 17) works. The search side is broken: it uses a
different Qdrant API than enrollment, so searches never find enrolled prints.

## Tasks

### T1: Fix the search router to use the same Qdrant collection as enrollment
**File:** `src/api/routers/latent_search.py`

Replace `PolyglotMatchingService` (uses `QdrantRepository`, wrong collection)
with `QdrantRagMatchingService.search_async()` (uses `QdrantChunkRepository`,
same collection enrollment writes to).

Make the endpoint properly async — `search_async()` is already async and
runs CPU-bound processing in `run_in_executor`.

### T2: Add a QdrantRagMatchingService dependency
**File:** `src/api/dependencies.py`

Add `get_rag_matching_service()` dependency function.

### T3: Wire endpoints into main FastAPI app
**File:** `src/main.py`

Ensure the latent_search and capture routers are mounted.

### T4: Integration/E2E verification
**Files:** `tests/api/test_latent_search.py`, scripts

Verify:
1. POST /persons → create person
2. POST /persons/{id}/fingerprints → create slot
3. POST /fingerprints/{id}/captures → upload image, process, index
4. POST /matching/search → search latent, returns ranked person matches

### T5: Enrich search response with person details
**File:** `src/api/routers/latent_search.py`

When a match is found, lookup the Person record by `person_id` and include
`full_name`, `external_id` in the response so the examiner can identify the match.

## Acceptance Criteria

- [ ] POST /matching/search returns ranked candidates for an enrolled print
- [ ] Search response includes person details (name, external_id)
- [ ] Full E2E flow works: create person → fingerprint → upload → search → match
- [ ] All tests pass
- [ ] PO admits the feature

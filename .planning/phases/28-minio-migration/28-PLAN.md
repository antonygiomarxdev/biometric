# Phase 28 PLAN: MinIO Migration + Minutiae-as-Data

**Status**: Plan ready
**Depends on**: Phase 27 (match algorithm convergence)
**Mode**: clean rebuild (destructive migration)

## Plan summary

5 tasks, 1 atomic migration, 1 verification benchmark. After this
phase: images in MinIO only, minutiae in PG only, no legacy columns.

## Tasks

### Task 1: Migration — drop `enhanced_image`, create `capture_minutiae`

**Single Alembic migration.** Reverses all old data (re-enroll after
deploy).

```python
# alembic upgrade head
op.create_table(
    "capture_minutiae",
    sa.Column("id", UUID(as_uuid=True), primary_key=True),
    sa.Column("capture_id", UUID, ForeignKey("fingerprint_captures.id", ondelete="CASCADE")),
    sa.Column("person_id", UUID, ForeignKey("persons.id", ondelete="CASCADE")),
    sa.Column("minutia_index", Integer, nullable=False),
    sa.Column("x", Float, nullable=False),
    sa.Column("y", Float, nullable=False),
    sa.Column("angle", Float, nullable=False),
    sa.Column("type", Integer, nullable=False),
    sa.Column("quality", Float, nullable=False),
    sa.Column("hash", String(64), nullable=False),
    sa.Column("algo_version", String(50), nullable=False, default="pairs-v1"),
    sa.Column("extracted_at", DateTime(timezone=True), server_default=now()),
    sa.UniqueConstraint("capture_id", "minutia_index", name="uq_capture_minutia"),
)
op.create_index("ix_capture_minutiae_capture_id", "capture_minutiae", ["capture_id"])
op.create_index("ix_capture_minutiae_person_id", "capture_minutiae", ["person_id"])
op.drop_column("fingerprint_captures", "enhanced_image")
```

**Verify**: `psql -c "\d capture_minutiae"` shows all columns. No
`enhanced_image` column on `fingerprint_captures`.

### Task 2: `FingerprintStorage` service (MinIO wrapper)

New module: `src/services/fingerprint_storage.py`

```python
class FingerprintStorage:
    """MinIO wrapper for capture PNGs."""

    @staticmethod
    def object_key(capture_id: str) -> str: ...

    @staticmethod
    def upload(capture_id: str, png_bytes: bytes) -> str | None: ...

    @staticmethod
    def get_bytes(capture_id: str) -> bytes | None: ...

    @staticmethod
    def get_url(capture_id: str) -> str | None: ...

    @staticmethod
    def delete(capture_id: str) -> bool: ...
```

**Layout**: `captures/{capture_id}.png`. Bucket `fingerprints`
(config: `config.minio_bucket`).

**Verify**: upload + get_bytes roundtrip returns the same bytes.

### Task 3: `CaptureMinutiaRepository` (PG CRUD)

New module: `src/db/repositories/capture_minutia_repository.py`

```python
class CaptureMinutiaRepository:
    @staticmethod
    async def bulk_insert(session, *, capture_id, person_id, minutiae, algo_version) -> int: ...

    @staticmethod
    async def list_for_capture(session, capture_id) -> list[CaptureMinutia]: ...

    @staticmethod
    async def list_for_person(session, person_id) -> list[CaptureMinutia]: ...

    @staticmethod
    async def delete_for_capture(session, capture_id) -> int: ...
```

**Minutia hash** (forensic, computed at insert time):
```python
def _hash_minutia(m: dict) -> str:
    key = f"{m['x']:.6f}|{m['y']:.6f}|{m['angle']:.6f}|{m['type']}|{m.get('quality', 0):.6f}"
    return hashlib.sha256(key.encode()).hexdigest()
```

**Verify**: bulk_insert 64 minutiae, list_for_capture returns 64,
deletion cascades.

### Task 4: Enrollment flow refactor (single source of truth)

Modify `src/services/fingerprint_enrollment_service.py`:

```python
async def create_capture(self, fingerprint_id, image_bytes, ...):
    # 1. Quality pipeline (existing) — gets normalized image + minutiae
    pipeline = await loop.run_in_executor(None, mcc._run_quality_pipeline, image_bytes)
    normalized = pipeline["normalized"]  # 256x256 Gabor-enhanced
    minutiae = pipeline["minutiae"]

    # 2. Create capture row (placeholder URI until MinIO upload)
    capture = await FingerprintCaptureRepository.create(
        session, ..., image_uri="minio://pending/...",
    )

    # 3. Upload to MinIO
    png = cv2.imencode(".png", normalized)[1].tobytes()
    object_key = FingerprintStorage.upload(str(capture.id), png)
    if object_key:
        await FingerprintCaptureRepository.update(session, capture.id, image_uri=f"minio://{object_key}")

    # 4. Persist minutiae (with hash)
    for idx, m in enumerate(minutiae):
        m["index"] = idx
        m["hash"] = _hash_minutia(m)
    await CaptureMinutiaRepository.bulk_insert(session, capture_id=capture.id, person_id=fp.person_id, minutiae=minutiae)

    # 5. Index pairs in Qdrant (existing path)
    await self._index_pairs(capture, fp, image_bytes)
```

**Cut** (same deploy): remove `preview` method (was used for legacy
enhanced_image storage). The `/preview` endpoint uses its own logic.

**Verify**: enrollment creates capture with MinIO URI, minutiae in
PG, pairs in Qdrant. Old `enhanced_image` keyword arg removed from
`FingerprintCaptureRepository.create`.

### Task 5: Image endpoint — MinIO only, no 503 fallback

Modify `src/api/routers/captures.py`:

```python
@router.get(API_PREFIX + "/captures/{capture_id}/image")
async def get_capture_image(capture_id, session):
    c = await FingerprintCaptureRepository.get_by_id(session, capture_id)
    if c is None:
        raise HTTPException(404, "Capture not found")
    png = FingerprintStorage.get_bytes(str(c.id))
    if png is None:
        raise HTTPException(404, f"Image not in MinIO for capture {capture_id}")
    return Response(content=png, media_type="image/png")
```

**Cut** (no legacy):
- No `c.enhanced_image is None` check
- No 503 response
- No "re-enroll required" fallback message

**Verify**: GET on existing capture returns PNG from MinIO. GET on
capture without MinIO file returns 404 with clean message.

### Task 6: Cleanup — remove all `enhanced_image` references

Run `grep -rn "enhanced_image" apps/backend/src/` and delete every
match. Includes:

- `src/core/interfaces.py` (PipelineContext.enhanced_image)
- `src/services/mcc_matching_service.py` (`preview_thinning` return)
- `src/processing/post_hooks.py` (uses ctx.enhanced_image)
- `src/processing/spurious_filter.py`
- `src/processing/skeletonize_step.py`
- `src/processing/graph_extractor.py`
- `src/processing/of_filter.py`
- `src/db/models.py` (already removed, double-check)
- Any test files

The legacy `RidgeGraphExtractor` pipeline (mcc_descriptor.py,
graph_extractor.py) is dead code from the cylinder era. Remove
those too if they have no other users.

**Verify**: `grep -rn "enhanced_image" apps/backend/src/` returns
empty.

### Task 7: Re-enroll + verify

After all code changes:

```bash
# 1. Apply migration
cd apps/backend && uv run alembic upgrade head

# 2. Clean Qdrant (drop old pair_features, recreate)
curl -X DELETE http://localhost:6333/collections/pair_features

# 3. Re-enroll all 5 SOCOFING subjects via re-enrollment script
uv run python scripts/reenroll_pairs.py

# 4. Run standard benchmark
uv run python ../../scripts/benchmark_pairs.py
# Expected: 20/20 = 100% top-1

# 5. Run robustness benchmark
MCC_MULTI_ORIENT=1 uv run python ../../scripts/benchmark_pairs_robustness.py
# Expected: clean 100%, noise 80-100%, others improved from baseline
```

**Acceptance**: benchmark numbers match or exceed Phase 27 results.
All 6 scenarios show improvement over pre-Phase-28 baseline.

## File changes

| File | Action | Lines |
|---|---|---|
| `migrations/versions/0009_add_capture_minutiae.py` | create | ~50 |
| `src/services/fingerprint_storage.py` | create | ~70 |
| `src/db/repositories/capture_minutia_repository.py` | create | ~100 |
| `src/services/fingerprint_enrollment_service.py` | refactor | -30 +50 |
| `src/api/routers/captures.py` | simplify | -15 +5 |
| `src/db/models.py` | remove column | -5 |
| `src/services/mcc_matching_service.py` | remove `preview` | -20 |
| `src/processing/post_hooks.py` | remove | -50 (or refactor) |
| `src/processing/spurious_filter.py` | remove | (if dead) |
| `src/processing/skeletonize_step.py` | remove | (if dead) |
| `src/processing/graph_extractor.py` | remove | (if dead) |
| `src/processing/mcc_descriptor.py` | remove | (cylinder-era) |
| `src/core/interfaces.py` | remove `enhanced_image` field | -3 |

Net: ~+200 LOC (new modules) - ~500 LOC (dead cylinder code) = **-300 LOC**

## Rollback plan

If something fails in deploy:
1. Revert the migration: `alembic downgrade -1` (drops
   capture_minutiae, recreates enhanced_image column).
2. Restore code from git: `git revert <merge-commit>`
3. Re-enroll from MinIO backup: if MinIO data is intact, the
   restored code can read from there. Otherwise re-enroll from
   SOCOFING sources.

MinIO bucket is durable (data persists). The migration is
idempotent (re-running `alembic upgrade head` is safe).

## Commit strategy

Single PR with 4 commits:
1. `feat(28): MinIO storage service + capture_minutiae migration`
   (Tasks 1, 2, 3 — schema, MinIO wrapper, repository)
2. `refactor(28): enrollment uses MinIO + persists minutiae`
   (Task 4 — the integration)
3. `refactor(28): image endpoint serves from MinIO`
   (Task 5 — the read path)
4. `chore(28): remove legacy enhanced_image code`
   (Task 6 — the cleanup)

After all 4 commits: re-enroll + verify (Task 7).

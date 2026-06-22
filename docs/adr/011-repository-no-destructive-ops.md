# ADR-011: Repositories must not perform destructive operations

**Status**: Accepted
**Date**: 2026-06-22
**Phase**: 29 (Deep Embedding Migration)
**Deciders**: dev team
**Supersedes**: —

## Context

During Phase 29 cleanup we found two antipatterns in the Qdrant
adapter (`QdrantEmbeddingRepository`):

1. A `drop_old: bool = True` parameter on `ensure_collection()`.
2. Inline deletion of the legacy `ridge_graphs` collection inside
   `ensure_collection()`, gated on the same flag.

Both make a runtime request handler one method call away from
silently destroying a production gallery of 6000 vectors:

- Any dependency-injection call site (e.g. `get_embedding_repository`
  in `api/dependencies.py`) would run the destructive branch on
  startup if the default were flipped or the caller forgot the kwarg.
- The cleanup logic for the old `ridge_graphs` collection belonged to
  the Phase 29 migration. Inlining it into a hot path made it
  impossible to remove after the migration was done.
- Both behaviours were untestable: there was no way to assert "this
  method does not delete data" because the deletion was a side effect
  of the public API.

This is a class of problem, not just a Qdrant issue. The same
antipattern appeared earlier in `scripts/bulk_enroll_socofing.py`
(which directly called `MccMatchingService` and bypassed PG/MinIO).
The general rule being violated:

> Repositories own *reads and writes of their own data*. Lifecycle,
> migration, and destructive operations are separate concerns that
> belong in scripts and migration tools, not in request-handler
> call paths.

## Decision

1. **`QdrantEmbeddingRepository.ensure_collection()` no longer
   accepts a `drop_old` parameter.**  It is a no-op if the
   collection exists and creates it if missing.  The only
   configurable behaviour is dim validation, which raises
   `ValueError` on mismatch.

2. **No inline cleanup of unrelated collections.**  The
   `ridge_graphs` deletion block is gone.  Legacy collection
   deletion lives in `scripts/cleanup_qdrant.py`, which is run
   once during the migration, gated by human review, and reviewed
   in version control.

3. **The same rule applies to PG repositories.**  `EvidenceRepository`,
   `PersonRepository`, etc. do not accept `drop_table` / `truncate`
   parameters on their read paths.  PG schema changes are
   Alembic migrations only.

4. **Bulk-load scripts call REST endpoints, not repositories
   directly.**  `scripts/quick_enroll.py` now goes through
   `POST /persons → POST /fingerprints → POST /captures` so the
   normal validation, audit, and MinIO upload paths run.

## Consequences

### Positive
- Destructive operations require an explicit, human-reviewed code
  path. A future bug in dependency injection cannot wipe a gallery.
- The `ensure_collection()` method becomes trivially safe to call
  on every request.
- Cleanup scripts are first-class, diffable artifacts in the repo
  rather than hidden branches of a hot-path method.

### Negative
- A developer who wants to "reset the gallery" must now invoke a
  separate script instead of setting a flag.  This is the cost of
  safety.
- The dim-mismatch `ValueError` in `ensure_collection()` will
  surface loudly on first startup if config and existing collection
  disagree.  This is intentional — it is the right failure mode.

## Antipatterns to avoid in code review

Reject any PR that:

- Adds a `drop_*/reset_*/truncate_*` boolean to a repository method
  on a non-migration code path.
- Calls `delete_collection` / `drop_table` / `TRUNCATE` from inside
  a service that is also used by request handlers.
- Inlines "cleanup of legacy X" inside a method whose stated
  purpose is "ensure X exists".
- Lets a batch script call a repository directly when a REST
  endpoint would do.

## Related

- `apps/backend/src/db/qdrant_embedding_repository.py` —
  class docstring documents the rule.
- `apps/backend/scripts/cleanup_qdrant.py` — single canonical
  cleanup entry point for Qdrant collections.
- `apps/backend/src/db/migrations/versions/0010_drop_legacy_minutiae_and_graphs.py`
  — Alembic migration that drops the PG-side legacy tables.
- ADR-009 (Remove cylinder matcher) — same principle applied to
  matchers: dev = prod, no hidden branches.
- ADR-010 (Single source of truth for minutiae) — same principle
  applied to data: one canonical store, no parallel writes.

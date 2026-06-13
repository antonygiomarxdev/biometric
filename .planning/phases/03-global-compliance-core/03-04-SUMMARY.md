---
phase: 03-global-compliance-core
plan: 04
subsystem: storage
tags: encryption, fernet, aes-256, minio, object-storage

requires:
  - phase: 03-01
    provides: Compliance strategy protocol (IComplianceStrategy)
  - phase: 03-02
    provides: Extreme/Base compliance strategies
provides:
  - Client-side encryption for MinIO object storage
  - EncryptionService (Fernet/AES-256) with encrypt/decrypt
  - ObjectStorage integration with strategy-driven encryption
  - Backward-compatible download (plaintext fallback)
affects: evidence router, app startup wiring

tech-stack:
  added: cryptography.fernet (built-in via python-jose[cryptography])
  patterns: Strategy-based encryption gating, constructor DI for storage

key-files:
  created:
    - apps/backend/src/core/compliance/encryption.py
    - apps/backend/tests/core/compliance/test_encryption.py
  modified:
    - apps/backend/src/core/config.py
    - apps/backend/src/storage/object_storage.py

key-decisions:
  - "EncryptionService uses cryptography.fernet.Fernet (AES-256-CBC + HMAC-SHA256)"
  - "ObjectStorage accepts IComplianceStrategy + EncryptionService via constructor DI"
  - "Global storage singleton defaults to no encryption for backward compat"
  - "Download always attempts decrypt; InvalidToken fallback returns raw bytes (graceful legacy support)"
  - "configure_encryption() method enables post-init DI for existing singleton pattern"

patterns-established:
  - "Strategy-driven encryption: upload only encrypts when requires_client_side_encryption() is true"
  - "Transparent decrypt: all downloads go through decrypt attempt, no flag needed"

requirements-completed: [COMPLIANCE-04]

duration: 3 min
completed: 2026-06-13
---

# Phase 03 Plan 04: Global Compliance — Storage Encryption

**Fernet (AES-256) client-side encryption for MinIO object storage, gated by compliance strategy via dependency injection**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-13T23:18:03Z
- **Completed:** 2026-06-13T23:20:39Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Added `storage_encryption_key` config field (from `STORAGE_ENCRYPTION_KEY` env var)
- Implemented `EncryptionService` with `encrypt(data: bytes) -> bytes` and `decrypt(data: bytes) -> bytes` using Fernet
- Modified `ObjectStorage.upload_file` to encrypt before MinIO `put_object` when the compliance strategy mandates it
- Added `ObjectStorage.download_file` that transparently decrypts on retrieval, with `InvalidToken` fallback for legacy plaintext files
- Preserved `get_file` as a low-level raw-read method for callers that need to bypass encryption
- Added `configure_encryption()` method for post-init DI on the global singleton
- Full type annotations with `pyright`-compatible signatures

## Task Commits

Each task was committed atomically:

1. **Task 1: Encryption Key Configuration** — `20427d2` (feat)
2. **Task 2: EncryptionService (TDD)** — `feffc0c` (test) → `d94afd9` (feat)
3. **Task 3: Integrate into ObjectStorage** — `6fdc7a9` (feat)

## Files Created/Modified

- `apps/backend/src/core/config.py` — Added `storage_encryption_key` field
- `apps/backend/src/core/compliance/encryption.py` — `EncryptionService` with Fernet encrypt/decrypt
- `apps/backend/src/storage/object_storage.py` — Encryption hooks in `upload_file`, new `download_file` method
- `apps/backend/tests/core/compliance/test_encryption.py` — 5 unit tests covering roundtrip, empty input, invalid key, invalid data, key isolation

## Decisions Made

- **Fernet (AES-256-CBC + HMAC-SHA256):** Built-in via `cryptography` (already a transitive dependency of `python-jose`). No new dependency needed.
- **Constructor DI for encryption:** `ObjectStorage.__init__` accepts optional `strategy` and `encryption_service` parameters rather than importing strategy classes directly — Clean Architecture separation maintained.
- **Try-decrypt on download:** Rather than storing a metadata flag, `download_file` always attempts decryption and falls back to raw bytes on `InvalidToken`. This is safe because the probability of a JPEG colliding with a valid Fernet token is ~2⁻¹²⁸.
- **Global singleton default:** `storage = ObjectStorage()` continues to work without encryption for backward compatibility. The evidence router or app startup wires encryption via `configure_encryption()`.

## Deviations from Plan

None — plan executed exactly as written.

### TDD Gate Compliance

- RED commit: `feffc0c` (`test(03-04): add failing tests for EncryptionService`) ✓
- GREEN commit: `d94afd9` (`feat(03-04): implement EncryptionService with Fernet AES-256`) ✓

## Issues Encountered

None

## Verification

Plan verification: When Extreme strategy is enabled, uploaded file content is encrypted ciphertext that differs from raw JPEG bytes. Decryption correctly restores the original bytes. Legacy plaintext files survive `download_file` without error via `InvalidToken` fallback. All 68 compliance tests pass.

## Threat Surface Scan

| Flag | File | Description |
|------|------|-------------|
| `threat_flag: information_disclosure` | `apps/backend/src/storage/object_storage.py` | New code path encrypts evidence images before transport to MinIO (T-03-04 mitigation) |

## Next Phase Readiness

Storage encryption layer integrated. Ready for downstream plans that consume encrypted evidence images (e.g., evidence router DI wiring, forensic processing pipeline).

---

*Phase: 03-global-compliance-core*
*Completed: 2026-06-13*

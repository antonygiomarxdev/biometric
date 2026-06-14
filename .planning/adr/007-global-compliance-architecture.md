# ADR 007: Global Compliance, Privacy & Cross-Cutting Jurisdiction Architecture

**Date:** 2026-06-13
**Status:** Accepted
**Context:** Scaling a forensic AFIS system across different countries requires adapting not just UI text or AI prompts, but fundamental data handling, privacy, and security constraints. What is legal in Nicaragua may violate GDPR in Europe or LGPD in Colombia.

## Problem
Currently, security mechanisms (Audit Hashing), Logging (standard Python logger), and Storage (MinIO/PostgreSQL) are hardcoded. If a jurisdiction mandates that PII (Personally Identifiable Information) must be scrubbed from all system logs, or that biometric images must be encrypted application-side before hitting object storage, the current architecture requires rewriting core infrastructure.

## Decision: The "Compliance Context" Middleware
We will elevate "Jurisdiction" from a simple configuration to a **Cross-Cutting Compliance Strategy** that intercepts data at every boundary.

### 1. The IComplianceStrategy Interface
Every country/region implementation must define a strategy covering:
- **Logging & PII:** `scrub_pii(log_record)` -> Determines what data can be written to stdout/files.
- **Data Storage:** `requires_client_side_encryption()` -> Determines if the app must encrypt images/DB fields before sending them to MinIO/PostgreSQL.
- **Audit Trails:** `get_audit_strictness()` -> Determines if the standard SHA-256 chain is enough, or if external cryptographic time-stamping (RFC 3161) is required.
- **AI Boundaries:** `anonymize_prompt_data(data)` -> Strips names/IDs before sending data to an LLM, replacing them with tokens (e.g., `[SUSPECT_1]`).

### 2. Implementation: Core Middleware
- **Logs:** Override the standard Python `logging.Formatter` to pass all messages through the active Compliance Strategy's PII scrubber.
- **Storage:** Introduce an `EncryptionService` injected into `ObjectStorage` and `Repository` that activates based on the Compliance Strategy.
- **AI:** The `LLMFactory` and `ReportGenerator` will pass SQL data through the strategy before inference, and de-tokenize the response afterward.

## Consequences
- **Positive:** The system can be sold to any government globally. Achieving ISO 27701 (Privacy Information Management) or local certifications becomes a matter of writing a new Strategy file, not refactoring the core.
- **Negative:** Adds processing overhead (regex/scrubbing on logs, potential encryption/decryption cycles on storage).

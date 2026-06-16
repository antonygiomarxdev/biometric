---
name: spike-findings-biometric
description: Implementation blueprint from spike experiments. Requirements, proven patterns, and verified knowledge for building biometric. Auto-loaded during implementation work.
---

<context>
## Project: biometric

Migrar las capas críticas de seguridad (Auth) y persistencia (DB) del sistema biométrico hacia librerías modernas, mantenidas y asíncronas para eliminar riesgos de seguridad y cuellos de botella de rendimiento.

Spike sessions wrapped: 2026-06-16
</context>

<requirements>
## Requirements

- No se pueden usar librerías abandonadas (passlib, python-jose).
- El hashing de contraseñas DEBE usar Argon2id (OWASP recommendation).
- Las llamadas a base de datos en endpoints FastAPI DEBEN ser asíncronas (no bloqueantes) para escalar a alta concurrencia.
- La migración de passwords antiguos (bcrypt a Argon2id) debe ser soportada de forma transparente.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| Security and Async DB | references/security-and-async-db.md | Se comprobó la viabilidad de `pwdlib` (Argon2) y `AsyncSession` (SQLAlchemy) como drop-in replacements. |

## Source Files

Original spike source files are preserved in `sources/` for complete reference.
</findings_index>

<metadata>
## Processed Spikes

- 01-security-async-db
</metadata>
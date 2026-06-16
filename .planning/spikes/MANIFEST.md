# Spike Manifest

## Idea
Migrar las capas críticas de seguridad (Auth) y persistencia (DB) del sistema biométrico hacia librerías modernas, mantenidas y asíncronas para eliminar riesgos de seguridad y cuellos de botella de rendimiento.

## Requirements
- No se pueden usar librerías abandonadas (passlib, python-jose).
- El hashing de contraseñas DEBE usar Argon2id (OWASP recommendation).
- Las llamadas a base de datos en endpoints FastAPI DEBEN ser asíncronas (no bloqueantes) para escalar a alta concurrencia.
- La migración de passwords antiguos (bcrypt a Argon2id) debe ser soportada de forma transparente.

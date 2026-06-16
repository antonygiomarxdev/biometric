# Spike Conventions

Patterns and stack choices established across spike sessions. New spikes follow these unless the question requires otherwise.

## Stack
- Python 3.12+ gestionado vía `uv`
- Monorepo gestionado con `pnpm` y `Turborepo`

## Structure
- Los Spikes residen en `.planning/spikes/<numero>-<nombre>`
- El entorno virtual y las dependencias de prueba se inicializan dentro del directorio del propio Spike o usando scripts autocontenidos (`uv run`).

## Patterns
- **Database:** SQLAlchemy 2.0 sintaxis asíncrona (`AsyncSession`, `select`, `await session.execute`).
- **Security:** Hashing asíncrono robusto (Argon2id) y JWT nativos.

## Tools & Libraries
- `pwdlib[argon2]`
- `PyJWT`
- `sqlalchemy[asyncio]`
- `aiosqlite` (para mocks locales de DB)

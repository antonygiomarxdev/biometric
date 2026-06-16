# ADR 009: Modernización Crítica de Seguridad (Auth) y Concurrencia (DB)

## Estado
Aceptado / En Planificación

## Contexto
Durante una auditoría de arquitectura y dependencias descubrimos dos vulnerabilidades críticas en el diseño actual del backend:

1. **Riesgo de Seguridad en la Cadena de Suministro:**
   - La librería utilizada para el hash de contraseñas (`passlib`) está abandonada hace años y dejará de funcionar en Python 3.13 debido a la remoción del módulo interno `crypt`.
   - La librería de tokens JWT (`python-jose`) también está sin mantenimiento, representando un riesgo criptográfico en un sistema biométrico/forense que requiere los estándares más altos.
2. **Cuello de Botella Asíncrono en FastAPI:**
   - Se están usando constructos modernos asíncronos de FastAPI (`async def` en los endpoints), pero la comunicación con PostgreSQL se realiza mediante `psycopg2` y el motor clásico de `sqlalchemy` (`create_engine`, `Session`). Esto es I/O bloqueante puro, congelando el *Event Loop* de Python y penalizando dramáticamente el rendimiento del servidor ante concurrencia.

## Decisión
Basado en los resultados comprobados del **Spike 01** (`.planning/spikes/01-security-async-db`), hemos decidido:

### 1. Migración de Criptografía
*   **Hash de Contraseñas:** Reemplazar `passlib` con `pwdlib` (el sucesor oficial avalado por el equipo de FastAPI).
*   **Algoritmo:** Elevar el estándar de `bcrypt` a **Argon2id**. Argon2 es la recomendación actual de la Fundación OWASP por su resistencia frente a ataques mediante hardware paralelo (GPUs), vital si un volcado de la DB forense llega a verse comprometido.
*   **Manejo JWT:** Reemplazar `python-jose` por la librería nativa y mantenida activamente `PyJWT`.
*   **Transición:** Habilitar un *downgrade compatible* temporal. Si un usuario se loguea con un hash `bcrypt` antiguo, el sistema autorizará el acceso y lo reescribirá transparentemente a `Argon2id`.

### 2. Migración a SQLAlchemy 2.0 Asíncrono
*   Cambiar los drivers de persistencia de `psycopg2` a **`psycopg` (v3) asíncrono** (o `asyncpg`).
*   Modificar la inyección de dependencias (`src/api/dependencies.py`) para utilizar `create_async_engine` y `AsyncSession`.
*   Actualizar todos los repositorios (`src/db/repositories/`) para abandonar `.query()` a favor de la sintaxis moderna `await session.execute(select(...))`.

## Consecuencias
*   **Positivas:** El backend se blinda contra CVEs de librerías abandonadas, cumple con el estándar de grado militar Argon2, y aumenta su capacidad para servir peticiones en un orden de magnitud (10x-50x mejor rendimiento) gracias al I/O no bloqueante genuino.
*   **Negativas:** Obliga a refactorizar masivamente la sintaxis de todos los repositorios de datos y casos de uso subyacentes, lo cual es laborioso pero mecánicamente predecible.
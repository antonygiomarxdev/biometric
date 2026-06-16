# Spike 01: Modernización de Seguridad y SQLAlchemy Asíncrono

## Objetivo
Validar la viabilidad y el impacto de migrar las capas de base de datos y autenticación del sistema biométrico hacia librerías modernas, mantenidas y de alto rendimiento. Por la naturaleza crítica del sistema, la **seguridad es prioridad #1**.

## Contexto y Riesgos Actuales
1. **Riesgo Crítico de Seguridad (Passlib / Python-JOSE):** `passlib` lleva sin mantenimiento varios años. Dejará de funcionar en Python 3.13 debido a la remoción del módulo interno `crypt`. `python-jose` también está inactivo. En un sistema forense, usar bibliotecas de criptografía sin soporte activo es inaceptable.
2. **Cuello de Botella de Rendimiento (SQLAlchemy Síncrono):** El I/O bloqueante (por usar `psycopg2-binary` y `create_engine` tradicional) bloquea el event loop de FastAPI, colapsando bajo peticiones concurrentes altas (especialmente al registrar nuevas huellas o auditar).

## Experimentos Realizados en el Spike

### 1. Autenticación y Criptografía Segura (`auth_spike.py`)
- Reemplazo de `passlib` por **`pwdlib`**.
- Configuración de algoritmo **Argon2** (en lugar de bcrypt). OWASP recomienda fuertemente Argon2id para hashes de contraseñas de alta seguridad ya que es resistente a ataques de GPU (crítico hoy en día).
- Reemplazo de `python-jose` por **`PyJWT`**.
- *Resultado:* El código de encriptado y validación es un reemplazo casi directo (drop-in) para FastAPI. 

### 2. Capa de Base de Datos Concurrente (`db_spike.py`)
- Se validó la inicialización de **`create_async_engine`** y **`AsyncSession`**.
- Se comprobó la sintaxis moderna (SQLAlchemy 2.0) usando `Mapped` y `mapped_column`, eliminando los metadatos globales obsoletos.
- *Resultado:* Las consultas no bloqueantes funcionan a la perfección. La migración de `psycopg2` a `psycopg` (v3) permitirá consultas totalmente asíncronas sin necesidad de cambiar los dialectos de las consultas de SQLAlchemy actuales.

## Plan de Acción Propuesto (Migración Segura)

### Fase 1: Hardening de Seguridad (Auth)
1. Modificar `pyproject.toml` / `requirements.txt`:
   - Eliminar `passlib` y `python-jose`.
   - Añadir `pwdlib[argon2]` y `PyJWT`.
2. Actualizar `src/services/auth_service.py` aplicando los conceptos del spike.
3. Actualizar la inyección de dependencias `get_current_user` en FastAPI para usar el nuevo algoritmo de decodificación.
4. *Nota sobre datos heredados:* `pwdlib` soporta la lectura de hashes `bcrypt` antiguos y los puede re-hashear a `Argon2` en el momento en el que el usuario hace login de nuevo. (Estrategia de migración de contraseñas paulatina).

### Fase 2: Performance (Async DB)
1. Reemplazar `psycopg2-binary` por `psycopg[binary,pool]` o `asyncpg`.
2. Actualizar `src/api/dependencies.py`:
   - Cambiar a `create_async_engine`.
   - Modificar la dependencia `get_db` a `async def get_db() -> AsyncGenerator[AsyncSession, None]`.
3. Actualizar los repositorios en `src/db/repositories/`:
   - Cambiar todas las firmas de `session.query(...)` por `await session.execute(select(...))`.
   - Usar `result.scalars().all()` para obtener los resultados.

## Conclusión
La optimización es totalmente factible. El retorno de inversión (ROI) es altísimo dado que eliminamos dos vectores de riesgo de seguridad severo por uso de librerías abandonadas, y además liberamos todo el potencial de velocidad de FastAPI.

# Security and Async Database

## Requirements

- No se pueden usar librerías abandonadas (passlib, python-jose).
- El hashing de contraseñas DEBE usar Argon2id (OWASP recommendation).
- Las llamadas a base de datos en endpoints FastAPI DEBEN ser asíncronas (no bloqueantes) para escalar a alta concurrencia.
- La migración de passwords antiguos (bcrypt a Argon2id) debe ser soportada de forma transparente.

## How to Build It

### Seguridad y Criptografía (Auth)
1. Instalar `pwdlib[argon2]` y `PyJWT`.
2. Reemplazar la instancia de `CryptContext` por `PasswordHash` usando `Argon2Hasher`:
   ```python
   from pwdlib import PasswordHash
   from pwdlib.hashers.argon2 import Argon2Hasher

   password_hash = PasswordHash((Argon2Hasher(),))

   def hash_password(password: str) -> str:
       return password_hash.hash(password)

   def verify_password(plain_password: str, hashed_password: str) -> bool:
       return password_hash.verify(plain_password, hashed_password)
   ```
3. Para JWT, usar directamente `jwt.encode` y `jwt.decode` desde la librería `jwt` (PyJWT) en lugar de la importación `jose`.

### Base de Datos Asíncrona (SQLAlchemy 2.0)
1. Instalar un driver no bloqueante como `psycopg` (v3) con `[binary,pool]` o `asyncpg`.
2. Usar `create_async_engine` y `async_sessionmaker` en lugar de la versión síncrona:
   ```python
   from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

   engine = create_async_engine(DATABASE_URL)
   AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
   ```
3. En las dependencias de FastAPI (`get_db`), el generador debe ser `async def get_db() -> AsyncGenerator[AsyncSession, None]`.
4. En los repositorios, reemplazar `session.query(...)` con llamadas asíncronas explícitas:
   ```python
   stmt = select(User).where(User.username == username)
   result = await session.execute(stmt)
   user = result.scalar_one_or_none()
   ```

## What to Avoid

- Evitar el uso de dependencias bloqueantes tradicionales (`psycopg2-binary`) en FastAPI.
- No usar dependencias criptográficas abandonadas que van a romper en Python 3.13 debido a la deprecación de los módulos internos en C de Python.

## Constraints

- Se requiere reescribir la capa de persistencia (repositorios) para soportar constructos asíncronos (`await session.execute`).
- Se necesita soporte paulatino de contraseñas: `pwdlib` permite verificar un bcrypt antiguo y el servicio debe regenerar y guardar el hash como Argon2id silenciosamente.

## Origin

Synthesized from spikes: 01-security-async-db
Source files available in: sources/01-security-async-db/

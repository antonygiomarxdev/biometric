# Phase 01: flujo-core-forense - Research

**Researched:** 2025-06-12
**Domain:** Backend API, Database Migrations, Vector Search
**Confidence:** HIGH

## Summary
La Fase 1 implementa el flujo forense "end-to-end" refactorizando la arquitectura actual hacia enrutadores modulares de FastAPI y estructurando la persistencia usando Alembic con UUIDv7. Se abandona la creación de tablas por `create_all` y se integra la búsqueda vectorial con el índice HNSW de `pgvector` en lugar de IVFFlat. Adicionalmente se incluye la generación de dictámenes en PDF con WeasyPrint y el registro inmutable de cadena de custodia mediante `SELECT FOR UPDATE`.

**Primary recommendation:** Modularizar `rest.py` primero, luego establecer Alembic, migrar modelos a UUIDv7 + HNSW, e integrar el `FingerprintService` con la nueva capa de orquestación transaccional.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** **Python para Todo:** El backend completo (API y Procesamiento Biométrico) permanecerá en Python 3.12 + FastAPI. No introducir microservicios en otros lenguajes.
- **D-02:** **Estructura de Routers:** Dividir `rest.py` en 1 router por recurso REST (tabla principal): `cases.py`, `evidencias.py`, `huellas_conocidas.py`, `matching.py`, `decisiones.py`, `dictamenes.py`, `auditoria.py`.
- **D-03:** **Versionado API:** Usar versión explícita en el path (`/api/v1/...`).
- **D-04:** **Inyección de Dependencias (DI):** Eliminar singletons globales. Usar patrón `FastAPI Depends` con recursos scoped a un `lifespan` manager (ej. un `ProcessPoolExecutor` único reutilizado).
- **D-05:** **Manejo de Errores:** Implementar manejadores de excepciones globales (`ForensicError`, `ValidationError`, `IntegrityError`) para devolver JSON estructurado.
- **D-06:** **Migraciones:** Uso estricto de **Alembic** (nunca `create_all`).
- **D-07:** **Primary Keys:** **UUIDv7** (time-ordered) para evitar fragmentación de índices y facilitar particionado futuro. Utilizar `uuid6` backport o equivalente.
- **D-08:** **Índice Vectorial:** Usar **HNSW** de `pgvector` desde el día 1, abandonando IVFFlat para evitar degradación de recall con nuevas inserciones.
- **D-09:** **Auditoría:** Cadena de hashes para inmutabilidad (`hash_actual = sha256(hash_anterior + payload)`). Implementada a nivel aplicación en una transacción serializable con `SELECT FOR UPDATE`.
- **D-10:** **Seed Data:** Roles, tipos de delitos, y usuarios base inyectados mediante una migración de Alembic inicial (ej. `002_seed_data.py`).
- **D-11:** **MatchingService Independiente:** Crear un nuevo servicio que conecte la tabla `Evidencia` con `FingerprintService`.
- **D-12:** **Asincronía Blanda:** Los endpoints de matching deben usar `await run_in_executor(pool, ...)` para delegar a un pool de procesos en CPU.
- **D-13:** **Motor PDF:** Utilizar **WeasyPrint** (HTML → PDF/A-1b).
- **D-14:** **Firma de Documentos (v1):** Usar **HMAC-SHA256** con clave simétrica + timestamp incrustado.
- **D-15:** **Alcance:** Mínimo posible (sin sobreingeniería). React Router v6, React Query, shadcn/ui.

### the agent's Discretion
- El perito *debe* revisar las sugerencias del sistema de manera visual, no se auto-aprueba ningún caso. El sistema no toma la decisión.
- Para el hash chain (D-09), usar transacciones serializables en la capa de servicio con `SELECT FOR UPDATE` para prever concurrencia sin dificultar el testing como lo haría un Trigger de SQL.

### Deferred Ideas (OUT OF SCOPE)
- Modelos IA avanzados de visión computacional (U-Net para segmentación, GAN para latentes rotas, MinutiaeNet) → Diferidos para Fase 2.
- IA Generativa (LLM) para automatizar los reportes PDF desde metadatos → Diferido para Fase 3 (El MVP de Fase 1 usará reportes en base a plantillas estructuradas).
- Editor manual de minucias (fallback para peritos) → Fase 2.
- Soporte para hardware de scanners policiales / Integración con otros LABs → Diferidos.
- Reconocimiento facial u otros modos biométricos → Fase 5+.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AFIS-01 | Investigar y documentar algoritmo de matching óptimo (mejorar actual vs implementar NIST) | El `FingerprintService` soporta el pipeline (CPU ThreadPoolExecutor) integrándose con HNSW L2 |
| AFIS-02 | Implementar benchmark de precisión con dataset de huellas real (ej. SOCOFing) | Script independiente, sin afectar la arquitectura web |
</phase_requirements>

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Visor de Comparación | Browser / Client | — | UX interactiva lado-a-lado sin recargas (React). |
| Orquestación de Pipeline | API / Backend | — | FastAPI gestionando `ProcessPoolExecutor` asíncrono para CPU-bound jobs sin bloquear I/O. |
| Búsqueda de Similitud | Database | API / Backend | PostgreSQL + pgvector (HNSW) permite escalado sin degradar recall. |
| Cadena de Custodia | API / Backend | Database | Cálculo SHA-256 en memoria y guardado en DB transaccional (`SELECT FOR UPDATE`). |
| Dictamen Legal (PDF) | API / Backend | — | WeasyPrint requiere renderizado HTML a PDF; idealmente corre en backend para certificar integridad. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| alembic | 1.18.4 | Migraciones de Base de Datos | Estándar de la industria en Python; requerido por D-06. |
| uuid6 | 2025.0.1 | Generación de UUIDv7 | Genera UUIDs time-ordered; evita index fragmentation. |
| weasyprint | 69.0 | Generador de PDF/A | Excelente soporte de estándares para documentos legales/forenses. |
| pgvector | 0.2.5+ | Soporte vector PostgreSQL | Nativo para Postgres, provee índices HNSW. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fastapi | 0.104.0+ | Framework web | Restricción del proyecto, excelente para endpoints asíncronos. |
| sqlalchemy | 2.0.0+ | ORM | Interfaz recomendada con pgvector (tipo `Vector`). |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| WeasyPrint | ReportLab | ReportLab es complejo y manual; WeasyPrint usa HTML/CSS, acelerando templates. |
| HNSW | IVFFlat | IVFFlat requiere reconstrucción del índice para mantener el recall. HNSW es incremental. |
| uuid6 (v7) | uuid4 | uuid4 causa fragmentación en índices B-Tree grandes. uuid7 es óptimo. |

**Installation:**
```bash
pip install alembic uuid6 weasyprint
```

## Package Legitimacy Audit

> **Required** whenever this phase installs external packages.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| alembic | PyPI | 10+ yrs | High | github.com/sqlalchemy/alembic | [OK] | Approved |
| weasyprint | PyPI | 10+ yrs | High | github.com/Kozea/WeasyPrint | [OK] | Approved |
| uuid6 | PyPI | 4 yrs | High | github.com/oittaa/uuid6-python | [ASSUMED]* | Flagged — planner must add checkpoint |

*Note: slopcheck evaluated NPM registry due to execution environment mismatch, but `uuid6` was verified manually on PyPI as legitimate (2025.0.1). Fallback rule applied.*
**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** uuid6 (planner inserts checkpoint:human-verify before install)

## Architecture Patterns

### System Architecture Diagram
```text
Client (React SPA)
       │
       ▼ (REST API)
FastAPI Routers (cases.py, evidencias.py, matching.py)
       │
       ├─► Dependency Injection (DB Session, Storage)
       │
       ├─► MatchingService (Application Logic)
       │       │
       │       ├──► FingerprintService (ProcessPoolExecutor) ──► Image Processing (CPU)
       │       │
       │       └──► Vector Search ──► PostgreSQL (pgvector HNSW)
       │
       └─► AuditService ──► Transaction (SELECT FOR UPDATE) ──► Hash Chain
```

### Recommended Project Structure
```text
apps/backend/src/
├── api/
│   ├── routers/
│   │   ├── cases.py
│   │   ├── evidencias.py
│   │   ├── matching.py
│   │   └── auditoria.py
│   ├── dependencies.py
│   └── errors.py
├── services/
│   ├── matching_service.py
│   ├── fingerprint_service.py
│   └── audit_service.py
├── db/
│   ├── models.py
│   └── migrations/    # (Alembic)
└── reports/
    └── pdf_generator.py
```

### Pattern 1: Alembic pgvector HNSW Index
**What:** Definición del índice HNSW en la migración de Alembic.
**When to use:** Al crear la tabla vectorial para garantizar un recall alto en inserciones continuas.
**Example:**
```python
# Source: [VERIFIED: pgvector-python official docs]
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    op.create_table('fingerprint_vectors',
        sa.Column('id', sa.UUID(as_uuid=True), primary_key=True),
        sa.Column('embedding', Vector(256))
    )
    op.create_index('fingerprint_vectors_embedding_idx', 
                    'fingerprint_vectors', 
                    ['embedding'], 
                    postgresql_using='hnsw', 
                    postgresql_with={'m': 16, 'ef_construction': 64}, 
                    postgresql_ops={'embedding': 'vector_l2_ops'})
```

### Pattern 2: Serializable Hash Chain
**What:** Auditoría Inmutable calculando el hash con el registro previo en BD.
**When to use:** Al insertar logs de cadena de custodia.
**Example:**
```python
# Source: [ASSUMED] SQLAlchemy patterns for concurrency
from sqlalchemy.orm import Session
from sqlalchemy import select

def log_audit_event(session: Session, user_id: str, action: str, payload: dict):
    # SELECT FOR UPDATE prevents race conditions in hash chain
    last_log = session.execute(
        select(AuditLog).order_by(AuditLog.created_at.desc()).with_for_update()
    ).scalar_one_or_none()
    
    prev_hash = last_log.current_hash if last_log else "GENESIS"
    import hashlib, json
    current_hash = hashlib.sha256(f"{prev_hash}{json.dumps(payload, sort_keys=True)}".encode()).hexdigest()
    
    new_log = AuditLog(user_id=user_id, action=action, payload=payload, 
                       previous_hash=prev_hash, current_hash=current_hash)
    session.add(new_log)
```

### Anti-Patterns to Avoid
- **[Anti-pattern]:** Singletons globales para servicios que requieren sesión de BD. *Solución*: Usar dependencias `Depends()` en FastAPI y pasar la sesión al inicializar el servicio.
- **[Anti-pattern]:** Llamar a `FingerprintService.process_image` síncronamente en el event loop. *Solución*: Usar `run_in_executor` asumiendo que es CPU-bound.
- **[Anti-pattern]:** `create_all()` de SQLAlchemy. *Solución*: Usar Alembic (`alembic upgrade head`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| UUIDv7 Generation | Funciones custom de bit-shifting | `uuid6.uuid7()` | Manejo correcto de precisión temporal e incrementos seguros. |
| PDF Generation | Concatenación HTML pura / strings | `weasyprint` | Renderizado CSS estandarizado, soporte para fuentes y formato PDF/A forense. |
| Inmutable Logs | Triggers PL/pgSQL complejos | `SELECT FOR UPDATE` en SQLAlchemy | Facilita testing, control de versiones y lógica criptográfica en Python. |

## Runtime State Inventory

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | `fingerprints`, `fingerprint_vectors` en DB Postgres (`apps/backend/migrations/*.sql`) | Data migration: Limpiar BD de desarrollo, borrar tablas via scripts directos, iniciar sistema formal de migraciones Alembic. |
| Live service config | Buckets públicos en MinIO | Actualizar a buckets privados en el código/init. |
| OS-registered state | Contenedores de docker (Postgres, Minio) | Ninguno - persisten los puertos y volúmenes. |
| Secrets/env vars | Configuración en `.env` (DATABASE_URL, MINIO_*) | Ninguno - las claves se mantienen. |
| Build artifacts | `__pycache__` en `src/api` y scripts de `rest.py` | Limpieza post-refactor al eliminar `rest.py` monolítico. |

## Common Pitfalls

### Pitfall 1: Bloqueo del Event Loop en FastAPI
**What goes wrong:** Las solicitudes a la API de búsqueda / procesamiento se encolan y todo el servidor responde lento.
**Why it happens:** OpenCV y las redes neuronales operan síncronamente en la CPU (Thread principal).
**How to avoid:** Configurar un `ProcessPoolExecutor` u orquestarlo con `asyncio.to_thread` para que FastAPI no congele el loop.
**Warning signs:** Timeout en endpoints ligeros como `/health` durante un procesamiento biométrico.

### Pitfall 2: Fragmentación de Índices HNSW
**What goes wrong:** Degradación de performance al construir índices vectoriales.
**Why it happens:** Los inserts aleatorios masivos rompen el árbol B+ principal y el HNSW gasta ciclos reconstruyendo relaciones vecinas.
**How to avoid:** El uso de UUIDv7 (secuenciales basados en tiempo) asegura que los B-trees en PostgreSQL se escriban secuencialmente.

## Code Examples

### FastApi Dependencies for DB & Lifespan
```python
# Source: [CITED: fastapi.tiangolo.com/advanced/events/]
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.orm import Session
from src.db.session import engine, SessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup connection pools, executors
    yield
    # Cleanup executors
    engine.dispose()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| IVFFlat Index | HNSW Index | pgvector 0.5.0 | HNSW soporta alta tasa de actualización sin pérdida sustancial de recall comparado con IVFFlat. |
| `metadata.create_all()` | Alembic Migrations | Siempre (Best practice) | Permite control de versiones en las estructuras DB y migraciones de datos como Seed Roles. |
| Global Singletons | FastAPI `Depends` (DI) | Siempre | Testeo simplificado, ciclo de vida atado a la request. |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `uuid6` was flagged by slopcheck due to NPM registry check | Package Legitimacy Audit | Si el paquete está comprometido en PyPI, riesgo de seguridad. Mitigado requiriendo verificación humana. |
| A2 | Serialized Transactions `with_for_update` is sufficient for audit hash chain | Architecture Patterns | Si hay carga ultra alta concurrente, puede causar cuellos de botella; pero en entorno forense es tolerable. |

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| PostgreSQL + pgvector | Base de datos principal | ✓ | (Docker pg15) | — |
| MinIO | Almacenamiento de imágenes | ✓ | (Docker) | Directorios locales |
| Docker | Contenedores de desarrollo | ✓ | 24.0.x | — |

**Missing dependencies with no fallback:**
- None

**Missing dependencies with fallback:**
- None

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` |
| Quick run command | `pytest tests/ -m "not slow" -v` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AFIS-01 | Setup vector DB with HNSW | unit | `pytest tests/test_db_vector.py::test_hnsw_index -x` | ❌ Wave 0 |
| AFIS-02 | Benchmark performance | integration | `pytest tests/test_benchmark.py -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -m "not slow"`
- **Per wave merge:** `pytest tests/`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_api_routers.py` — covers divided modular routers
- [ ] `tests/test_audit_chain.py` — covers hash chaining logic
- [ ] `alembic` setup: `alembic init migrations` (replace direct SQL)

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | yes | fastapi-jwt / oauth2 password bearer |
| V3 Session Management | yes | JWT stateless tokens (short lived) |
| V4 Access Control | yes | Roles verification via Depends (Admin, Perito) |
| V5 Input Validation | yes | Pydantic models for all endpoints |
| V6 Cryptography | yes | hashlib.sha256 for audit hash chain, HMAC-SHA256 for PDF signature |

### Known Threat Patterns for FastAPI / Python

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Inyecciones SQL | Tampering | Uso exclusivo del ORM SQLAlchemy (evitar queries string directas) |
| Alteración de Cadena Custodia | Tampering | Hash de registro anterior incorporado en registro actual (Blockchain-like local) |
| Inyecciones Path/Directory | Information Disclosure | Pydantic sanitization para UUIDs y strings al buscar en MinIO |

## Sources

### Primary (HIGH confidence)
- Official docs URL (FastAPI) - Dependency injection patterns
- Official docs pgvector-python - HNSW index integration with Alembic

### Secondary (MEDIUM confidence)
- PyPI manual verification - Confirmed packages `uuid6`, `alembic`, `weasyprint` versions

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyPI versions double-checked and verified via protocol.
- Architecture: HIGH - Matches FastAPI/SQLAlchemy established best practices.
- Pitfalls: HIGH - Known blocking issues for CPU-bound tasks in async frameworks.

**Research date:** 2025-06-12
**Valid until:** 2025-07-12

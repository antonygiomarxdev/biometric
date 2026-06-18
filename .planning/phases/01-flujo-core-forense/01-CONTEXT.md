# Phase 01: flujo-core-forense - Context

**Gathered:** 2025-06-12
**Status:** Ready for planning

<domain>
## Phase Boundary

Implementar el flujo forense completo end-to-end (MVP Vertical): El perito crea un caso, ingresa evidencia latente, el sistema la procesa y busca contra base local AFIS, y el perito compara visualmente lado a lado para tomar una decisión (Identificación / Exclusión / Inconcluso). Finaliza con dictamen PDF/A y cadena de custodia en base de datos.
</domain>

<decisions>
## Implementation Decisions

### Arquitectura de Backend
- **D-01:** **Python para Todo:** El backend completo (API y Procesamiento Biométrico) permanecerá en Python 3.12 + FastAPI. No introducir microservicios en otros lenguajes.
- **D-02:** **Estructura de Routers:** Dividir `rest.py` en 1 router por recurso REST (tabla principal): `cases.py`, `evidencias.py`, `huellas_conocidas.py`, `matching.py`, `decisiones.py`, `dictamenes.py`, `auditoria.py`.
- **D-03:** **Versionado API:** Usar versión explícita en el path (`/api/v1/...`).
- **D-04:** **Inyección de Dependencias (DI):** Eliminar singletons globales. Usar patrón `FastAPI Depends` con recursos scoped a un `lifespan` manager (ej. un `ProcessPoolExecutor` único reutilizado).
- **D-05:** **Manejo de Errores:** Implementar manejadores de excepciones globales (`ForensicError`, `ValidationError`, `IntegrityError`) para devolver JSON estructurado.

### Base de Datos y Migraciones
- **D-06:** **Migraciones:** Uso estricto de **Alembic** (nunca `create_all`).
- **D-07:** **Primary Keys:** **UUIDv7** (time-ordered) para evitar fragmentación de índices y facilitar particionado futuro. Utilizar `uuid6` backport o equivalente.
- **D-08:** **Índice Vectorial:** Usar **HNSW** de `Qdrant` desde el día 1, abandonando IVFFlat para evitar degradación de recall con nuevas inserciones.
- **D-09:** **Auditoría:** Cadena de hashes para inmutabilidad (`hash_actual = sha256(hash_anterior + payload)`). Implementada **a nivel aplicación en una transacción serializable con `SELECT FOR UPDATE`**, no como trigger de SQL, para facilitar el testing.
- **D-10:** **Seed Data:** Roles, tipos de delitos, y usuarios base inyectados mediante una migración de Alembic inicial (ej. `002_seed_data.py`).

### Integración de Pipeline Biométrico
- **D-11:** **MatchingService Independiente:** Crear un nuevo servicio que conecte la tabla `Evidencia` con `FingerprintService`, manteniendo el pipeline genérico ignorante del dominio de "Casos/Evidencias".
- **D-12:** **Asincronía Blanda:** Los endpoints de matching deben usar `await run_in_executor(pool, ...)` para delegar a un pool de procesos en CPU, sin bloquear el event loop principal.

### Generación de Dictamen Legal
- **D-13:** **Motor PDF:** Utilizar **WeasyPrint** (HTML → PDF/A-1b) por su soporte de CSS forense y cumplimiento nativo del estándar de archivo a largo plazo.
- **D-14:** **Firma de Documentos (v1):** Usar **HMAC-SHA256** con clave simétrica + timestamp incrustado, y registrar el evento en `auditoria_log`. Escalable a certificados X.509 en el futuro.

### Frontend
- **D-15:** **Alcance:** Mínimo posible (sin sobreingeniería). React Router v6, React Query, shadcn/ui. Patrón centrado en casos (Dashboard → Caso → Comparación → Dictamen).
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planeamiento y Visión
- `.planning/PROJECT.md` — Visión corregida de la herramienta, roles, constraints on-premise, y workflows (Doble Motor IA).
- `.planning/ROADMAP.md` — Mapeo de la Fase 1 (MVP Vertical).
- `.planning/STATE.md` — Estado de avance y tareas de alto nivel.

### Código Base Existente (Para entender integración de pipelines)
- `apps/backend/src/services/fingerprint_service.py` — El pipeline core que no debe conocer del dominio "Casos", recibe array numpy, devuelve NormalizedFingerprint.
- `apps/backend/src/api/rest.py` — Monolito actual que debe ser refactorizado en la arquitectura modular de routers.
- `apps/backend/src/core/types.py` — Tipos de dominio existentes (Fingerprint, MinutiaCandidate).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `FingerprintService`: Utilizar la orquestación actual para Enhance → Extract → Normalize (Pipeline v1 Tradicional).
- Configuración de FastAPI / `Config`: Aprovechar clases Pydantic para validar env vars y manejar DB connections.

### Established Patterns
- **Frozen Dataclasses:** Continuar uso para datos inmutables de dominio biométrico (ej. `MinutiaCandidate`).
- **Arquitectura Limpia:** Respetar API → Services → Processing → Storage.

### Integration Points
- Refactorización de `rest.py`: Se requiere romper el monolito de 806 líneas inyectando dependencias correctamente y levantando un `lifespan` manager que posea el DB engine y el CPU thread pool.

</code_context>

<specifics>
## Specific Ideas

- El perito *debe* revisar las sugerencias del sistema de manera visual, no se auto-aprueba ningún caso. El sistema no toma la decisión.
- Para el hash chain (D-09), usar transacciones serializables en la capa de servicio con `SELECT FOR UPDATE` para prever concurrencia sin dificultar el testing como lo haría un Trigger de SQL.

</specifics>

<deferred>
## Deferred Ideas

- Modelos IA avanzados de visión computacional (U-Net para segmentación, GAN para latentes rotas, MinutiaeNet) → Diferidos para Fase 2.
- IA Generativa (LLM) para automatizar los reportes PDF desde metadatos → Diferido para Fase 3 (El MVP de Fase 1 usará reportes en base a plantillas estructuradas).
- Editor manual de minucias (fallback para peritos) → Fase 2.
- Soporte para hardware de scanners policiales / Integración con otros LABs → Diferidos.
- Reconocimiento facial u otros modos biométricos → Fase 5+.
</deferred>

---

*Phase: 01-flujo-core-forense*
*Context gathered: 2025-06-12*
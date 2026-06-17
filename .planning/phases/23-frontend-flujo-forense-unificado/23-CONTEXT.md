# Phase 23: Frontend — Flujo Forense Unificado — Context

**Gathered:** 2026-06-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Consolidar la UI del perito forense en un único flujo: **ver** detalles de huellas enroladas, **enrolar** nuevas huellas desde una persona pre-existente, y **buscar** candidatos visualizando **qué minucias del probe coincidieron con cuáles del candidato** (match trace a nivel de cilindro MCC). El flujo opera contra la API v1 con MCC (Phase 21). Eliminar todo el camino legacy (ScannerPage + cliente OpenAPI generado).

Fuera de alcance: auth, gestión de personas, audit, reports PDF, GenAI UI, facial (Phase 22). Phase 23 es un MVP operable con datos pre-sembrados desde SOCOFing.
</domain>

<decisions>
## Implementation Decisions

### Enrollment entry point
- **D-01:** Página dedicada `/enroll` (ruta nueva en `App.tsx`). Sin botón inline en ComparisonView.
- **D-02:** El perito selecciona una persona desde un dropdown poblado por el script seed. NO hay formulario de creación de persona en esta fase.
- **D-03:** La página `/enroll` es lineal: seleccionar persona → subir imagen → revisar/editar minucias → confirmar.

### Match explanation visualization
- **D-04:** Overlay side-by-side: canvas izquierdo = imagen latente con minucias del probe; canvas derecho = huella enrolada del candidato con minucias del candidato. Las minucias que matchearon comparten color identificable y se conectan con líneas entre los dos canvas.
- **D-05:** Cada par matched (probe_minutia_idx ↔ candidate_minutia_idx) se renderiza con un color por cilindro (paleta cíclica) para evitar colisiones visuales.
- **D-06:** Líneas conectoras usan opacidad proporcional al `similarity` del match (1.0 = opaca, 0.5 = translúcida).

### Candidate detail panel
- **D-07:** Al hacer click en un candidato, se expande un panel inferior con la huella enrolada (overlay de minucias que matchearon con su color) y la lista tabular de pares `(probe_cylinder_idx, candidate_cylinder_idx, similarity)`.
- **D-08:** Si el candidato tiene múltiples huellas enroladas, mostrar la que más cylinders aportó al score (la "fingerprint contributed most" en `contributing_fingerprints`).

### Coordinate system
- **D-09:** Pixel coords en cada canvas con `object-fit: contain`. No se aplica corrección de rotación ni normalización 0-1. El overlay muestra ambas imágenes lado a lado, alineadas al pixel, escaladas proporcionalmente al tamaño del contenedor.

### Backend extension (in-phase)
- **D-10:** Extender el response de `POST /api/v1/matching/search` con un nuevo campo `match_trace: list[MatchTraceEntry]` por candidato, donde:
  ```python
  class MatchTraceEntry(BaseModel):
      probe_cylinder_index: int       # índice de la minucia en el probe
      probe_x: int
      probe_y: int
      probe_angle: float
      candidate_capture_id: str
      candidate_fingerprint_id: str
      candidate_x: int
      candidate_y: int
      candidate_angle: float
      similarity: float               # cosine similarity en [0, 1]
  ```
  Esto se construye en `MccMatchingService.search()` antes de retornar, proyectando los `MccCylinderHit` con el join al payload de Qdrant (que ya tiene `person_id, fingerprint_id, capture_id` — falta agregar `x, y, angle` al insertar).
- **D-11:** Modificar `QdrantMccRepository.bulk_insert_cylinders()` para almacenar `x, y, angle` de cada minutia en el payload. El endpoint `extract_cylinders` ya retorna cilindros con esos datos (de `NormalizedFingerprint`).
- **D-12:** Añadir también `probe_minutiae: list[MinutiaSummary]` al response top-level (NO por candidato) para que el frontend pueda dibujar las minucias del probe sin un round-trip extra a `/extract`.

### Type alignment (frontend)
- **D-13:** Al inicio de Phase 23, actualizar `lib/api.ts`:
  - `MatchCandidate` → `{ person_id, total_score, hits, full_name, external_id, match_trace: MatchTraceEntry[] }`
  - `MatchSearchResponse` → añade `probe_minutiae: MinutiaSummary[]`, `query_time_ms: number`
- **D-14:** Regenerar `openapi.json` desde el backend Phase 21 + extensión Phase 23 antes de regenerar `src/client/`. Sin embargo, el frontend NO usará `src/client/` regenerado — se queda con `lib/api.ts` para tener control tipado manual.

### Legacy cleanup
- **D-15:** Eliminar `apps/frontend/src/pages/ScannerPage.tsx` y la ruta `/scanner` de `App.tsx`.
- **D-16:** Eliminar `apps/frontend/src/client/` completo (generado stale).
- **D-17:** Eliminar `apps/frontend/src/components/face/FaceViewer.tsx` (huérfano, Phase 22 lo rehace).
- **D-18:** Eliminar `apps/frontend/openapi.json` (ya no se usa `gen:client`).
- **D-19:** Quitar `@types/react` related a `DefaultService` y `OpenAPI` de `ScannerPage` (se van con el archivo).

### Data source for persons (MVP)
- **D-20:** Script `scripts/seed_socofing.py` (nuevo) que lee `apps/backend/static/SOCOFing/` y crea N `Person` records (uno por sujeto identificado en el dataset, con `external_id` derivado del filename y `full_name` sintético). Idempotente.
- **D-21:** El script NO inserta fingerprints — solo personas. Las huellas se enrolan interactivamente desde la UI (es la prueba del flujo).
- **D-22:** `scripts/load_socofing.py` ya existe — verificar si hace lo mismo; si sí, refactorizar a `seed_socofing.py` con la semántica correcta de Phase 23.

### Reusable components
- **D-23:** Reusar `useCanvasDrawer` como base para el canvas de match overlay. Extender con un modo nuevo `"match"` que reciba `matchTrace` y dibuje las líneas conectoras entre canvas hermanos (requiere refactor del hook para emitir eventos o un componente wrapper).
- **D-24:** Reusar `MinutiaeEditor` para el paso de revisión de minucias en `/enroll`, sin cambios (modo view + edit).
- **D-25:** Reusar `lib/query.tsx` (QueryClient setup) sin cambios.

### Routing
- **D-26:** `App.tsx` añade ruta `/enroll` y `/cases/:caseId/enroll` (atajo contextual desde un caso). Mantiene `/` y `/cases/:caseId/compare`.
- **D-27:** Dashboard (`/`) muestra botón "Enrolar Huella" prominente que navega a `/enroll`.

### API client consolidation
- **D-28:** Toda la comunicación con `/api/v1/...` pasa por `lib/api.ts`. No se importa `DefaultService` en ningún archivo de la app.
- **D-29:** `lib/api.ts` añade funciones: `listPersons(skip, limit)`, `getPerson(id)`, `enrollFingerprint(personId, file)`, `getMinutiaeForImage(file)` (wrapper de `/extract` para previsualización en enrollment).

### Claude's Discretion
- Diseño visual exacto de los canvas (colores, tamaños, animaciones de aparición de minucias).
- Manejo de errores de carga de imagen (retry, fallback a imagen vacía).
- Skeleton de carga en `/enroll` mientras se cargan las personas.
- Estructura interna del componente `MatchOverlay` (puede ser un componente compuesto o varios).
- Validaciones de tamaño/formato de imagen (ya existen en `ComparisonView` — replicar el patrón).
- Decidir si las minucias del probe se obtienen vía `/extract` previo o vía el nuevo `probe_minutiae` del search response (preferir el response si está disponible).
</decisions>

<specifics>
## Specific Ideas

- **"MVP para probar":** el usuario explícitamente pidió un MVP operable, no un sistema completo. Las decisiones reflejan esto: sin gestión de personas, sin auth, sin inline enrollment.
- **Reutilización pragmática:** "lo que existe creo que mira que puedes usar y que tiras" — el usuario pidió reusar lo útil (MinutiaeEditor, useCanvasDrawer, RegistrationForm si aplica) y eliminar lo que sobra.
- **Pre-seed con SOCOFing:** el usuario propuso sembrar personas desde SOCOFing vía script para que el enrollment + search funcione end-to-end sin formularios de creación. Es un atajo válido porque SOCOFing ya está validado como dataset (Phase 20).
- **Visualización como diferenciador:** "que me salgan las minucias, los puntos, cuando busque que me salga con cual minucia coincide" — el foco explícito es la visualización del match, no la gestión de identidad.
- **Doctrina "No Legacy":** aplicar a frontend. Si Phase 21 deja el backend en MCC, el frontend no debe tener un fallback a Delaunay.
</specifics>

<canonical_refs>
## Canonical References

### Backend MCC (a extender en Phase 23)
- `.planning/phases/21-mcc-integration/PLAN.md` — Plan actual del backend MCC; tasks 1-10 definen `MccMatchingConfig`, `MccCylinderHit`, `IMccMatcher`, `QdrantMccRepository`, `MccMatchingService`. Phase 23 añade tasks para el `match_trace`.
- `apps/backend/src/processing/mcc_descriptor.py` — `CylinderConfig` (12 sectors × 4 rings × 3 features = 144D), `extract_cylinders()`. La salida de cada cilindro son los 144D; el position info viene del minutia input dict.
- `apps/backend/src/db/qdrant_mcc_repository.py` — `QdrantMccRepository` (a crear en Phase 21 Task 4). Phase 23 modifica `bulk_insert_cylinders` para persistir `x, y, angle` en el payload.
- `apps/backend/src/services/mcc_matching_service.py` — `MccMatchingService.search()` retorna `MccSearchHit[]`. Phase 23 modifica el retorno para incluir `match_trace`.
- `apps/backend/src/core/types.py` (líneas 251-270 aprox) — `MccCylinder`, `MccCylinderHit`, `MccPersonHit`. Phase 23 define `MatchTraceEntry` adyacente.
- `apps/backend/src/api/routers/latent_search.py` — `POST /api/v1/matching/search`. Phase 23 modifica el response model.

### Frontend existente (a reusar o eliminar)
- `apps/frontend/src/lib/api.ts` — Cliente fetch manual para v1. Punto único de cambio para tipos.
- `apps/frontend/src/lib/query.tsx` — TanStack Query setup.
- `apps/frontend/src/components/fingerprint/MinutiaeEditor.tsx` — Canvas con view/add/delete/move. Reusar en `/enroll`.
- `apps/frontend/src/hooks/useCanvasDrawer.ts` — Hook de dibujo de minucias. Extender para match overlay.
- `apps/frontend/src/components/fingerprint/FingerprintViewer.tsx` — Visualizador biométrico con stats overlay. Referencia para el detail panel.
- `apps/frontend/src/components/fingerprint/RegistrationForm.tsx` — Form de registro. Ver si se reusa o se elimina.
- `apps/frontend/src/components/fingerprint/FingerprintList.tsx` — Lista de huellas. Ver si se reusa.
- `apps/frontend/src/components/fingerprint/ResultPanel.tsx` — Panel de resultado legacy. Reemplazar.
- `apps/frontend/src/components/layout/MainLayout.tsx` + `Sidebar.tsx` — Layout components sin uso. Evaluar integración o eliminación.
- `apps/frontend/src/components/ui/*` — `card`, `button`, `input`, `badge`, `toast`, `dropdown-menu`. Reusar sin cambios.
- `apps/frontend/src/pages/Dashboard.tsx` — Lista de casos. Modificar para añadir botón "Enrolar Huella".
- `apps/frontend/src/pages/ComparisonView.tsx` — Flujo principal de comparación. Refactorizar para consumir `match_trace`.
- `apps/frontend/src/App.tsx` — Router. Añadir `/enroll` y `/cases/:caseId/enroll`, eliminar `/scanner`.
- `apps/frontend/openapi.json` — Stale (8 endpoints legacy). Eliminar.

### Dataset y scripts
- `apps/backend/static/SOCOFing/` — Dataset SOCOFing (huellas + metadata). Fuente del seed.
- `apps/backend/tests/fixtures/socofing_fixtures.py` — Fixtures de testing. Referencia para el formato.
- `scripts/load_socofing.py` — Script existente. Evaluar si se refactoriza o se reemplaza.
- `scripts/test_socofing.py` — Test runner del dataset.

### Spike findings
- `.claude/skills/spike-findings-biometric/SKILL.md` — Patrones validados (Argon2id, AsyncSession). No directamente relevante para Phase 23, pero documenta la doctrina de "no usar librerías abandonadas".

### Project-level
- `.planning/PROJECT.md` §"Pivot Estratégico del MVP" — Define el foco en criminalística forense con MCC + Gabor.
- `.planning/REQUIREMENTS.md` — UI-03 (panel de resultados forenses), UI-06 (visualización de minucias superpuestas), AFIS-03 (tasa de identificación aceptable). Phase 23 contribuye a UI-03 y UI-06.
- `.planning/STATE.md` §"Tech Stack" — Confirma MCC como matching vigente.
</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`useCanvasDrawer`** (`apps/frontend/src/hooks/useCanvasDrawer.ts:62`): Hook genérico para dibujar minucias con modos view/add/delete/move. **Extender** con un modo `"match"` que reciba `(imageSource, probeMinutiae, candidateMinutiae, matchTrace)` y dibuje dots en ambos canvas + líneas conectoras via un evento compartido.
- **`MinutiaeEditor`** (`apps/frontend/src/components/fingerprint/MinutiaeEditor.tsx:1`): Editor interactivo con 4 modos. **Reutilizar tal cual** en `/enroll` para que el perito revise/edite minucias extraídas antes de confirmar.
- **`FingerprintViewer`** (`apps/frontend/src/components/fingerprint/FingerprintViewer.tsx:12`): Visualizador con stats overlay (minutiae count, terminations, bifurcations). **Referencia visual** para el detail panel de Phase 23.
- **`lib/api.ts`** (`apps/frontend/src/lib/api.ts:1`): Cliente fetch manual. **Único punto de mantenimiento** de tipos y endpoints v1.
- **`lib/query.tsx`**: TanStack Query setup. Sin cambios.
- **UI primitives** (`apps/frontend/src/components/ui/*`): Reusar.

### Established Patterns
- **Fetch con wrapper `request<T>()`**: El helper maneja FormData vs JSON, status codes, error wrapping. Toda nueva función de `lib/api.ts` debe usarlo.
- **Tipado estricto con TS strict**: Cero `any`, cero `@ts-ignore`. Los tipos en `lib/api.ts` deben ser interfaces con campos snake_case (mirror de Pydantic).
- **TanStack Query para fetching**: `useQuery` con `queryKey` jerárquico (`["cases"]`, `["case", caseId]`). Nuevas queries: `["persons"]`, `["match", caseId]`.
- **Toast para feedback**: `useToast()` de `@/components/ui/toast`. Ya integrado en `ComparisonView`.

### Integration Points
- **Backend Phase 21 (MCC)**: El endpoint `/api/v1/matching/search` está siendo migrado en Phase 21. Phase 23 espera que Phase 21 esté mergeado (o coordina con su branch).
- **Backend Phase 17 (Data Model)**: `Person` y `Fingerprint` y `FingerprintCapture` ya están modelados. Phase 23 consume sus endpoints v1 (`/api/v1/persons`, `/api/v1/captures`).
- **Backend script `load_socofing.py`**: Si ya cubre el seed, refactorizar; si no, crear `seed_socofing.py`.
- **SOCOFing dataset**: `apps/backend/static/SOCOFing/` con estructura de directorios. Necesita inspección previa al seed script.
</code_context>

<deferred>
## Deferred Ideas

- **Gestión de personas (CRUD completo):** Página para listar, crear, editar, eliminar personas. Diferido a fase futura. Phase 23 consume personas pre-sembradas.
- **Auth y login UI:** Diferido. Phase 23 asume que el sistema corre en LAN de laboratorio.
- **Auditoría visual (audit log viewer):** Diferido. El backend audita; la UI no lo muestra.
- **Reportes PDF:** Diferido. El backend `reports.py` existe pero no se consume.
- **GenAI UI (asistente forense + generación de dictámenes):** Diferido. Backend `genai.py` listo, sin UI.
- **Inline enrollment en ComparisonView:** Diferido. Si la búsqueda no encuentra match, hoy se ofrece solo "veredicto: inconcluso"; no se ofrece enrolar desde ahí.
- **Facial recognition UI:** Phase 22. FaceViewer eliminado en Phase 23 porque es huérfano.
- **i18n / l10n:** Diferido. Strings hardcoded en español, alineado con AGENTS.md.
- **Tests E2E con Playwright / Vitest:** Diferido. Phase 23 valida con testing manual sobre el dataset SOCOFing.
- **Mobile / responsive:** Diferido. Web-first (perito en escritorio de laboratorio).
- **Real-time updates (WebSocket):** Diferido. TanStack Query con polling manual si es necesario.
- **Multi-capture per finger (roll, slap, plain):** Diferido. Phase 23 soporta una captura por enrollment.
- **Histograma de scores / ROC curve en UI:** Diferido. La validación numérica ya existe en Phase 20 spike.
- **Persona picker autocomplete / búsqueda fuzzy:** Diferido. Para 10-100 personas pre-sembradas, un `<select>` nativo es suficiente.
- **Exportar match trace a PDF/cadena de custodia:** Diferido. La decisión del perito se registra en DB; el reporte PDF viene en fase futura.
- **Comparativa side-by-side de DOS candidatos simultáneos:** Diferido. Phase 23 muestra uno a la vez (selected card).
- **Manejo de imágenes corruptas / no-huella (validación cliente):** Diferido. El backend puede rechazar; el cliente solo muestra el error.
- **Animaciones y transiciones premium (framer-motion):** Diferido. CSS transitions existentes bastan.
</deferred>

---

*Phase: 23-frontend-flujo-forense-unificado*
*Context gathered: 2026-06-17*

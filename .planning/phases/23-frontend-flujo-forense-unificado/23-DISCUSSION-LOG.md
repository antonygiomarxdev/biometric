# Phase 23: Frontend — Flujo Forense Unificado — Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-17
**Phase:** 23-frontend-flujo-forense-unificado
**Areas discussed:** Scope initial, Inventory, Phase numbering, Enrollment entry, Match explanation, Candidate detail, Legacy cleanup, Backend extension, Enrollment UX unification, Coordinate system, Type alignment, Inline person data

---

## Initial Scope Discovery

| Option | Description | Selected |
|--------|-------------|----------|
| Integración end-to-end MCC | Consolidar ScannerPage, regenerar OpenAPI, integrar auth/audit/reports | |
| UI/Auth base + Layout unificado | Login, MainLayout, i18n, design system primitives | |
| Limpieza de deuda técnica | Eliminar legacy, regenerar cliente, eliminar huérfanos | |
| Vertical slice — Personas + Capturas | Gestión de personas, captura vía v1, audit panel | |
| **Enfoque del usuario (freeform)** | "honestamente de momento lo que quiero es una parte donde tenga para ver las huellas, poder enrolar poder buscar, que me salgan las minucias, los puntos, cuando busque que me salga con cual minucia coincide, etc, no tanto tener usuarios y demas, eso puede esperar un poco" | ✓ |

**Notes:** Usuario explícitamente pidió reducir scope: ver + enrolar + buscar con visualización de match. Diferir usuarios/auth/audit.

---

## Enrollment Entry Point

| Option | Description | Selected |
|--------|-------------|----------|
| Inline post-búsqueda en ComparisonView | Botón "Enrolar este caso" cuando búsqueda no encuentra match | |
| Página dedicada `/enroll` full-screen | Wizard lineal subir → revisar → confirmar | |
| Modal desde Dashboard | "+ Nueva captura" abre modal | |
| Reusar ScannerPage como enrollment-only | Convertir ScannerPage en enrollment-only | |
| **Híbrido (user freeform)** | "lo que existe creo que mira que puedes usar y que tiras, pero en general tener la opcion como la 1 y como la 2, total esto posiblemente cambie, lo quiero como un mvp para probar" | ✓ (refinado) |

**Notes:** Decisión refinada en ronda posterior a "página dedicada /enroll" con dropdown de personas pre-sembradas (sin inline form). Usuario aprobó refinamiento cuando se aclaró el patrón de seed SOCOFing.

---

## Match Explanation Visualization

| Option | Description | Selected |
|--------|-------------|----------|
| Overlay lado a lado con líneas conectoras | Probe a la izquierda, candidato a la derecha, líneas conectando matched minutiae | ✓ |
| Heatmap en probe | Solo imagen latente con intensidad por minucia | |
| Lista tabular de pares coincidentes | Tabla (probe, candidate, similitud) | |
| Compuesto: overlay + lista on-click | Overlay principal + panel expandible | |

**User's choice:** "Overlay lado a lado con líneas (Recommended)"

---

## Candidate Detail Panel

| Option | Description | Selected |
|--------|-------------|----------|
| Match trace a nivel de cylinder | Imagen enrolada + overlay minucias + score por cylinder | ✓ |
| Historial completo del candidato | TODAS las huellas enroladas de la persona | |
| Solo metadata + score | Nombre, ID, score (patrón actual) | |
| Vista comparativa fullscreen | Navegación a /cases/{id}/compare/{candidateId} | |

**User's choice:** "Match trace a nivel de cylinder (Recommended)"

---

## Legacy ScannerPage Fate

| Option | Description | Selected |
|--------|-------------|----------|
| Eliminar en bloque (ScannerPage + client/ + openapi.json) | Doctrina No Legacy estricta | ✓ |
| Deprecar con feature flag | Mantener accesible, marcar como deprecated | |
| Mantener paralelo indefinidamente | Acumula deuda | |
| Eliminar ScannerPage pero regenerar OpenAPI client | Cliente regenerado como single source | |

**User's choice:** "Eliminar en bloque (Recommended)"

---

## Backend Extension for Per-Minutia Match Data

| Option | Description | Selected |
|--------|-------------|----------|
| Extender `/matching/search` con `match_trace` field | Backend extension dentro de Phase 23 | ✓ |
| Endpoint dedicado `/matching/search/{case_id}/trace` | Sin tocar Phase 21 | |
| Re-match client-side con Qdrant | Costoso, zero backend change | |
| Mostrar solo minucias del probe, no links | Honesto pero pierde visualización de match | |

**User's choice:** "Extender backend en Phase 23 (Recommended)"

**Notes:** Decisión crítica. El backend MCC actual solo expone score agregado, no qué cilindro matcheó con cuál. Se modifica `QdrantMccRepository.bulk_insert_cylinders` para persistir `x, y, angle` en payload, y `MccMatchingService.search` para proyectar match trace.

---

## Enrollment UX Unification (Inline + /enroll)

| Option | Description | Selected |
|--------|-------------|----------|
| EnrollmentWizard compartido (subir → revisar → confirmar) | Reusar en inline y /enroll | (cuestionado) |
| Reutilizar RegistrationForm + MinutiaeEditor sin cambios | Modal en ComparisonView, página en /enroll | (cuestionado) |
| Solo /enroll, sin inline (navegación post-search) | Sin componentes inline duplicados | (cuestionado) |
| **Refinamiento a seed SOCOFing (user freeform)** | "dale, pero si metemos lo de persona se nos complica esta vuelta, pq tendriamos que tener el registrar personas etc, que piensas?" | ✓ (refinado) |

**Notes:** Usuario identificó correctamente que la creación inline de personas en el wizard complicaba Phase 23. Solución propuesta y aceptada: pre-sembrar personas desde SOCOFing vía script; enrollment es solo "elegir persona + subir imagen".

---

## Coordinate System for Overlay

| Option | Description | Selected |
|--------|-------------|----------|
| Pixel coords + scaling proporcional | object-fit contain, coords en pixel nativo de cada imagen | ✓ |
| Normalizado [0,1] con pose correction | Aplicar ángulo del minutia como rotación | |
| Solo overlay en probe | Candidato a la derecha sin overlay | |

**User's choice:** "Pixel coords + scaling proporcional (Recommended)"

---

## Type Alignment

| Option | Description | Selected |
|--------|-------------|----------|
| Actualizar tipos al inicio de Phase 23 | Asumir contrato nuevo de Phase 21 desde día 1 | ✓ |
| Adapter con backward-compat | Normalizar ambas formas | |
| Esperar a Phase 21 | Riesgo: Phase 23 bloqueada | |

**User's choice:** "Actualizar tipos al inicio (Recommended)"

---

## Inline Person Data

| Option | Description | Selected |
|--------|-------------|----------|
| Inline "Datos del Individuo" en wizard | Nombre + cédula en paso 2 | (rechazado) |
| Enrollment sin persona (huella huérfana) | Atrasa el problema: matching no funciona | (rechazado) |
| Picker de persona preexistente | Solo si admin ya creó personas (bloquea flujo) | (rechazado) |
| Diferir enrollment a fase futura | Sacar enrollment de Phase 23 | (rechazado) |
| **Seed SOCOFing (user freeform)** | "podemos crear personas random en base a lo que tenemos en socofing, con algun script y de momento al buscar pues debe coincidir con alguien que ya este registrado no?" | ✓ |

**User's choice:** "podemos crear personas random en base a lo que tenemos en socofing, con algun script y de momento al buscar pues debe coincidir con alguien que ya este registrado no?"

**Notes:** Excelente simplificación. SOCOFing ya está validado como dataset (Phase 20). El script seed es dev tool, no producción. En producción, fase futura añade el form de creación de personas.

---

## Claude's Discretion

- Diseño visual exacto de los canvas (colores, tamaños, animaciones de aparición de minucias).
- Manejo de errores de carga de imagen (retry, fallback a imagen vacía).
- Skeleton de carga en `/enroll` mientras se cargan las personas.
- Estructura interna del componente `MatchOverlay` (compuesto vs monolítico).
- Validaciones de tamaño/formato de imagen (replicar patrón de ComparisonView).
- Decidir si las minucias del probe se obtienen vía `/extract` previo o vía el nuevo `probe_minutiae` del search response (preferir el response si está disponible).
- Evaluación de `scripts/load_socofing.py` existente: refactor vs reemplazo.
- Selección de paleta de colores para los pares matched (cíclica por índice).

---

## Deferred Ideas

- **Gestión de personas (CRUD):** Diferido a fase futura. Phase 23 consume personas pre-sembradas.
- **Auth y login UI:** Diferido. Sistema asume LAN de laboratorio.
- **Auditoría visual:** Diferido. Backend audita, UI no lo muestra.
- **Reportes PDF:** Diferido. Backend `reports.py` existe pero no se consume en esta fase.
- **GenAI UI (asistente + dictámenes):** Diferido. Backend `genai.py` listo, sin UI.
- **Inline enrollment en ComparisonView:** Diferido.
- **Facial recognition UI:** Phase 22. FaceViewer eliminado en Phase 23.
- **i18n / l10n:** Diferido. Strings hardcoded español (alineado con AGENTS.md).
- **Tests E2E (Playwright/Vitest):** Diferido. Validación manual con SOCOFing.
- **Mobile / responsive:** Diferido. Web-first escritorio.
- **WebSocket / real-time updates:** Diferido.
- **Multi-capture per finger (roll, slap, plain):** Diferido. Una captura por enrollment.
- **Histograma ROC en UI:** Diferido. Validación numérica ya existe (Phase 20).
- **Persona picker con autocomplete / fuzzy:** Diferido. `<select>` nativo suficiente para 10-100 personas.
- **Exportar match trace a PDF / cadena de custodia:** Diferido.
- **Comparativa simultánea de DOS candidatos:** Diferido.
- **Validación cliente de imagen (huella válida vs foto random):** Diferido. Backend rechaza; cliente solo muestra error.
- **Animaciones premium (framer-motion):** Diferido. CSS transitions existentes bastan.

---

*Phase: 23-frontend-flujo-forense-unificado*
*Discussion log generated: 2026-06-17*

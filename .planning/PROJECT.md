# Biometric — Sistema AFIS para Identificación Gubernamental

## What This Is

Sistema biométrico de identificación por huellas dactilares (AFIS) diseñado para uso gubernamental y forense. Procesa imágenes de huellas, extrae minucias, y realiza identificación por matching vectorial. Actualmente en fase de investigación y desarrollo, con visión de convertirse en plataforma multimodal (huellas + facial + iris).

## Core Value

Identificar personas por sus huellas dactilares con precisión forense, rapidez y auditabilidad, en un sistema propio y soberano para el gobierno de Nicaragua.

## Context

- Proyecto iniciado como aprendizaje personal que evolucionó a necesidad real identificada en Nicaragua
- Contacto informal con potenciales usuarios en el ámbito gubernamental
- Backend funcional con pipeline de procesamiento (enhancement → extracción → normalización → matching)
- Arquitectura Clean Code con Strategy Pattern para múltiples biometrías
- Matching actual: pgvector L2 + reranking coseno (híbrido)
- Sin autenticación, sin auditoría, sin CI/CD
- Face provider es stub (no implementado)
- Sistema desplegable en modalidad híbrida: servidores on-premise y equipos forenses locales

## Constraints

- **Tecnología**: Stack actual Python/FastAPI/PostgreSQL mantenerse o evolucionar justificadamente
- **Seguridad**: Al ser uso gubernamental/forense, autenticación, autorización y audit trail son críticos
- **Precisión**: El matching debe cumplir estándares forenses (investigar qué estándares aplicar)
- **Despliegue**: On-premise (servidores gobierno) + equipos forenses locales → sin dependencia cloud
- **Rendimiento**: Capacidad de procesar volúmenes grandes de huellas (escala civil)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Nombre "Biometric" | Renombrado de BioSecure Gov para identidad propia | — Pending |
| Matching híbrido (pgvector + coseno) | Implementación actual, investigar si cumple estándar AFIS | — Pending |
| On-premise + forense | Sin cloud, control total de datos | — Pending |

## Requirements

### Validated

- ✓ Extracción de minucias desde imágenes de huellas — existing
- ✓ Pipeline de procesamiento (enhance + extract + normalize) — existing
- ✓ Registro de personas con huellas — existing
- ✓ Identificación por similitud vectorial — existing
- ✓ Almacenamiento de imágenes en MinIO — existing
- ✓ API REST con 8 endpoints — existing
- ✓ Frontend React con UI de identificación/registro — existing
- ✓ Soporte GPU (CuPy) con fallback CPU — existing
- ✓ Arquitectura Strategy para múltiples biometrías — existing

### Active

- [ ] **AFIS-01**: Investigar y definir algoritmo de matching con precisión forense
- [ ] **AFIS-02**: Autenticación y control de acceso al sistema
- [ ] **AFIS-03**: Audit trail de todas las operaciones (quién, cuándo, qué)
- [ ] **AFIS-04**: Pipeline de CI/CD para desarrollo seguro
- [ ] **AFIS-05**: Tests de integración reales (no mocked)
- [ ] **AFIS-06**: Frontend tests (Vitest/Playwright)
- [ ] **AFIS-07**: Producción-ready (SSL, secrets, reverse proxy)
- [ ] **AFIS-08**: Reportes y exportación de resultados forenses
- [ ] **AFIS-09**: Carga batch de huellas desde escáner AFIS
- [ ] **AFIS-10**: Dashboard de monitoreo y métricas

### Out of Scope

- **Reconocimiento Facial** — Postergado a fase posterior del roadmap
- **Reconocimiento de Iris** — Postergado a fase posterior
- **App móvil** — Web-first, interfaz forense de escritorio prioritaria
- **Cloud público** — On-premise obligatorio por naturaleza de los datos

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2025-06-12 after initialization*

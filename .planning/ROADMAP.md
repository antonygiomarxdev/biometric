# Roadmap: Biometric

**Created:** 2025-06-12 | **Updated:** 2026-06-14
**Strategy:** Entregas Verticales (Vertical Slices) — cada fase entrega valor real y usable para el perito.

---

### Fase 1: El Flujo Core Forense (MVP Vertical)

**Objetivo:** El perito puede subir una foto, el sistema la procesa con CV básico, busca en BD, y el perito compara visualmente para decidir.
**Valor:** Reemplaza la búsqueda manual en archivos físicos por una búsqueda digital instantánea.

**Plans:** 8 plans

Plans:

- [x] 01-01-PLAN.md — DB Foundation & FastAPI Core
- [x] 01-02-PLAN.md — Audit System & Seed Data
- [x] 01-03-PLAN.md — Matching Engine & Benchmark
- [x] 01-04-PLAN.md — Core CRUD Routers
- [x] 01-05-PLAN.md — Legal Report Generator (PDF)
- [x] 01-06-PLAN.md — Auth Service & Security Base
- [x] 01-07-PLAN.md — API Wiring & Monolith Teardown
- [x] 01-08-PLAN.md — Forensic Frontend (UI)

### Fase 2: IA de Visión Computacional (CONGELADA PARA EL MVP)

**Decisión de Producto:** Para acelerar el Go-To-Market del MVP, el procesamiento de imágenes usará el algoritmo tradicional de CPU. La extracción por Deep Learning queda diferida. El perito compensará los errores del algoritmo tradicional usando el editor manual (Fallback Editor) construido en esta fase.

**Plans:** 6 plans

Plans:

- [x] 02-01-PLAN.md — Enhancement Spike (U-Net vs alternative evaluation)
- [x] 02-02-PLAN.md — AI Infrastructure (ModelManager, GPU detection, config, types)
- [x] 02-03-PLAN.md — AI Enhancers (SegmentationEnhancer + EnhancementEnhancer)
- [x] 02-04-PLAN.md — DL Minutiae Extractor (AiFeatureExtractor)
- [ ] 02-05-PLAN.md — Fallback Minutiae Editor (React canvas editor)
- [x] 02-06-PLAN.md — Pipeline Integration & Benchmark (AI vs traditional on SOCOFing)

### Fase 3: Global Compliance & Security Core (Cross-Cutting)

**Objetivo:** Implementar una arquitectura de cumplimiento dinámico que garantice la privacidad de datos (PII), auditoría criptográfica y seguridad de almacenamiento adaptables a cualquier jurisdicción global (GDPR, leyes LATAM).
**Valor:** Blindaje legal del sistema ante auditorías forenses internacionales (ISO 27037). Pre-requisito crítico antes de integrar modelos generativos de IA.

**Plans:** 4 plans

Plans:

- [x] 03-01-PLAN.md — Strategy Pattern & Interfaces
- [x] 03-02-PLAN.md — Log PII Scrubber
- [x] 03-03-PLAN.md — AI Data Tokenizer (Masking)
- [x] 03-04-PLAN.md — Storage Encryption

### Fase 4: IA Generativa y Burocracia Forense (El Cerebro)

**Objetivo:** Automatizar la redacción y generación del dictamen pericial.
**Valor:** Le devuelve al perito el 50% de su tiempo, automatizando el papeleo legal.

**Plans:** 5 plans

Plans:

- [x] 03-01-PLAN.md — Core LLM Infrastructure Foundation
- [x] 03-02-PLAN.md — Text-to-SQL Engine for NLP Assistant
- [x] 03-03-PLAN.md — Structured Content Generation for Forensic Report
- [x] 03-04-PLAN.md — FastAPI Router Integration & GenAI Endpoints
- [x] 03-05-PLAN.md — Evaluation, Tracing, and CI/CD Setup

### Fase 5: Clean Architecture Strict Refactor

**Objetivo:** Eliminar la lógica de negocio y llamadas a la base de datos (`db.add`, `db.commit`) de la capa de Controladores (Routers).
**Valor:** Desacoplamiento total para facilitar el Testing y cumplir estrictamente el principio de Responsabilidad Única.

**Requirements:** [CA-01, CA-02, CA-03]

**Plans:** 7 plans

Plans:

- [x] 05-01-PLAN.md — Case & Evidence Services
- [x] 05-02-PLAN.md — Decision Service & Audit wiring
- [x] 05-03-PLAN.md — MatchingService Refactor
- [x] 05-04-PLAN.md — Audit Repository Pattern
- [x] 05-05-PLAN.md — Decision Repository
- [x] 05-06-PLAN.md — Matching Repository + Auth Tests
- [x] 05-07-PLAN.md — Service Tests (fingerprint_service + pdf_generator)

### Fase 6: Test Coverage & Quality Assurance (>90%)

**Objetivo:** Alcanzar más del 90% de code coverage en el backend mediante unit tests aislados (mockeando DB y modelos de IA pesados) para garantizar robustez a nivel empresarial.
**Valor:** Seguridad de que las refactorizaciones y nuevos modelos no rompen la lógica de negocio ni el compliance legal.

**Plans:** 4 plans

Plans:

- [ ] 06-01-PLAN.md — Coverage Infrastructure & Setup
- [x] 06-02-PLAN.md — Compliance Core Coverage
- [x] 06-03-PLAN.md — GenAI & AI Vision Coverage
- [x] 06-04-PLAN.md — Routers & Services Integration Coverage

### Fase 7: Despliegue, Infraestructura y Operación Policial

**Objetivo:** Sistema listo para producción on-premise y soporte a hardware físico.
**Valor:** Sistema autónomo, seguro y extensible a las calles.

1. **CI/CD y Pruebas E2E:** Automatización total.
2. **Contenedores de Producción:** Docker Compose optimizado para on-premise sin internet.
3. **Soporte WSQ / NFIQ 2.0:** Estándares del FBI.
4. **Integración con Scanner:** Soporte para captura en vivo (operación policial).

---
## Fases Futuras (Post-Fase 10)

### Phase 11: Hexagonal Graph Topology (Minutiae Stars)
**Goal:** As a forensic examiner, I want to match latent fingerprints using topological graph structures (Voronoi) so that matches are robust against non-linear elastic skin stretching.
**Requirements:** [PHASE11-01, PHASE11-02, PHASE11-03, PHASE11-04]
**Plans:** 4 plans
Plans:
- [ ] 11-01-PLAN.md — Spike and TopologicalGraphBuilder
- [ ] 11-02-PLAN.md — PostgreSQL JSONB Storage
- [ ] 11-03-PLAN.md — NetworkX Subgraph Matching Service
- [ ] 11-04-PLAN.md — API Endpoints & E2E Testing

- [ ] **Phase 12:** Reconocimiento Facial (Multimodal)
- [ ] **Phase 13:** Sincronización entre múltiples laboratorios regionales.

- [x] **Phase 08:** Fingerprint CPU Engine Refactor (Modular Pre/Post Hooks & Fusion)
- [x] **Phase 09 (ARCHIVADA/PIVOT):** Extracción IA con CNN. *Decisión: El modelo entrenado como post-processor fue archivado en favor del acercamiento geométrico experto (RAG).*
- [x] **Phase 10 (ACTUAL):** RAG Dactilar (Matching Geométrico Vectorial)
  - Chunking de huellas usando Triangulación de Delaunay.
  - Asignación de pesos basados en distancia al Core.
  - Reglas de validación forense estricta (>=8 enroll, >=2 search) vía Patrón Strategy.
  - Relación 1-to-N en Base de Datos usando `pgvector`.
  - Aggregation matching combinando Similitud × Peso del Chunk.
  - Blueprint arquitectónico documentado: Escalamiento a 50M+ huellas vía Embudo Coarse-to-Fine y Caching (Redis).


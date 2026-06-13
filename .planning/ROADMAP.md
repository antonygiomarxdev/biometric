# Roadmap: Biometric

**Created:** 2025-06-12 | **Updated:** 2025-06-12
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

### Fase 2: IA de Visión Computacional (El Músculo)

**Objetivo:** Reemplazar el procesamiento tradicional con Deep Learning para que el sistema funcione bien con fotos de la escena del crimen, no solo de escáner.
**Valor:** Aumenta radicalmente el "Hit Rate" en huellas de mala calidad.

**Plans:** 6 plans

Plans:

- [x] 02-01-PLAN.md — Enhancement Spike (U-Net vs alternative evaluation)
- [x] 02-02-PLAN.md — AI Infrastructure (ModelManager, GPU detection, config, types)
- [x] 02-03-PLAN.md — AI Enhancers (SegmentationEnhancer + EnhancementEnhancer)
- [x] 02-04-PLAN.md — DL Minutiae Extractor (AiFeatureExtractor)
- [ ] 02-05-PLAN.md — Fallback Minutiae Editor (React canvas editor)
- [x] 02-06-PLAN.md — Pipeline Integration & Benchmark (AI vs traditional on SOCOFing)

### Fase 3: IA Generativa y Burocracia Forense (El Cerebro)

**Objetivo:** Automatizar la redacción y generación del dictamen pericial.
**Valor:** Le devuelve al perito el 50% de su tiempo, automatizando el papeleo legal.

1. **Motor de Reportes:** Generación de PDF forense estándar.
2. **GenAI Reportes:** Integración de LLM (local o API segura) para redactar el dictamen legal a partir de los datos técnicos del match.
3. **Dashboard Analítico:** Estadísticas del laboratorio.
4. **Asistente (NLP):** Consultas en lenguaje natural sobre la base de datos (ej. "Cadena de custodia de caso X").

### Fase 4: Despliegue, Infraestructura y Operación Policial

**Objetivo:** Sistema listo para producción on-premise y soporte a hardware físico.
**Valor:** Sistema autónomo, seguro y extensible a las calles.

1. **CI/CD y Pruebas E2E:** Automatización total.
2. **Contenedores de Producción:** Docker Compose optimizado para on-premise sin internet.
3. **Soporte WSQ / NFIQ 2.0:** Estándares del FBI.
4. **Integración con Scanner:** Soporte para captura en vivo (operación policial).

---
## Fases Futuras (Post-Fase 4)

- **Fase 5:** Reconocimiento Facial (Multimodal)
- **Fase 6:** Sincronización entre múltiples laboratorios regionales.

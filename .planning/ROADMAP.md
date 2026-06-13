# Roadmap: Biometric

**Created:** 2025-06-12 | **Updated:** 2025-06-12
**Strategy:** Entregas Verticales (Vertical Slices) — cada fase entrega valor real y usable para el perito.

---

### Fase 1: El Flujo Core Forense (MVP Vertical)
**Objetivo:** El perito puede subir una foto, el sistema la procesa con CV básico, busca en BD, y el perito compara visualmente para decidir.
**Valor:** Reemplaza la búsqueda manual en archivos físicos por una búsqueda digital instantánea.

**Plans:** 8 plans

Plans:
- [ ] 01-01-PLAN.md — DB Foundation & FastAPI Core
- [ ] 01-02-PLAN.md — Audit System & Seed Data
- [ ] 01-03-PLAN.md — Matching Engine & Benchmark
- [ ] 01-04-PLAN.md — Core CRUD Routers
- [ ] 01-05-PLAN.md — Legal Report Generator (PDF)
- [ ] 01-06-PLAN.md — Auth Service & Security Base
- [ ] 01-07-PLAN.md — API Wiring & Monolith Teardown
- [ ] 01-08-PLAN.md — Forensic Frontend (UI)

### Fase 2: IA de Visión Computacional (El Músculo)
**Objetivo:** Reemplazar el procesamiento tradicional con Deep Learning para que el sistema funcione bien con fotos de la escena del crimen, no solo de escáner.
**Valor:** Aumenta radicalmente el "Hit Rate" en huellas de mala calidad.

1. **Segmentación IA (U-Net/CNN):** Recorte automático de la huella, eliminando el fondo.
2. **Enhancement (GAN):** Integración de modelo generativo para limpiar y reconstruir latentes.
3. **Extracción Deep Learning:** Reemplazo de la extracción tradicional por una red neuronal (ej. MinutiaeNet).
4. **Editor de Fallback:** Interfaz manual para que el perito edite minucias si la IA falla (5% de los casos).

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
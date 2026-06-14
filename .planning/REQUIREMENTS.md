# Requirements: Biometric

**Defined:** 2025-06-12
**Core Value:** Identificar personas por sus huellas dactilares con precisión forense, rapidez y auditabilidad

## v1 Requirements

### AI / ML Infrastructure

- [x] **AI-INFRA**: AI module structure, ModelManager, GPU detection, AiConfig, AlgorithmOrigin AI values

### AI / ML Segmentation & Enhancement

- [x] **AI-SEG**: AI fingerprint segmentation (U-Net ONNX) — SegmentationEnhancer implementing IEnhancer
- [x] **AI-ENH**: AI fingerprint enhancement (U-Net MobileNetV2 ONNX) — EnhancementEnhancer implementing IEnhancer

### Generative AI — Text-to-SQL

- [x] **GENAI-02**: Implementar motor Text-to-SQL (NLSQLTableQueryEngine) con conexión read-only y scoping explícito de tablas
- [x] **GENAI-03**: Implementar generación estructurada del Dictamen Pericial con esquema Pydantic y generación LLM vía as_structured_llm

### Test Coverage — AI & Vision Modules

- [x] **COV-02**: Alcanzar >90% de cobertura en módulos GenAI (assistant.py, llm.py, report_generator.py) y Visión Computacional (enhancer.py, extractor.py, segmentation.py, enhancement.py, extraction.py) con todos los modelos mockeados
- [x] **GENAI-04**: Exponer el asistente NLP y generador de dictámenes como endpoints REST (/api/v1/genai/assistant, /api/v1/genai/report/{caso_id})

### Matching & Algoritmo AFIS

- [ ] **AFIS-01**: Investigar y documentar algoritmo de matching óptimo (mejorar actual vs implementar NIST)
- [ ] **AFIS-02**: Implementar benchmark de precisión con dataset de huellas real (ej. SOCOFing)
- [ ] **AFIS-03**: Alcanzar tasa de identificación aceptable para uso forense

### Seguridad y Auditoría

- [ ] **AUTH-01**: Autenticación de usuarios mediante JWT
- [ ] **AUTH-02**: Roles y permisos (admin, operador, auditor)
- [ ] **AUTH-03**: API Key rotation y rate limiting
- [ ] **AUDIT-01**: Registro de auditoría de todas las operaciones
- [ ] **AUDIT-02**: Trazabilidad de cadena de custodia de imágenes
- [ ] **SEC-01**: Eliminar bucket público de MinIO
- [ ] **SEC-02**: Validación server-side de imágenes

### Global Compliance & Privacy

- [x] **COMPLIANCE-01**: Strategy pattern with IComplianceStrategy protocol, BaseStrategy, ExtremePrivacyStrategy
- [x] **COMPLIANCE-02**: Log PII scrubber with ComplianceLogFormatter, PIIFilter
- [x] **COMPLIANCE-03**: AI Data Tokenizer (DataMasker) with bidirectional text-level PII tokenization
- [ ] **COMPLIANCE-04**: Storage encryption for MinIO/PostgreSQL

### Infraestructura y CI/CD

- [ ] **INFRA-01**: Pipeline de CI (GitHub Actions + tests + lint + typecheck)
- [ ] **INFRA-02**: Docker compose ready para producción (sin reload, SSL)
- [ ] **INFRA-03**: Reverse proxy (nginx/traefik) con TLS
- [ ] **INFRA-04**: Scripts de backup y restore de la base de datos

### Testing

- [ ] **TEST-01**: Test de integración reales (no mocked) con base de datos de prueba
- [ ] **TEST-02**: Frontend testing con Vitest + Playwright
- [ ] **TEST-03**: Test de regresión visual del pipeline de procesamiento
- [ ] **TEST-04**: Benchmarks de rendimiento automatizados

### Frontend y UX Forense

- [ ] **UI-01**: Autenticación y login en frontend
- [ ] **UI-02**: Dashboard con métricas del sistema
- [ ] **UI-03**: Panel de resultados de identificación con detalle forense
- [ ] **UI-04**: Carga batch de imágenes
- [ ] **UI-05**: Reportes exportables (PDF/CSV)
- [ ] **UI-06**: Visualización de minucias superpuestas mejorada

### Refactor Técnico — Clean Architecture

- [x] **CA-01**: CaseService and EvidenceService own all DB/business logic; routers are pure HTTP controllers
- [x] **CA-02**: DecisionService owns all write operations; decisions router is an anemic HTTP controller
- [x] **CA-03**: MatchingService owns fingerprint vector persistence; known-fingerprints router is an anemic HTTP controller
- [ ] **REF-01**: Separar API en routers (rest.py está muy pesado)
- [ ] **REF-02**: Remover hardcodeo de URL en frontend (usar env vars)
- [ ] **REF-03**: Centralizar idioma a español (o inglés consistente)
- [ ] **REF-04**: Inyección de dependencias en lugar de singletons globales

## v2 Requirements (Deferred)

- **FACE-01**: Implementar proveedor de reconocimiento facial
- **FACE-02**: Pipeline de detección y encoding facial
- **IRIS-01**: Implementar proveedor de reconocimiento de iris
- **MULTI-01**: Matching multimodal (huella + facial)
- **SCALE-01**: Cola de tareas asíncrona (Celery/Redis)
- **SCALE-02**: Sharding de base de datos vectorial
- **SYNC-01**: Sincronización entre servidores on-premise y equipos forenses

## Out of Scope

| Feature | Reason |
|---------|--------|
| App móvil | Web-first, interfaz forense prioritaria |
| Cloud público | On-premise obligatorio por datos sensibles |
| Voz como biometría | Sin demanda identificada |
| Blockchain | Sin caso de uso real para el sistema |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| AI-INFRA | Phase 2 | ✅ Complete |
| AI-SEG | Phase 2 | ✅ Complete |
| AI-ENH | Phase 2 | ✅ Complete |
| GENAI-02 | Phase 3 | ✅ Complete |
| GENAI-03 | Phase 3 | ✅ Complete |
| GENAI-04 | Phase 3 | ✅ Complete |
| AFIS-01 | Phase 1 | Pending |
| AFIS-02 | Phase 1 | Pending |
| AUTH-01 | Phase 2 | Pending |
| AUTH-02 | Phase 2 | Pending |
| AUTH-03 | Phase 2 | Pending |
| AUDIT-01 | Phase 2 | Pending |
| AUDIT-02 | Phase 2 | Pending |
| SEC-01 | Phase 2 | Pending |
| SEC-02 | Phase 2 | Pending |
| COMPLIANCE-01 | Phase 3 | ✅ Complete |
| COMPLIANCE-02 | Phase 3 | ✅ Complete |
| COMPLIANCE-03 | Phase 3 | ✅ Complete |
| COMPLIANCE-04 | Phase 3 | Pending |
| INFRA-01 | Phase 3 | Pending |
| INFRA-02 | Phase 3 | Pending |
| INFRA-03 | Phase 3 | Pending |
| INFRA-04 | Phase 3 | Pending |
| TEST-01 | Phase 3 | Pending |
| TEST-02 | Phase 3 | Pending |
| TEST-03 | Phase 3 | Pending |
| TEST-04 | Phase 3 | Pending |
| UI-01 | Phase 4 | Pending |
| UI-02 | Phase 4 | Pending |
| UI-03 | Phase 4 | Pending |
| UI-04 | Phase 4 | Pending |
| UI-05 | Phase 4 | Pending |
| UI-06 | Phase 4 | Pending |
| CA-01 | Phase 5 | ✅ Complete |
| CA-02 | Phase 5 | ✅ Complete |
| CA-03 | Phase 5 | ✅ Complete |
| REF-01 | Phase 5 | Pending |
| REF-02 | Phase 5 | Pending |
| REF-03 | Phase 5 | Pending |
| REF-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 32 total
- Mapped to phases: 29
- Unmapped: 0 ✓

---
*Requirements defined: 2025-06-12*
*Last updated: 2026-06-13*

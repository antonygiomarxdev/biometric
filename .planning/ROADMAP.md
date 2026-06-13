# Roadmap: Biometric

**Created:** 2025-06-12
**Phases:** 5 | **Requirements mapped:** 27/27 ✓

---

### Phase 1: Investigación y Benchmark de Matching
**Goal:** Determinar el algoritmo de matching óptimo para uso forense
**Mode:** mvp
**Requirements:** AFIS-01, AFIS-02, AFIS-03
**Success Criteria:**
1. Documento de investigación con comparación de enfoques (pgvector actual vs NIST vs otros)
2. Benchmark cuantitativo ejecutado contra dataset SOCOFing o similar
3. Decisión documentada sobre qué algoritmo implementar
4. Reproducibilidad del benchmark (scripts + datos de prueba)

### Phase 2: Seguridad y Auditoría
**Goal:** Sistema seguro con autenticación, control de acceso y cadena de custodia
**Mode:** mvp
**Requirements:** AUTH-01, AUTH-02, AUTH-03, AUDIT-01, AUDIT-02, SEC-01, SEC-02
**Success Criteria:**
1. Usuarios pueden autenticarse con JWT con roles (admin, operador, auditor)
2. Cada operación (registro, identificación, consulta) queda registrada con timestamp + usuario
3. Imágenes de huellas protegidas (bucket no público)
4. Validación server-side de imágenes subidas
5. Frontend protegido por autenticación

### Phase 3: Infraestructura y CI/CD
**Goal:** Pipeline de desarrollo robusto con tests reales y deploy automatizado
**Mode:** mvp
**Requirements:** INFRA-01, INFRA-02, INFRA-03, INFRA-04, TEST-01, TEST-02, TEST-03, TEST-04
**Success Criteria:**
1. GitHub Actions corre tests, lint y typecheck en cada PR
2. Tests de integración con base de datos de prueba (no mocked)
3. Tests de frontend con Vitest + Playwright
4. Docker compose listo para producción (sin --reload, con healthchecks reales)
5. Reverse proxy con TLS configurado
6. Scripts de backup/restore funcionales

### Phase 4: UI Forense y Reportes
**Goal:** Interfaz completa para uso forense con reportes exportables
**Mode:** mvp
**Requirements:** UI-01, UI-02, UI-03, UI-04, UI-05, UI-06
**Success Criteria:**
1. Login UI funcional con manejo de sesión
2. Dashboard con métricas en tiempo real (procesamientos, identificaciones, errores)
3. Panel de resultados con detalle forense (score, minucias coincidentes, imagen)
4. Carga batch de múltiples imágenes
5. Reportes exportables en PDF y CSV
6. Visualización mejorada de minucias en canvas

### Phase 5: Refactor Técnico y Deuda
**Goal:** Limpiar deuda técnica acumulada para base sólida hacia fase 6+
**Mode:** mvp
**Requirements:** REF-01, REF-02, REF-03, REF-04
**Success Criteria:**
1. API dividida en routers (rest.py < 300 líneas)
2. URL del frontend configurable por env var
3. Código en idioma consistente
4. Dependencias inyectadas en lugar de singletons globales
5. Tests pasando después de refactor

---

## Fase 6+ (Post-MVP)

Las siguientes fases se definirán después de completar las 5 fases iniciales:

- **Phase 6**: Implementación de matching AFIS definitivo (según resultados de Phase 1)
- **Phase 7**: Reconocimiento facial
- **Phase 8**: Reconocimiento de iris
- **Phase 9**: Matching multimodal
- **Phase 10**: Escalabilidad (cola de tareas, sharding)
- **Phase 11**: Sincronización servidor ↔ equipos forenses

# Biometric — Sistema AFIS para Criminalística

## What This Is

**Sistema de identificación dactilar para laboratorios de criminalística.** El perito forense sube una foto de una huella levantada en escena del crimen, el sistema ayuda a segmentarla, mejorarla, extraer minucias, y buscar candidatos en base de datos. El perito revisa, ajusta y decide — el sistema no reemplaza al experto.

Secundariamente, soportará identificación con scanner para uso operativo policial.

## Core Value

Darle al perito forense una herramienta digital para identificar huellas latentes más rápido y con trazabilidad, manteniendo su criterio como autoridad final.

## Who Uses It

| Usuario | Qué hace | Sistema |
|---------|----------|---------|
| **Perito forense** | Sube foto de huella latente, el sistema procesa automáticamente y busca en BD, el perito revisa candidatos y decide | Web app en laboratorio |
| **Admin técnico** | Configura sistema, gestiona usuarios, monitorea | Web app |
| **Auditor** | Revisa cadena de custodia, reportes | Solo consulta |
| **Futuro: Operador policial** | Captura con scanner, identificación rápida | Web app + scanner |

## Real Workflow (Criminalística)

```
PRIMARIO (95% de los casos):
1. SUBIR FOTO → Foto de huella latente (JPEG, cualquier ángulo/fondo)
2. PIPELINE AUTOMÁTICO → Segmentación + Enhancement + Extracción
3. BUSCAR 1:N → Devuelve TOP-K candidatos con score
4. PERITO REVISA → Compara visualmente lado a lado
5. PERITO DECIDE → Match confirmado o descartado
6. REPORTE PDF → Con cadena de custodia

FALLBACK (5%, casos difíciles):
Si el pipeline automático no da buen resultado:
  → Perito ajusta parámetros de enhancement
  → O marca/edita minucias manualmente
  → Re-busca
  → Revisa y decide
```

## What We Have Now

- Pipeline básico de procesamiento (enhancement → extracción CN → normalización → matching vectorial)
- API REST (8 endpoints)
- Frontend React con carga de imagen y visualización básica
- PostgreSQL + pgvector + MinIO en Docker
- Sin auth, sin auditoría, sin herramientas forenses

## Estrategia de Inteligencia Artificial (Doble Motor)

Nuestra ventaja competitiva radica en democratizar tecnología de punta que los gigantes de la industria venden a precios exorbitantes, empaquetada en una herramienta accesible para laboratorios en LATAM:

1. **IA de Visión Computacional (El Músculo):**
   - *Segmentación (U-Net/CNN):* Aislar automáticamente la huella de fondos complejos (madera, papel, ruido).
   - *Enhancement (GANs):* Reconstruir crestas degradadas en huellas latentes sin inventar minucias falsas.
   - *Extracción (Deep Learning):* Detección robusta de minucias basada en redes neuronales, superior a la skeletonización tradicional.
   - *Matching:* Búsqueda vectorial 1:N ultrarrápida usando embeddings y `pgvector`.

2. **IA Generativa (El Cerebro Operativo):**
   - *Generación de Dictámenes (LLM local/seguro):* Redacción automática de borradores de informes periciales en lenguaje judicial, basados en los hallazgos técnicos (ahorra 50% del tiempo de papeleo).
   - *Asistente Forense:* Consultas en lenguaje natural (Text-to-SQL) para estadísticas y auditoría de cadena de custodia.
   - *Explicabilidad (XAI):* Traducción de métricas de matching a justificaciones comprensibles para la corte.

## What We Need to Build

| Prioridad | Funcionalidad | Para qué |
|-----------|--------------|----------|
| 🔴 **Crítica** | Pipeline automático robusto | Segmentar + mejorar + extraer de foto real (no scanner) |
| 🔴 **Crítica** | Búsqueda 1:N con candidatos | Devolver top-K para que perito revise |
| 🔴 **Crítica** | Revisión visual lado a lado | Perito compara y decide |
| 🔴 **Crítica** | Reporte forense PDF | Admisible en corte |
| 🔴 **Crítica** | Cadena de custodia | Trazabilidad de cada acción |
| 🟡 **Alta** | Carga de foto (JPEG/PNG) | Input principal: foto de cámara |
| 🟡 **Alta** | Editor manual de minucias (fallback) | Solo para casos difíciles donde lo automático no funciona |
| 🟢 **Media** | Autenticación + roles | Perito, admin, auditor |
| 🟢 **Baja** | WSQ / scanner | Para futuro modo operativo policial |
| 🟢 **Baja** | Velocidad en tiempo real | El perito espera, no es crítica |

## Tech Stack (confirmado)

| Componente | Tecnología | Estado |
|-----------|-----------|--------|
| Backend | Python 3.12+ / FastAPI | ✅ Confirmado |
| Frontend | React + TypeScript + Vite | ✅ Confirmado |
| Database | PostgreSQL + pgvector (HNSW) | ✅ Confirmado |
| Storage | MinIO (imágenes) | ✅ Confirmado |
| Queue | Redis + Celery (para async) | 📅 Futuro |
| Auth | JWT + bcrypt | 📅 Fase 2 |

## Constraints

- **On-premise:** Sin cloud, datos nunca salen del laboratorio
- **Perito decide:** El sistema asiste, no reemplaza al experto
- **Auditable:** Cada operación trazable para uso judicial
- **Idioma:** Español (usuarios en Nicaragua)

## Key Principles

1. **Tool, not replacement** — El perito es la autoridad, el sistema es su herramienta
2. **Forensic first** — Cadena de custodia, reportes, auditabilidad desde el día 1
3. **Modos de operación** — Criminalística (foto, revisión manual) + futuro policial (scanner, rápido)
4. **On-premise** — Soberanía de datos, sin dependencia externa

---
*Last updated: 2025-06-12 — visión corregida a criminalística forense*

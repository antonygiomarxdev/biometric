# Biometric — Sistema AFIS para Criminalística

> **Last updated:** 2026-06-22
> **Supersedes:** Previous revision from 2025-06-12 (which described the
> pre-Phase-29 minutiae-based architecture with MCC + Gabor).
> **Current architecture:** AFR-Net deep embedding (Phase 29). Classical
> minutiae pipeline (MCC, triplets, pairs, Gabor, thinning) was tried
> in Phases 24-27 and **deleted**; see `docs/LESSONS_LEARNED.md` §
> "Anti-Patterns Observed" for why.

## What This Is

**Sistema de identificación dactilar para laboratorios de criminalística.** El perito forense sube una foto de una huella levantada en escena del crimen, el sistema computa un embedding de 512-D con AFR-Net, busca candidatos por similitud coseno en Qdrant, y devuelve el top-K con score. El perito revisa, compara y decide — el sistema no reemplaza al experto.

Secundariamente, soportará identificación con scanner para uso operativo policial.

## Core Value

Darle al perito forense una herramienta digital para identificar huellas latentes más rápido y con trazabilidad, manteniendo su criterio como autoridad final.

## Who Uses It

| Usuario | Qué hace | Sistema |
|---------|----------|---------|
| **Perito forense** | Sube foto de huella latente, el sistema computa embedding y busca en BD, el perito revisa candidatos y decide | Web app en laboratorio |
| Admin técnico | Configura sistema, gestiona usuarios, monitorea | Web app |
| Auditor | Revisa cadena de custodia, reportes | Solo consulta |
| Futuro: Operador policial | Captura con scanner, identificación rápida | Web app + scanner |

**Current deployment scope** (2026-06-22): un perito con su caso a la vez,
en su escritorio del laboratorio. No multi-tenancy. No tiempo real.
No es para uso operativo policial todavía.

## Real Workflow (Criminalística)

```
PRIMARIO (95% de los casos):
1. SUBIR FOTO → Foto de huella latente (JPEG, cualquier ángulo/fondo)
2. EMBEDDING → AFR-Net (224×224 grayscale → 512-D vector)
3. BUSCAR 1:N → Qdrant cosine KNN, devuelve TOP-K dedos con score
4. PERITO REVISA → Compara visualmente lado a lado (original vs candidato)
5. PERITO DECIDE → Match confirmado o descartado
6. REPORTE PDF → Con cadena de custodia

FALLBACK (5%, casos difíciles):
  → El perito puede re-enrolar o ajustar parámetros U-Net
  → Re-busca
  → Revisa y decide
```

## What We Have Now (2026-06-22)

- **Matching**: AFR-Net (ConvNeXt-T + ViT-T hybrid + ArcFace, 34M
  params) → 512-D embeddings. Cosine KNN en Qdrant.
- **Enhancement**: U-Net cargado (no wired a endpoint, 29-02 pendiente).
- **Validation**: 6K SOCOFing indexados, Altered-Hard PASS
  (CR margin 0.12, Zcut margin 0.02).
- **API REST completa con 13 routers (async, tipado estricto)**.
- **Frontend React con carga de imagen y comparación lado a lado**.
- **PostgreSQL 17 + Qdrant + MinIO en Docker**.
- **Auth con Argon2id + JWT, auditoría inmutable con hash chain**.
- **Idempotencia end-to-end** (UNIQUE + ON CONFLICT + pg_advisory locks).

## What We Need to Build

| Prioridad | Funcionalidad | Para qué |
|-----------|--------------|---------|
| 🔴 **Crítica** | UX del candidato (multi-finger match) | Perito ve "1 match + N supporting" no "N candidatos iguales" |
| 🔴 **Crítica** | U-Net enhance toggle | Para latentes con ruido/fondo |
| 🔴 **Crítica** | Búsqueda 1:N con candidatos | Ya funciona, falta pulir presentación |
| 🔴 **Crítica** | Revisión visual lado a lado | Funciona con comparación original vs candidato; falta pulir UX |
| 🟡 **Alta** | NIST SD27 validación | Calibración de latentes reales |
| 🟡 **Alta** | Reporte forense PDF | Admisible en corte |
| 🟡 **Alta** | Cadena de custodia | Trazabilidad de cada acción |
| 🟢 **Media** | Autenticación + roles | Perito, admin, auditor |
| 🟢 **Baja** | WSQ / scanner | Para futuro modo operativo policial |
| 🟢 **Baja** | Modo multi-perito | Hoy es 1 perito a la vez |

## Tech Stack (2026-06-22)

| Componente | Tecnología | Estado |
|-----------|-----------|--------|
| Backend | Python 3.12+ / FastAPI | ✅ Async (psycopg3) |
| Frontend | React + TypeScript + Vite | ✅ |
| Database | PostgreSQL 17 | ✅ |
| Vectores | Qdrant | ✅ (512-D cosine) |
| Almacenamiento | MinIO | ✅ (`captures/{id}.png`) |
| Modelos | AFR-Net + U-Net (PyTorch) | ✅ Loaded, lazy |
| Auth | Argon2id + PyJWT | ✅ |

## AFR-Net Matching — Resultados (Phase 29)

- **Top-1** en Real vs Real: 100% (trivial self-match, score 1.0)
- **Top-1** en Altered-Hard CR (Central Rotation): correct person,
  score 0.54, margin 0.12 sobre #2
- **Top-1** en Altered-Hard Zcut (Cut): correct person, score 0.53,
  margin 0.02
- **512-D** descriptor por dedo (no por minucia)
- **Embedding time**: ~50ms por imagen en GPU
- **Search time**: <100ms para 6K vectors en Qdrant
- GradCAM computado en cada search (backward pass) — **no se muestra al perito** (no le fue útil en validación, ver LESSONS_LEARNED §"Phase 29: AFIS Quality Pre-Requisitos")
- Invariante a rotación, traslación y escala (training set incluye
  augmented SOCOFing)

## Constraints

- **Deployment Flexible (Local/Cloud):** El sistema debe soportar
  ejecución on-premise (datos nunca salen) usando modelos locales,
  pero la arquitectura debe ser agnóstica y soportar conexión a
  modelos remotos/cloud mediante configuración.
- **Perito decide:** El sistema asiste, no reemplaza al experto
- **Auditable:** Cada operación trazable para uso judicial
- **Idioma:** Español (usuarios en Nicaragua)
- **1 perito, 1 caso a la vez:** El sistema no es multi-tenant;
  concurrencia de enrollment sí soportada (4 workers), concurrencia
  de búsqueda también.

## Key Principles

1. **Tool, not replacement** — El perito es la autoridad, el sistema es
   su herramienta
2. **Forensic first** — Cadena de custodia, reportes, auditabilidad
   desde el día 1
3. **Modos de operación** — Criminalística (foto, revisión manual) +
   futuro policial (scanner, rápido)
4. **On-premise** — Soberanía de datos, sin dependencia externa
5. **No Legacy** — Toda la doc y código describe el sistema en
   producción. Investigación que no shipped se borra, no se comenta.
6. **Idempotency** — Re-ejecución es segura. Capturas se deduplican
   por `image_hash_sha256`. Qdrant points se upserten por
   `hash(capture_id)`.


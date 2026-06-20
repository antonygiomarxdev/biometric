# Biometric

**Sistema AFIS Forense — Identificación de huellas dactilares para criminalística**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791.svg)](https://www.postgresql.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://reactjs.org/)
[![NIST Bozorth3](https://img.shields.io/badge/NIST-Bozorth3-red.svg)](https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis)

---

## Qué hace

El perito forense sube una huella latente (levantada en escena del crimen). El sistema la procesa, extrae sus minucias, y busca en la base de datos contra huellas enroladas. Devuelve un ranking de candidatos ordenados por similitud.

**80% de acierto en primer lugar con solo 3 minucias. 100% con 15 minucias.**

---

## Cómo funciona el pipeline

### 1. Extracción de minucias

```
 Huella raw
     │
     ▼
 ┌──────────────────────────────────┐
 │  Orientation Field (Gabor)    │  ← detecta dirección local de crestas
 └──────────────────────────────────┘
     │
     ▼
 ┌──────────────────────────────────┐
 │  Filtro Gabor orientado       │  ← realza crestas, suprime ruido
 └──────────────────────────────────┘
     │
     ▼
 ┌──────────────────────────────────┐
 │  Esqueletonización            │  ← Zhang-Suen thinning (1 px de grosor)
 └──────────────────────────────────┘
     │
     ▼
 ┌──────────────────────────────────┐
 │  Crossing Number (CN)         │  ← CN=1 terminación, CN=3 bifurcación
 └──────────────────────────────────┘
     │
     ▼
 ┌──────────────────────────────────┐
 │  Filtro spurious + calidad    │  ← elimina falsas minucias
 └──────────────────────────────────┘
     │
     ▼
  Lista de minucias: [(x, y, θ), ...]
```

### 2. Enrolamiento vs Búsqueda de latente

```
 ENROLAMIENTO                        BÚSQL. LATENTE (FORENSE)
 ────────────────────              ────────────────────────
 Minucias                            Minucias de latente
     │                                  │
     ▼                                  ▼
 Pares 5-D                          Pares 5-D (probe)
 (Δx, Δy, sinΔθ, cosΔθ, d)          (Δx, Δy, sinΔθ, cosΔθ, d)
     │                                  │
     ▼                                  ▼
  Qdrant DB              KNN coseno en Qdrant
 (vectores indexados)    top-K hits por par probe
                                    │
                                    ▼
                           Bozorth3 Linker
                           ┌─────────────────────┐
                           │ dθ = angle_hit -     │
                           │      angle_probe      │
                           │                       │
                           │ Union-Find agrupa     │
                           │ pares con mismo dθ    │
                           │ (± 0.20 rad)          │
                           │                       │
                           │ n ≥ 3 AND n/total     │
                           │   ≥ 25% para validar  │
                           └─────────────────────┘
                                    │
                                    ▼
                            score = (margin + 1) / 2
                            Ranking de candidatos
```

### 3. Por qué Bozorth3 es invariante a rotación y traslación

La misma huella física puede aparecer rotada o desplazada según cómo cayó en la superficie. El Bozorth3 **no necesita pre-alinear** las imágenes porque:

```
 Huella latente (rotada 30°)    Huella enrolada

   m1 ← θ=75°                   m1' ← θ=45°
   m2 ← θ=120°                  m2' ← θ=90°

 Par (m1,m2): dθ = 120-75 = 45°    Par (m1',m2'): dθ = 90-45 = 45°
                          └───── IGUALES ─────┘

 Si N pares distribuidos en la imagen tienen el mismo dθ consistente
 → son la misma impresión, independientemente de posición o rotación.
```

Base científica: Watson et al., NISTIR 7020, 2004 (FBI/NIST).

---

## Arquitectura del sistema

```
┌──────────────┐     ┌──────────────────┐     ┌───────────┐
│  React + TS  │───►│  FastAPI (async)  │───►│ PostgreSQL │
│  (Frontend)  │     │  Clean Arch       │     │ (relacional)│
└──────────────┘     │                   │     └───────────┘
                     │  Pipeline Gabor   │     ┌───────────┐
                     │  Bozorth3 Linker  │───►│   Qdrant    │
                     │  MCC Cylinders    │     │ (vectores) │
                     └──────────────────┘     └───────────┘
```

## Stack

| Componente | Tecnología |
|-----------|-----------|
| Backend | Python 3.12 / FastAPI (async, psycopg3) |
| Frontend | React + TypeScript + Vite |
| DB | PostgreSQL 17 |
| Vectores / Búsqueda | Qdrant + Pares 5-D (Fase B: MCC 144-D) |
| Matching | Bozorth3 linker (NIST FBI, 1993) + Union-Find angular |
| Almacenamiento | MinIO |
| Auth | Argon2id + PyJWT |
| Auditoría | Hash chain inmutable |
| GenAI | LlamaIndex + Ollama / OpenAI |

## Quick Start

```bash
# 1. Dependencias
cd apps/backend
docker compose -f docker-compose.dev.yml up -d   # PostgreSQL + Qdrant + MinIO

# 2. Backend (hot reload en :8000)
uv run dev

# 3. Frontend
cd apps/frontend
pnpm install && pnpm run dev

# OpenAPI docs: http://localhost:8000/docs
```

## Estructura

```
/apps/backend    — API (FastAPI), pipeline CV, matching Bozorth3+MCC, GenAI
/apps/frontend   — UI forense (React + TypeScript)
/.planning       — Roadmap, ADRs, fases, estado del proyecto
/docs            — Documentación técnica y científica
```

## Documentación

| Documento | Contenido |
|-----------|----------|
| [`docs/FINGERPRINT_SCIENCE.md`](docs/FINGERPRINT_SCIENCE.md) | Base científica completa del pipeline — Bozorth3, MCC, TPS, referencias NIST |
| [`docs/PIPELINE_FLOW.md`](docs/PIPELINE_FLOW.md) | Diagramas Mermaid del flujo de extracción y matching |
| [`docs/MATCHING_PROCESS.md`](docs/MATCHING_PROCESS.md) | Proceso de matching paso a paso con visualizaciones |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Arquitectura modular y patrones de diseño |
| [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) | Lecciones aprendidas y decisiones técnicas |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Guía de contribución |

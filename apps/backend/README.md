# Biometric Backend

API construida en **FastAPI** con Clean Architecture, tipado estricto y Domain-Driven Design.

## Core Features

- **Pipeline de procesamiento:** Gabor enhancement → skeleton → ridge graph → extracción de minucias
- **Matching MCC:** Minutia Cylinder Code — 144D descriptor por minucia con similitud coseno
- **Búsqueda vectorial:** Qdrant con score normalizado por fingerprint (elimina bias de población)
- **80% Rank-1 con 3 minucias, 100% con 15 minucias**, búsqueda en ~216ms
- **Burocracia Generativa:** Agentes IA para dictámenes PDF forenses y Text-to-SQL
- **Auditoría inmutable:** Hash chain + enmascaramiento de PII

## Stack

| Componente | Tecnología |
|-----------|-----------|
| Framework | FastAPI + Pydantic v2 |
| DB | PostgreSQL 17 + SQLAlchemy 2.0 Async (psycopg3) |
| Vectores | Qdrant (Docker) |
| Almacenamiento | MinIO |
| Auth | Argon2id + PyJWT |
| Matching | MCC Cylinders (144D) + Cosine Similarity |
| CV | OpenCV, NumPy, Scipy |
| GenAI | LlamaIndex, Ollama / OpenAI |
| Testing | Pytest, Hypothesis |

## Quick Start

```bash
# Dependencias (Docker)
docker compose -f docker-compose.dev.yml up -d    # PostgreSQL + Qdrant + MinIO

# Backend
uv run dev                                          # Hot reload en :8000
uv run python -m pytest tests/ --ignore=tests/integration  # 677 tests

# OpenAPI docs: http://localhost:8000/docs
```

## Matching MCC — Arquitectura

Cada cylinder captura la estructura de crestas alrededor de una minucia: 12 sectores angulares × 4 anillos radiales × 3 features (orientación, conteo de crestas, espaciado). Invariante a rotación, traslación y escala. Score normalizado por fingerprint para eliminar bias estadístico.

### Cylinder MCC — Una minucia y su descriptor

![Cylinder MCC](docs/images/cylinder_explanation.png)

### Matching — Búsqueda de huella latente contra enroladas

![Matching Explanation](docs/images/matching_explanation.png)

### Precisión del matching

![Benchmark](docs/images/benchmark_accuracy.png)

## Visualizaciones

Diagramas en `docs/images/`. Para regenerar: `uv run python scripts/mcc_viz_v4.py && uv run python scripts/mcc_match_v5.py`

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/v1/persons` | Crear persona |
| GET | `/api/v1/persons` | Listar personas |
| POST | `/api/v1/persons/{id}/fingerprints` | Crear slot de huella |
| POST | `/api/v1/fingerprints/{id}/captures` | Subir imagen y procesar |
| POST | `/api/v1/matching/search` | Buscar huella latente |
| GET | `/api/v1/cases` | Listar casos |
| GET | `/api/v1/audit/logs` | Auditoría |
| POST | `/api/v1/auth/login` | Login |

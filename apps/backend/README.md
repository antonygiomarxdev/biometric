# Biometric Backend

El motor Core del sistema Biometric. API construida en **FastAPI** implementando principios estrictos de **Clean Architecture**, Tipado Estricto (Pyright) y Domain-Driven Design.

## Core Features

- **Extracción de Minucias:** Pipeline escalable (`IPipelineStep`) con pre-hooks (Orientación, Singularidades) y post-hooks (Spur Remover, Quality Filter).
- **RAG Dactilar (Matching Geométrico):** Vectorización usando Triangulación de Delaunay para resistencia a rotación y traslación. Ponderación por distancia al "Core".
- **Burocracia Generativa:** Agentes IA que escriben dictámenes PDF periciales forenses e interactúan con la BD usando Text-to-SQL.
- **Global Compliance:** Sistema de auditoría inmutable (Hash Chain) y enmascaramiento dinámico de PII.

## Stack Tecnológico

- **Framework:** FastAPI, Pydantic v2
- **Base de Datos:** SQLAlchemy 2.0, Alembic, PostgreSQL + `pgvector`
- **Computer Vision:** OpenCV, NumPy, Scipy (Delaunay)
- **Generative AI:** LlamaIndex, Ollama / OpenAI
- **Observability:** Arize Phoenix, OpenTelemetry
- **Testing:** Pytest, Hypothesis (Property-based testing)

## RAG Dactilar (Phase 10)

El sistema evolucionó de un vector monolítico a una estrategia **RAG de Chunks Geométricos**:

- **Enrollment (`/api/v1/known-fingerprints`)**: Requiere una huella de alta calidad ($\ge8$ minucias). Extrae todos los triángulos (chunks), asigna pesos usando decaimiento exponencial según su proximidad al "Core" de la huella, e inserta a la tabla 1-a-N `rag_vector_chunks`.
- **Latent Search (`/api/v1/matching/search`)**: Acepta fragmentos latentes (escena del crimen) con tan solo $\ge2$ minucias. Extrae los chunks de consulta y realiza un KNN por similitud coseno contra `pgvector`, sumando los scores ponderados por sospechoso.

### Arquitectura de Escalamiento a 5 Billones de Vectores

A nivel nacional (50M de dedos), se generarían **~5 Billones de vector chunks**. Para evitar saturar HNSW, el diseño prevé:

1. **Fase 1: Coarse Matcher (Filtro Rápido)**
   - Un solo vector global + filtros de metadata que bajan de 50,000,000 a 10,000 candidatos en milisegundos.
2. **Capa de Caching (Redis/Memcached)**
   - Previene consultas redundantes y mantiene sospechosos frecuentes en RAM.
3. **Fase 2: Fine Matcher (RAG Dactilar)**
   - Nuestro motor matemático. Busca solo en los ~1,000,000 chunks pertenecientes a los 10,000 sospechosos devueltos por la Fase 1, entregando el match forense definitivo.

## Desarrollo y Testing

```bash
# Variables de entorno
cp .env.example .env

# Ejecutar backend
python -m uvicorn src.main:app --reload --port 8000

# Testing (>90% Coverage)
pytest tests/ --cov=src

# Property-based Testing & E2E RAG
pytest tests/properties/ tests/integration/test_rag_matching_e2e.py

# Verificación de Tipos Estrictos
python -m pyright src/
```

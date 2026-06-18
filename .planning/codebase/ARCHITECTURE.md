# Architecture: BioSecure Gov

**Last updated:** 2025-06-12

## System Pattern

Clean Architecture / layered architecture with Strategy pattern for biometric providers.

```
Frontend (React/Vite)
  Scanner | Register | Identify | Results
        │ HTTP (REST)
Backend (FastAPI)
  ┌─────────────────────────────┐
  │ API Layer (src/api/)        │
  │ rest.py, cli.py             │
  └──────────┬──────────────────┘
             │
  ┌──────────▼──────────────────┐
  │ Service Layer (src/services/)│
  │ fingerprint_service.py      │
  │ comparison_service.py       │
  │ biometrics/ (Strategy)      │
  │   base.py, factory.py       │
  │   providers/ (fingerprint,  │
  │              face stub)     │
  └──────────┬──────────────────┘
             │
  ┌──────────▼──────────────────┐
  │ Processing Layer (src/      │
  │ processing/)                │
  │ enhancer.py, extractor.py   │
  │ normalization.py,           │
  │ vectorizer.py               │
  └──────────┬──────────────────┘
             │
  ┌──────────▼──────────────────┐
  │ Storage Layer (src/storage/)│
  │ database.py, repository.py  │
  │ vector_index.py,            │
  │ object_storage.py           │
  └─────────────────────────────┘
```

## Data Flow

### Fingerprint Pipeline
```
Image Upload → Decode (cv2) → Enhance (GPU/CPU) → Skeletonize →
Crossing Number Detection → Filter → Normalize →
Vector Embedding → Store/Search (Qdrant)
```

### Identification Flow
1. Client uploads fingerprint image
2. Server decodes image via OpenCV
3. Thread pool executes fingerprint_service.process_image()
4. Enhancement (CuPy GPU or multiprocess CPU)
5. Minutiae extraction via Crossing Number algorithm
6. Normalization and consensus filtering
7. Vector embedding generation
8. Qdrant L2 search for top-K candidates
9. Hybrid reranking (L2 + cosine weighted)
10. Return match result

### Registration Flow
Same pipeline as identification, plus:
- Store original image in MinIO
- Insert person record in fingerprints table
- Insert vector embedding in fingerprint_vectors table
- Store minutiae data as JSONB

## Key Design Decisions

### Strategy Pattern for Biometrics
BiometricProvider abstract base class allows adding new modalities without modifying core logic.

### Hybrid Matching
Two-phase matching:
1. Fast candidate retrieval via Qdrant IVFFlat (L2 distance)
2. In-memory reranking: score = 0.7 * L2_score + 0.3 * cosine_score

### GPU/CPU Transparency
create_enhancer() auto-detects CuPy availability. Falls back to CpuEnhancer with ProcessPoolExecutor.

### Thread Pool for CPU-bound Work
FastAPI async endpoints delegate to ThreadPoolExecutor to avoid blocking the event loop.

## Entry Points
- **REST API:** src.api.rest:app (FastAPI app)
- **CLI:** src.api.cli:cli (Click CLI)
- **Development:** src.api.rest:main (uvicorn.run)

## Configuration
Frozen dataclass Config in src.core.config. Loads from environment variables with sensible defaults.

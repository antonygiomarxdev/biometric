# Structure: BioSecure Gov

**Last updated:** 2025-06-12

## Monorepo Layout

```
biometric/
├── apps/
│   ├── backend/
│   │   ├── src/
│   │   │   ├── api/rest.py         # FastAPI app & endpoints (806 lines)
│   │   │   ├── api/cli.py          # Click CLI commands
│   │   │   ├── core/config.py      # Frozen dataclass Config
│   │   │   ├── core/types.py       # Domain types (MinutiaCandidate, MatchResult)
│   │   │   ├── core/interfaces.py  # Abstract interfaces (IEnhancer, IFeatureExtractor)
│   │   │   ├── core/metrics.py     # Performance metrics collection
│   │   │   ├── core/gpu_utils.py   # GPU detection utilities
│   │   │   ├── processing/
│   │   │   │   ├── enhancer.py     # create_enhancer() factory
│   │   │   │   ├── extractor.py    # SkeletonMinutiaeExtractor
│   │   │   │   ├── normalization.py# MinutiaNormalizer
│   │   │   │   ├── vectorizer.py   # Embedding generation
│   │   │   │   └── enhancers/      # GPU/CPU implementations
│   │   │   ├── services/
│   │   │   │   ├── fingerprint_service.py # Pipeline orchestrator
│   │   │   │   ├── comparison_service.py  # Register/identify logic
│   │   │   │   └── biometrics/     # Strategy providers
│   │   │   └── storage/
│   │   │       ├── database.py     # SQLAlchemy engine & session
│   │   │       ├── repository.py   # CRUD operations
│   │   │       ├── vector_index.py # pgvector similarity search
│   │   │       └── object_storage.py # MinIO operations
│   │   ├── tests/
│   │   │   ├── test_api_e2e.py
│   │   │   ├── test_extractor.py
│   │   │   ├── test_integration.py
│   │   │   ├── test_models.py
│   │   │   ├── test_performance.py
│   │   │   └── test_vectorizer.py
│   │   └── migrations/
│   └── frontend/
│       ├── src/
│       │   ├── App.tsx              # Main app (483 lines)
│       │   ├── components/
│       │   │   ├── fingerprint/     # Fingerprint-specific components
│       │   │   ├── face/            # Face components (stub)
│       │   │   ├── layout/          # Sidebar, MainLayout
│       │   │   └── ui/              # Button, Card, Input, Toast, etc.
│       │   ├── hooks/               # useFingerprints, useCanvasDrawer
│       │   ├── client/              # Auto-generated OpenAPI client
│       │   ├── lib/                 # utils, logger
│       │   └── types/
│       └── openapi.json
├── docs/                            # 13 markdown documentation files
├── scripts/                         # 12 utility scripts
├── docker-compose.yml
└── Makefile
```

## Naming Conventions

### Backend
- **Files:** snake_case (fingerprint_service.py)
- **Classes:** PascalCase (FingerprintService)
- **Functions:** snake_case (process_image)
- **Types:** Frozen dataclasses in types.py

### Frontend
- **Files:** PascalCase for components (FingerprintViewer.tsx)
- **Components:** PascalCase
- **Functions:** camelCase
- **Hooks:** use prefix (useFingerprints)

## Key Files by Size

### Backend (>100 lines)
- src/api/rest.py: 806 lines — largest file, 8 endpoints + startup/shutdown
- src/storage/vector_index.py: 308 lines — pgvector operations
- src/services/fingerprint_service.py: 193 lines — pipeline orchestrator

### Frontend (>100 lines)
- src/App.tsx: 483 lines — main app (identify + register UI)

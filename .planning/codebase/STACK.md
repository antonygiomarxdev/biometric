# Stack: BioSecure Gov

**Last updated:** 2025-06-12
**Scope:** Full monorepo

## Languages

| Language | Version | Location | Purpose |
|----------|---------|----------|---------|
| Python | >=3.12 | `apps/backend/` | Backend API & processing |
| TypeScript | ~5.9 | `apps/frontend/` | Frontend application |
| SQL | - | `apps/backend/migrations/` | Database migrations |

## Backend Stack

### Runtime & Server
- **Framework:** FastAPI 0.104+ — async Python web framework
- **Server:** Uvicorn 0.24+ with `[standard]` extras
- **Entry point:** `src.api.rest:app`

### Core Processing
- **NumPy** 1.24+ — array operations, image representation
- **SciPy** 1.11+ — scientific computing utilities
- **scikit-image** 0.21+ — skeletonization (`skimage.morphology.skeletonize`)
- **OpenCV** 4.8+ — image I/O, preprocessing (`cv2.imdecode`, `cv2.imread`)

### GPU Acceleration
- **CuPy** 13.x — CUDA-accelerated array operations (optional, for NVIDIA GPUs)
- **Auto-detection:** `src.core.gpu_utils.GPUConfig` checks CUDA availability
- **Fallback:** CPU multiprocessing via `ProcessPoolExecutor`

### Database
- **PostgreSQL 15** with **pgvector** 0.2.5+ extension
- **SQLAlchemy** 2.0+ ORM
- **psycopg2-binary** 2.9+ driver
- **Port:** 5434 (host) → 5432 (container)
- **Vector index:** IVFFlat on L2 distance

### Object Storage
- **MinIO** — S3-compatible storage for fingerprint images
- **Library:** `minio` 7.2+
- **Bucket:** `fingerprints`

### API & Validation
- **Pydantic** 2.4+ — request/response models
- **Pydantic-Settings** 2.0+ — configuration management
- **python-multipart** 0.0.6+ — file upload handling

### Utilities
- **click** 8.1+ — CLI interface
- **tqdm** 4.66+ — progress bars
- **matplotlib** 3.8+ — visualization (scripts only)

## Frontend Stack

### Build & Dev
- **Vite** (rolldown-vite 7.2.5) — build tool
- **TypeScript** 5.9 — type safety
- **ESLint** 9.x — linting

### UI Framework
- **React** 19.x — UI library
- **Tailwind CSS** 4.x — utility-first styling
- **PostCSS** 8.x — CSS processing
- **lucide-react** 0.562+ — icons
- **Radix UI Slot** 1.2+ — accessible primitives

### State & Utilities
- **class-variance-authority** 0.7+ — component variants
- **clsx** 2.1+ — class merging
- **tailwind-merge** 3.4+ — Tailwind class deduplication

### API Client
- **openapi-typescript-codegen** 0.30+ — generates TypeScript client from OpenAPI spec
- Custom `fetch`-based client at `src/client/`

## Infrastructure

### Monorepo
- **Turborepo** 2.3+ — workspace orchestration
- **npm** workspaces — package management
- **npm** 10.x — package manager

### Containerization
- **Docker Compose** — local development (4 services)
- **Dockerfiles** — backend (Python), frontend (Node/Vite)

## Configuration

### Backend (`.env` / env vars)
All config in `src.core.config.Config` (frozen dataclass):
- `ENV`: development/production
- `DATABASE_URL`: PostgreSQL connection string
- `VECTOR_DIMENSION`: 256 (default)
- `MATCH_THRESHOLD`: 2000.0 (L2 distance)
- `TOP_K_MATCHES`: 5
- `WEIGHT_L2` / `WEIGHT_COS`: 0.7 / 0.3
- `FORCE_CPU`: flag to disable GPU
- `LOG_LEVEL`: INFO/DEBUG

### Frontend
- `OpenAPI.BASE` hardcoded to `http://localhost:8000` in `App.tsx`

## Development Environment
- **OS:** Windows (primary) with `run.bat`, Linux support via Makefile
- **Python package manager:** `uv` (uv.lock)
- **Git hooks:** None detected
- **Pre-commit:** None detected

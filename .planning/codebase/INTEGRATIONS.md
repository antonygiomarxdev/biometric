# Integrations: BioSecure Gov

**Last updated:** 2025-06-12

## External Services

### PostgreSQL + pgvector
- **Purpose:** Primary database for fingerprint records and vector embeddings
- **Type:** Self-hosted (Docker container `pgvector/pgvector:pg15`)
- **Connection:** `postgresql://postgres:postgres@localhost:5434/fingerprint`
- **Tables:**
  - `fingerprints` — person records, minutiae data, image paths
  - `fingerprint_vectors` — vector embeddings for similarity search
- **Extensions:** `vector` (pgvector), `ivfflat` index on embedding column
- **Access pattern:** Direct SQLAlchemy ORM, no connection pooling abstraction beyond SQLAlchemy pool
- **Ports:** 5434 (host) → 5432 (container)

### MinIO (S3-compatible)
- **Purpose:** Object storage for fingerprint images
- **Type:** Self-hosted (Docker container `minio/minio:latest`)
- **Console:** `http://localhost:9001`
- **API:** `http://localhost:9000`
- **Bucket:** `fingerprints` (auto-created via `createbuckets` service)
- **Credentials:** `minioadmin` / `minioadmin`
- **Access pattern:** Direct upload/download via `minio` Python SDK
- **Security:** Bucket set to public (via `mc anonymous set public`)
- **Ports:** 9000 (API), 9001 (Console)

## Internal APIs

### REST API (`http://localhost:8000`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/extract` | POST | Extract minutiae from fingerprint image |
| `/register` | POST | Register fingerprint with person data |
| `/identify` | POST | Identify fingerprint against database |
| `/extract/diagnostic` | POST | Diagnostic extraction pipeline |
| `/metrics` | GET | System performance metrics |
| `/metrics/reset` | POST | Reset metrics |
| `/fingerprints/{person_id}/image` | GET | Get stored fingerprint image |
| `/fingerprints/{person_id}/details` | GET | Get fingerprint metadata |

### CORS Configuration
- Allowed origins: `http://localhost:5173`, `http://localhost:3000`, `http://127.0.0.1:5173`

## Service Architecture

### Backend Internal Services
- `fingerprint_service` — Orchestrates enhancement → extraction → normalization pipeline
- `comparison_service` — Registration and identification logic
- `repository` — Database CRUD operations
- `vector_index` — pgvector similarity search
- `storage` — MinIO file operations
- `metrics` — Performance metrics collection
- `db_manager` — Database connection lifecycle

## No External Third-Party APIs
The system currently has zero external SaaS/cloud dependencies. All services are self-hosted via Docker. This is notable for:
- No cloud biometric APIs (all processing is local)
- No auth providers
- No monitoring services
- No CI/CD integrations

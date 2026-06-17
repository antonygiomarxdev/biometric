# Arquitectura: Biometric — Plataforma de Identificación Biométrica

**Versión:** 1.0
**Última actualización:** 2025-06-12
**Audiencia:** Arquitectos, desarrolladores, stakeholders técnicos

---

## 1. System Overview

### Core Value
Identificar personas por sus huellas dactilares con precisión forense, rapidez y auditabilidad, operando 100% on-premise para soberanía de datos.

### Design Principles

| Principio | Aplicación |
|-----------|------------|
| **Seguridad por diseño** | Autenticación, autorización, cifrado, auditoría desde el día 1 |
| **Privacidad primero** | Datos biométricos nunca salen del control del gobierno |
| **Extensibilidad** | Estrategia de proveedores biométricos (Strategy Pattern) para añadir modalidades sin tocar el núcleo |
| **Auditabilidad forense** | Cada operación tiene cadena de custodia trazable |
| **Resiliencia** | Operación autónoma en equipos forenses con sincronización diferida |
| **Rendimiento predecible** | Tiempos de respuesta consistentes incluso con millones de registros |

---

## 2. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ON-PREMISE SERVER                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Nginx    │  │  FastAPI  │  │ Celery   │  │  PostgreSQL   │  │
│  │  (TLS)    │──│  (API)    │──│ (Worker) │  │  + Qdrant   │  │
│  │  :443     │  │  :8000    │  │          │  │  :5432        │  │
│  └──────────┘  └──────────┘  └────┬─────┘  └───────────────┘  │
│                                    │                            │
│                           ┌────────▼────────┐                   │
│                           │   Redis         │                   │
│                           │  (Queue/Cache)  │                   │
│                           └─────────────────┘                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   MinIO       │  │  Prometheus  │  │  Grafana     │          │
│  │ (Imágenes)    │  │ (Métricas)   │  │ (Dashboards) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐                                               │
│  │  Loki/Syslog │  ← Auditoría inmutable                        │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ Sincronización (VPN/HTTPS)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EQUIPOS FORENSES (MÓVILES)                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Cliente React (PWA) + Scanners de huella                │   │
│  │  Captura offline → cola local → sincroniza cuando hay red│   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Cache local de resultados + buffer de auditoría         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 On-Premise Server

- **Ubicación:** Data center del gobierno de Nicaragua
- **Red:** Aislada de internet público, accesible solo vía VPN institucional
- **Alta disponibilidad:** Active-passive con failover (para v2)
- **Backup:** PostgreSQL WAL archiving + MinIO replication

### 2.2 Equipos Forenses (Móviles)

- **Hardware:** Laptops con escáner de huellas (500+ DPI)
- **Software:** PWA React con capacidad offline
- **Conexión:** VPN a servidor central, sincronización batch
- **Operación offline:** Captura local, cola de identificaciones pendientes

### 2.3 Hybrid Mode

| Operación | Online (Server) | Offline (Equipo) |
|-----------|-----------------|-------------------|
| Registro | Sí | No (requiere server) |
| Identificación 1:N | Sí (millones) | Sí (cache local) |
| Verificación 1:1 | Sí | Sí |
| Descarga de resultados | Sí | Sí (cuando conecta) |
| Auditoría | Sí (inmediata) | Buffer local → sync |

---

## 3. System Context & Actors

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Operador    │     │  Administrador│    │  Auditor     │
│  (forense)   │     │  (técnico)   │     │  (control)   │
│              │     │              │     │              │
│  - Capturar  │     │  - Configurar│     │  - Revisar   │
│  - Identificar│    │  - Gestionar │     │    auditoría │
│  - Registrar │     │    usuarios  │     │  - Reportes  │
│  - Reportes  │     │  - Monitorear│     │  - Cadena de │
└──────┬───────┘     └──────┬───────┘     │    custodia  │
       │                    │             └──────┬───────┘
       └────────────────────┼────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   Sistema     │
                    │   Biometric   │
                    └───────────────┘
```

### Role Model

| Rol | Permisos |
|-----|----------|
| **Operador** | Capturar, identificar, registrar, ver resultados |
| **Administrador** | Todo lo de operador + gestionar usuarios, configurar sistema, ver métricas |
| **Auditor** | Solo lectura: ver auditoría, reportes, cadena de custodia. No puede operar. |

---

## 4. Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY (Nginx)                          │
│              TLS termination, rate limiting, static files           │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│                        FASTAPI APPLICATION                           │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  API Layer (routers)                                      │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐ │       │
│  │  │ Health   │ │ Auth     │ │ Finger   │ │ Admin       │ │       │
│  │  │ Router   │ │ Router   │ │ print    │ │ Router      │ │       │
│  │  │          │ │          │ │ Router   │ │             │ │       │
│  │  └──────────┘ └──────────┘ └────┬─────┘ └─────────────┘ │       │
│  └──────────────────────────────────┼───────────────────────┘       │
│                                     │                                │
│  ┌──────────────────────────────────▼───────────────────────────┐   │
│  │  Service Layer                                                │   │
│  │  ┌───────────────────┐ ┌──────────────┐ ┌────────────────┐   │   │
│  │  │ AuthService       │ │ Fingerprint  │ │ AuditService   │   │   │
│  │  │ (JWT, RBAC)       │ │ Service      │ │ (Chain of      │   │   │
│  │  │                   │ │ (Pipeline    │ │  Custody)      │   │   │
│  │  └───────────────────┘ │  Orchestr.)  │ └────────────────┘   │   │
│  │                        └──────┬───────┘                      │   │
│  │  ┌───────────────────┐ ┌──────▼───────┐ ┌────────────────┐   │   │
│  │  │ ComparisonService │ │ Biometrics   │ │ SyncService    │   │   │
│  │  │ (Matching Logic)  │ │ Factory      │ │ (Offline sync) │   │   │
│  │  └───────────────────┘ └──────────────┘ └────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                     │                                │
│  ┌──────────────────────────────────▼───────────────────────────┐   │
│  │  Processing Layer (Strategy Pattern)                          │   │
│  │  ┌────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Enhancer   │ │ Extractor    │ │Normalizer│ │Vectorizer│  │   │
│  │  │ (GPU/CPU)  │ │ (CN-based,   │ │(Consenso,│ │(Embedding)│  │   │
│  │  │            │ │  ML-based)   │ │Canónico) │ │          │  │   │
│  │  └────────────┘ └──────────────┘ └──────────┘ └──────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                     │                                │
│  ┌──────────────────────────────────▼───────────────────────────┐   │
│  │  Storage Layer                                                │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐         │   │
│  │  │ Repository   │ │ VectorIndex  │ │ ObjectStorage│         │   │
│  │  │ (Hybrid      │ │ (Qdrant    │ │ (MinIO -    │         │   │
│  │  │  Matching)   │ │  IVFFlat)    │ │  Imágenes)  │         │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 Layer Responsibilities

| Layer | Responsabilidad | Tecnología |
|-------|----------------|------------|
| **API Layer** | HTTP handling, validación, auth, rate limiting | FastAPI routers |
| **Service Layer** | Orquestación de negocio, auth, auditoría | Python services |
| **Processing Layer** | Algoritmos biométricos puros (Strategy) | NumPy/CuPy/OpenCV |
| **Storage Layer** | Persistencia, búsqueda vectorial, object storage | SQLAlchemy/Qdrant/MinIO |

### 4.2 Key Interfaces

```python
# Core abstraction for biometric algorithms
class IEnhancer(ABC):
    def enhance(self, image: np.ndarray, **kwargs) -> np.ndarray: ...

class IFeatureExtractor(ABC):
    def extract(self, image: np.ndarray) -> List[MinutiaCandidate]: ...

class INormalizer(ABC):
    def normalize(self, candidates: List[MinutiaCandidate],
                  shape: Tuple[int, int]) -> NormalizedFingerprint: ...

class IMatcher(ABC):
    def identify(self, fp: NormalizedFingerprint, 
                 top_k: int) -> MatchResult: ...

# Multimodal expansion
class BiometricProvider(ABC):
    def extract_features(self, input_data: bytes) -> BiometricVector: ...
    def compare(self, a: BiometricVector, b: BiometricVector) -> float: ...
```

---

## 5. Data Architecture

### 5.1 Entity Relationship

```
┌──────────────┐       ┌──────────────────┐       ┌──────────────┐
│   Person     │       │  Fingerprint      │       │  MatchEvent  │
│──────────────│       │──────────────────│       │──────────────│
│ id: UUID     │──1:N──│ id: UUID          │       │ id: UUID     │
│ name         │       │ person_id (FK)    │       │ probe_id (FK)│
│ document     │       │ image_path (Mino) │       │ candidate_id │
│ created_at   │       │ vector_id (FK)    │   N   │ score        │
│ created_by   │       │ num_minutiae      │───────│ confidence   │
└──────────────┘       │ quality_score     │       │ algorithm    │
                       │ minutiae_data     │       │ timestamp    │
                       │ created_at        │       │ operator_id  │
                       └────────┬──────────┘       └──────────────┘
                                │                        │
                       ┌────────▼──────────┐    ┌───────▼────────┐
                       │  VectorEmbedding   │    │  AuditLog      │
                       │──────────────────│    │───────────────│
                       │ id (PK)           │    │ id: UUID       │
                       │ embedding VECTOR  │    │ action         │
                       │ (256)             │    │ entity_type    │
                       └───────────────────┘    │ entity_id      │
                                                │ operator_id    │
                                                │ details (JSONB)│
                                                │ ip_address     │
                                                │ timestamp      │
                                                │ signature      │
                                                └────────────────┘
```

### 5.2 Storage Strategy

| Data | Storage | Retention | Backup |
|------|---------|-----------|--------|
| Fingerprint images | MinIO (S3) | Indefinido | Replication + cold archive |
| Vector embeddings | Qdrant (PostgreSQL) | Indefinido | WAL archiving |
| Person records | PostgreSQL | Indefinido | WAL archiving |
| Audit logs | PostgreSQL (partitioned) | 10 años + archive | WAL archiving |
| Match events | PostgreSQL | 5 años | WAL archiving |
| Metrics | Prometheus | 30 días | - |
| App logs | Loki/Syslog | 90 días | Cold storage |

### 5.3 Vector Index Strategy

| Aspect | Current | Target (v1) | Target (v2) |
|--------|---------|-------------|-------------|
| Index type | IVFFlat (100 lists) | IVFFlat (auto-tuned) | HNSW |
| Dimension | 256 | 256 | 256 |
| Distance | L2 | L2 + Cosine hybrid | L2 + Cosine hybrid |
| Re-indexing | Manual | Auto-rebuild on threshold | Streaming HNSW |
| Sharding | None | None | By finger/hash |

---

## 6. Security Architecture

### 6.1 Authentication & Authorization

```
                    ┌──────────────┐
                    │  Login       │
                    │  POST /auth/ │
                    │  login       │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  JWT Issued   │
                    │  (15min TTL)  │
                    │  + Refresh    │
                    │  Token (7d)   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
         │Operador│  │  Admin │  │Auditor │
         │  Role  │  │  Role  │  │  Role  │
         └────────┘  └────────┘  └────────┘
```

- **Auth mechanism:** JWT (access + refresh tokens)
- **Password hashing:** bcrypt/argon2id
- **Token storage:** HttpOnly cookies + Secure flag
- **Rate limiting:** Por endpoint y por usuario (Nginx + backend)
- **API Keys:** Para integraciones con sistemas externos (forenses)

### 6.2 Chain of Custody (Cadena de Custodia)

```
Captura → Hash SHA-256 de imagen
  ↓
Registro en BD con timestamp NTP y operador
  ↓
MinIO almacena imagen + hash (immutable object)
  ↓
Cada identificación genera MatchEvent con:
  - Probe hash, candidate hash
  - Score, algoritmo usado, versión
  - Operador que ejecutó
  - Timestamp NTP sincronizado
  - Firma digital del evento
  ↓
AuditLog: append-only, firmado, no repudio
```

### 6.3 Data Protection

| Capa | Medida |
|------|--------|
| Transport | TLS 1.3 (HTTPS obligatorio) |
| Images at rest | MinIO cifrado (bucket-level SSE) |
| DB at rest | PostgreSQL TDE / filesystem encryption |
| Secrets | HashiCorp Vault (producción) o env file cifrado |
| Network | Red aislada, VLAN separada, firewall por servicio |

### 6.4 Audit Trail

```sql
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action VARCHAR(50) NOT NULL,  -- 'register', 'identify', 'login', 'delete', etc.
    entity_type VARCHAR(50),       -- 'fingerprint', 'user', 'person'
    entity_id VARCHAR(100),
    operator_id UUID REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    details JSONB,
    previous_hash TEXT NOT NULL,   -- SHA-256 of previous row
    row_hash TEXT NOT NULL,        -- SHA-256 of (previous_hash || this row data)
    signature TEXT                  -- Digital signature (optional, for non-repudiation)
);
```

---

## 7. API Design

### 7.1 Endpoints

```
Versioning: /api/v1/

AUTH:
  POST   /auth/login              → { access_token, refresh_token }
  POST   /auth/refresh            → { access_token }
  POST   /auth/logout             → invalidate token

FINGERPRINT:
  POST   /fingerprints/extract    → Minutiae, quality score
  POST   /fingerprints/register   → Register person + fingerprint
  POST   /fingerprints/identify   → 1:N identification
  POST   /fingerprints/verify     → 1:1 verification (two images)
  GET    /fingerprints/{id}       → Details + metadata
  GET    /fingerprints/{id}/image → Raw image from MinIO
  DELETE /fingerprints/{id}       → Soft delete (audit trail)

PERSONS:
  GET    /persons/{id}            → Person details
  GET    /persons/{id}/fingerprints → All fingerprints for person
  POST   /persons                 → Create person record

ADMIN:
  GET    /admin/metrics           → System metrics
  GET    /admin/health            → Health check
  GET    /admin/users             → User management

AUDIT:
  GET    /audit/logs              → Filtered audit log
  GET    /audit/chain/{entity_id} → Chain of custody for entity

SYNC (forensic equipment):
  POST   /sync/upload             → Upload offline captures
  GET    /sync/pending            → Pending sync items
  POST   /sync/ack                → Acknowledge sync
```

### 7.2 Response Standard

```json
{
    "success": true,
    "data": { ... },
    "error": null,
    "metadata": {
        "request_id": "uuid",
        "timestamp": "2025-06-12T10:00:00Z",
        "processing_time_ms": 145,
        "operator": "user_id"
    }
}
```

---

## 8. Scalability Model

### 8.1 Current Scale (v1)
- **Registros:** 10K - 100K personas
- **Servidor único:** Sin clustering
- **Procesamiento:** Síncrono en thread pool
- **DB:** PostgreSQL single instance con Qdrant

### 8.2 Target Scale (v2)
- **Registros:** 1M - 10M personas
- **Clustering:** Active-passive con replicación
- **Cola de tareas:** Celery + Redis para procesamiento asíncrono
- **Sharding de vectores:** Por tipo de dedo o hash de persona_id
- **Caché:** Redis para resultados frecuentes

### 8.3 Bottlenecks & Mitigations

| Bottleneck | Mitigación v1 | Mitigación v2 |
|-----------|---------------|---------------|
| Vector search O(n) | IVFFlat index | HNSW index + sharding |
| Image processing CPU | GPU (CuPy) + multiprocessing | Celery workers + GPU pool |
| DB writes per second | Connection pooling | Read replicas, write master |
| Storage growth | MinIO + lifecycle policies | S3-compatible tiered storage |
| Network bandwidth | Image compression (JPEG2000/WSQ) | Progressive loading, CDN local |

### 8.4 PostgreSQL Performance Budget

| Metric | Target | Method |
|--------|--------|--------|
| Vector search (1M) | < 100ms | HNSW index |
| Vector search (10M) | < 500ms | HNSW + sharding |
| Registration throughput | 10/sec | Batch inserts, async processing |
| Identification throughput | 5/sec | Async queue (v2) |
| DB size (1M records) | ~2GB vectors + ~1GB metadata | - |

---

## 9. Technology Stack & Rationale

### 9.1 Core Stack

| Technology | Decisión | Alternativas Consideradas | Razón |
|-----------|----------|--------------------------|-------|
| **Python 3.12+** | ✅ Mantener | Go, Rust, Java | Ecosistema científico (NumPy, CuPy, OpenCV), equipo con experiencia Python |
| **FastAPI** | ✅ Mantener | Django, Flask, Starlette | Async nativo, validación Pydantic, rendimiento, OpenAPI automático |
| **PostgreSQL + Qdrant** | ✅ Mantener | Pinecone, Weaviate, Milvus | Datos biométricos no salen del gobierno; Qdrant es extensión de PG, no servicio externo |
| **MinIO** | ✅ Mantener | AWS S3, Ceph | S3-compatible, on-premise, simple |
| **React + Vite** | ✅ Mantener | Next.js, Svelte, Vue | Ecosistema grande, PWA capabilities, shadcn/ui components |
| **CuPy** | ⚠️ Evaluar | PyTorch, TensorFlow, ONNX | Ya implementado, acelera Gabor filters en GPU. Considerar migrar a PyTorch si se usan modelos ML |
| **Celery + Redis** | 📅 v2 | RabbitMQ, Kafka, ZeroMQ | Sin necesidad actual; backlog de ~1M registros antes de necesitar cola |

### 9.2 Recommended Changes

| Current | Propuesto | Cuándo | Razón |
|---------|-----------|--------|-------|
| ThreadPoolExecutor en endpoints async | ✅ Mantener pero documentar | Fase 5 | Funciona, pero mezcla async/threading |
| IVFFlat estático | ✅ Reindexación periódica | Fase 3 | Garantizar calidad de búsqueda |
| Sin autenticación | 🔐 JWT + RBAC | Fase 2 | Crítico para gobierno |
| MinIO bucket público | 🔒 Bucket privado + presigned URLs | Fase 2 | Security hotspot |
| rest.py monolítico | 📦 Routers separados | Fase 5 | Mantenibilidad |
| Face provider stub | ❌ Eliminar o implementar | Fase 4 o Fase 7 | No prometer lo que no existe |
| OpenAPI.BASE hardcoded | 🔧 Env var | Fase 5 | Portable |

### 9.3 AFIS Algorithm Decision (Pendiente)

**Estado:** En investigación (Phase 1). Timeline:

| Periodo | Matching | Razón |
|---------|----------|-------|
| **v1 (ahora)** | Qdrant L2 + Cosine | Ya implementado, funcional para desarrollo/PoC |
| **v1.5** | Híbrido: Qdrant candidatos + reranking NBIS | Precisión forense sin perder velocidad |
| **v2** | NBIS BOZORTH3 nativo (o SourceAFIS) | Estándar gubernamental, interoperabilidad |

---

## 10. Integration Architecture

### 10.1 Forensic Equipment Integration

```
┌──────────────────┐          ┌──────────────────┐
│  Equipo Forense   │         │  Server Central   │
│                   │  VPN     │                   │
│  - Scanner USB    │◄────────│  - API REST       │
│  - PWA offline    │  HTTPS   │  - Full database  │
│  - Cache local    │         │  - Audit central  │
│  - Queue offline  │────────►│                   │
│                   │  Sync    │                   │
└──────────────────┘  batch   └──────────────────┘
```

**Protocolo de sincronización:**
1. Equipo forense opera offline, captura huellas
2. Cuando conecta VPN, envía batch de operaciones
3. Server valida, procesa, responde con resultados
4. Equipo forense actualiza cache local
5. Registry de sync: cada operación tiene ID único para evitar duplicados

### 10.2 External Systems

| System | Integration | Direction | Protocol |
|--------|------------|-----------|----------|
| Civil Registry | Person lookup | Outbound | REST/API Key |
| Police DB | Wanted persons | Bidirectional | REST/API Key |
| Court systems | Report delivery | Outbound | PDF/API |
| ID card system | Biometric enrollment | Inbound | Batch import |

---

## 11. Development & DevOps

### 11.1 CI/CD Pipeline (Fase 3)

```
Push/PR → GitHub Actions →
  ├─ Lint (ruff, ESLint)
  ├─ Type check (mypy, tsc)
  ├─ Unit tests (pytest, vitest)
  ├─ Integration tests (real PG + MinIO)
  ├─ Build Docker images
  └─ Deploy to staging
```

### 11.2 Testing Strategy

| Test Type | Coverage | Tool | Fase |
|-----------|----------|------|------|
| Unit (backend) | Processing pipeline | pytest | ✅ Ahora |
| Unit (frontend) | Components | Vitest | Fase 3 |
| Integration | API + DB + MinIO | pytest + testcontainers | Fase 3 |
| E2E | Full flow | Playwright | Fase 3 |
| Benchmark | Performance, accuracy | scripts/ | Fase 1 |
| Security | Auth, XSS, injection | OWASP ZAP | Fase 2 |
| Visual regression | Pipeline output | pytest + snapshot | Fase 3 |

### 11.3 Monitoring Stack (Fase 3)

- **Metrics:** Prometheus + FastAPI metrics endpoint
- **Dashboards:** Grafana (API latency, DB queries, processing throughput)
- **Logging:** Structured JSON logs (Loki)
- **Alerting:** Alertmanager (error rate > 1%, latency > 5s)
- **Tracing:** OpenTelemetry (optional, v2)

---

## 12. Evolution Roadmap

```
Fase 1 (AHORA): Investigación Matching
  ├─ RESEARCH.md ✓
  ├─ Benchmark SOCOFing
  └─ Decisión: Qdrant vs NBIS vs híbrido

Fase 2: Seguridad y Auditoría
  ├─ JWT auth + RBAC
  ├─ Audit trail (append-only, firmado)
  ├─ Chain of custody
  ├─ MinIO bucket privado + presigned URLs
  └─ Rate limiting

Fase 3: Infraestructura y CI/CD
  ├─ GitHub Actions
  ├─ Tests reales de integración
  ├─ Frontend tests (Vitest + Playwright)
  ├─ Docker compose producción (TLS, healthchecks)
  ├─ Backup/restore scripts
  └─ Nginx reverse proxy

Fase 4: UI Forense y Reportes
  ├─ Login UI
  ├─ Dashboard
  ├─ Resultados forenses
  ├─ Carga batch
  ├─ Reportes PDF/CSV
  └─ Canvas mejorado

Fase 5: Refactor Técnico
  ├─ Splits de routers
  ├─ Env var para URL frontend
  ├─ Idioma consistente
  └─ DI en lugar de singletons

Fase 6+: AFIS Definitivo + Multimodal
  ├─ NBIS BOZORTH3 implementation
  ├─ Face recognition
  ├─ Iris recognition
  ├─ Multimodal matching
  ├─ Async processing (Celery)
  └─ Sharding + escalabilidad
```

---

## 13. Decisiones Pendientes

| Decisión | Impacto | Depende de | Propuesta |
|----------|---------|------------|-----------|
| **Algoritmo de matching** | Arquitectura núcleo | Phase 1 benchmark | Híbrido: Qdrant + NBIS reranking |
| **Formato de imagen forense** | Almacenamiento, ancho de banda | Requisitos NIST | WSQ (estándar FBI) vs JPEG2000 |
| **WSQ support** | Pipeline de procesamiento | Librería WSQ disponible | NBIS incluye compresor WSQ |
| **Firmado digital de auditoría** | Compliance | Requisitos legales | SHA-256 chain + RSA signature |
| **HSM para claves** | Seguridad | Infraestructura | Evaluar en Fase 2 |
| **Backup strategy** | Operaciones | Volumen de datos | WAL archiving + MinIO replication |

---

## 14. Glossary

| Término | Definición |
|---------|------------|
| **AFIS** | Automated Fingerprint Identification System |
| **Minutia** | Punto característico en huella (terminación o bifurcación de crestas) |
| **NBIS** | NIST Biometric Image Software (estándar US) |
| **BOZORTH3** | Matcher de NBIS, basado en minutiae |
| **MINDTCT** | Minutiae detector de NBIS |
| **Qdrant** | Extensión de PostgreSQL para búsqueda vectorial |
| **IVFFlat** | Algoritmo de indexación vectorial (Inverted File with Flat centroids) |
| **HNSW** | Hierarchical Navigable Small World (índice vectorial de alto rendimiento) |
| **WSQ** | Wavelet Scalar Quantization (formato de imagen de huella estándar FBI) |
| **Chain of Custody** | Trazabilidad completa de evidencia desde captura hasta presentación |
| **EER** | Equal Error Rate (punto donde FNMR = FMR) |
| **FNMR** | False Non-Match Rate (genuinos rechazados) |
| **FMR** | False Match Rate (impostores aceptados) |

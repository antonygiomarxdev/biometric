# Concerns: BioSecure Gov

**Last updated:** 2025-06-12

## Technical Debt

### 1. API Layer Too Heavy (src/api/rest.py — 806 lines)
- All 8 endpoints, startup/shutdown, models, CORS in one file
- Startup logic mixes DB migrations with health checks
- Should split into routers

### 2. Mixed Language in Codebase
- Comments/logs in Spanish, variable names in English
- Docs split between languages
- Risk: confusion for future contributors

### 3. Hardcoded Frontend API URL
```typescript
OpenAPI.BASE = "http://localhost:8000"; // App.tsx
```
- No env variable or build-time config
- Breaks in production

### 4. No Frontend Tests
- Zero test infrastructure for React components
- No Vitest, Playwright, or component tests

### 5. API Tests Mock Entire Service Layer
- test_api_e2e.py mocks all services
- Tests pass even if processing pipeline broken
- Low real-world confidence

### 6. Global Singleton Pattern
- Services initialized at module level
- Hard to isolate for testing
- Import-time failures

### 7. Thread Pool vs Async Mismatch
- ThreadPoolExecutor for CPU work in async endpoints
- ProcessPoolExecutor in batch processing
- Mixed concurrency models

## Security Concerns

### 1. MinIO Bucket Public
- mc anonymous set public myminio/fingerprints
- Fingerprint images publicly accessible

### 2. No Authentication
- No API keys, JWT, or session auth
- Anyone can register/identify fingerprints

### 3. No Audit Trail
- No logging of operations or timestamps
- No chain of custody

## Performance Concerns

### 1. No Request Queuing
- No backpressure under load
- Thread pool exhaustion possible

### 2. Static Vector Index
- IVFFlat index doesn't auto-rebalance
- Quality degrades as vectors added

## Architecture Concerns

### 1. Face Provider is Stub
- providers/face.py is empty
- Promises unimplemented capability

### 2. No Async Processing
- All processing synchronous in thread pool
- No Celery/Redis task queue

## Operational Concerns

### 1. No CI/CD
- No GitHub Actions
- No automated testing
- No deployment pipeline

### 2. No Production Readiness
- No SSL/TLS
- No secrets management
- No reverse proxy

### 3. No Monitoring
- Basic metrics only
- No logging aggregation
- No alerts or dashboards

## Quality Summary

| Area | Rating | Key Issue |
|------|--------|-----------|
| Test Coverage | ⚠️ Low | No frontend tests, mocked-only API |
| Security | 🔴 Critical | No auth, public MinIO, no audit |
| Performance | ⚠️ Medium | No queuing, static vector index |
| Maintainability | ⚠️ Medium | Mixed languages, global singletons |
| Production Readiness | 🔴 Low | No CI/CD, SSL, or monitoring |
| Documentation | ✅ Good | Architecture, pipeline docs exist |
| Code Quality | ✅ Good | Clean architecture, type hints |

# Discussion Log: Phase 01-flujo-core-forense

**Gathered:** 2025-06-12

## Topic: Domain Rescoping
**Discussion:** Shifted from purely technical "Matching Engine" focus to "Vertical MVP Forensic Flow" (Case → Evidence → Match → Decide → Report) to deliver actual value to peritos in Phase 1.
**Selection:** User agreed to the "Zoom Out" rescoping of the MVP. 

## Topic: Python for Backend Stack
**Discussion:** Discussed the efficiency of Python for the entire backend versus a polyglot approach (e.g., API in Go, Processing in Python).
**Selection:** Keep Python 3.12 + FastAPI for the entire backend. Python is excellent for biometric processing, and FastAPI is efficient enough for forensic lab traffic levels. Optimization will happen by fixing bad async patterns and singletons, not by changing languages.

## Topic: Backend Scalability & Data Modeling
**Discussion:** Debated UUID generation, Alembic vs SQLAlchemy auto-migrations, vector indexing, and audit logging.
**Selection:** 
- **UUIDv7** for DB IDs.
- **HNSW** Qdrant index for 10M+ scale without degradation.
- **Alembic** migrations exclusively.
- **Audit Hash Chain:** Decided against a PL/pgSQL DB trigger in favor of an Application Layer Transaction with `SELECT FOR UPDATE`. The reasoning: at 10-50 logs/second peak, row-locking in Python guarantees safety with easier testability, avoiding SQL complexity without sacrificing real-world security.

## Topic: API Architecture Refactor
**Discussion:** Splitting up `rest.py` (800+ lines). 
**Selection:** Standard 1-router-per-table pattern, with robust dependency injection via FastAPI `Depends` and `lifespan` application resources (eliminating global singletons). 

## Topic: Frontend & PDFs
**Discussion:** Minimal over-engineering required. 
**Selection:** React Router v6 + React Query + shadcn/ui. 5 basic screens centered around cases. For PDFs, `WeasyPrint` was selected due to robust HTML/CSS support and PDF/A-1b archiving standard compliance.
# ADR 005: Estrategia de Base de Datos

**Estado:** Aceptado
**Fecha:** 2025-06-12
**Contexto:** Sistema AFIS gubernamental con workload asimétrico (90%+ lecturas, escrituras batch). Necesita vectores, auditoría inmutable, y datos relacionales.

**Decisión:** PostgreSQL single con pgvector, MinIO para imágenes, Redis para cache/cola.

## Esquema

```
PostgreSQL (single + pgvector)
  ├── fingerprints (relacional: metadata, minucias JSONB)
  ├── fingerprint_vectors (HNSW index, 256d)
  ├── audit_logs (partitioned by month, hash chain append-only)
  ├── users (operadores, admins, auditores)
  ├── persons (datos de personas)
  └── match_events (historial de identificaciones)

MinIO → Imágenes originales de huellas
Redis → Cache de resultados frecuentes + cola Celery para async
```

## Estrategia de Escalabilidad

| Fase | Capacidad | Estrategia |
|------|-----------|------------|
| **v1 (ahora)** | 10K - 100K registros | PG single + HNSW. Partitioning de audit_logs por mes. |
| **v1.5** | 100K - 1M | Réplica de lectura para búsquedas (horizontal). Master maneja writes. |
| **v2** | 1M - 10M | Múltiples réplicas: una para búsquedas, otra para reportes. Failover automático. |
| **v3** | 10M+ | Sharding por hash de persona_id + réplicas por shard. |

## Racional

1. **Workload asimétrico:** AFIS es read-heavy (identificaciones continuas) vs write-light (registros batch). Las réplicas de lectura escalan horizontalmente con alta eficacia.

2. **pgvector integrado:** Relaciones directas entre fingerprints y vectores sin movimientos entre servicios. Transacciones ACID entre metadata y embedding.

3. **HNSW sobre IVFFlat:** O(log n) vs O(n). Sin degradación con inserts. Más RAM pero 10x más rápido en búsqueda.

4. **Audit partitioned:** `audit_logs` partitionada por mes previene degradación de queries de auditoría con el tiempo.

5. **MinIO separado:** Blobs grandes fuera de PG. Cache local opcional para equipos forenses.

## Consecuencias

- PG single es punto único de falla hasta tener réplicas (aceptable para v1)
- HNSW en RAM → dimensionar servidor con suficiente memoria (~2GB/100K vectores)
- Migración a réplicas requiere configurar streaming replication de PostgreSQL
- Backup con WAL archiving + pg_dump periódico

# ADR 001: Despliegue On-Premise Exclusivo

**Estado:** Aceptado
**Fecha:** 2025-06-12
**Contexto:** Sistema biométrico para gobierno de Nicaragua con datos biométricos sensibles.

**Decisión:** Toda la infraestructura corre 100% on-premise. Sin dependencia de cloud público.

**Consecuencias:**
- PostgreSQL + Qdrant reemplazan cualquier vector DB externa (Pinecone, Weaviate)
- MinIO reemplaza S3 para almacenamiento de imágenes
- Backup/restore debe ser robusto (no hay managed services)
- Capacitación del equipo de infraestructura del gobierno
- Sin downtime por dependencias externas

# ADR 002: Estrategia de Matching Híbrido (L2 + Cosine)

**Estado:** Aceptado (provisional, pendiente de Phase 1)
**Fecha:** 2025-06-12
**Contexto:** Necesitamos alta precisión forense manteniendo velocidad en búsqueda 1:N.

**Decisión:** Matching híbrido de dos fases:
1. **Fase rápida:** Qdrant IVFFlat con distancia L2 recupera top-K candidatos
2. **Reranking:** Combinación ponderada de score L2 (70%) + similitud coseno (30%)

**Racional:**
- L2 es rápido gracias a indexación IVFFlat de Qdrant
- Cosine similarity compensa diferencias de magnitud entre vectores
- Umbral L2 duro de 2000.0 evita falsos positivos groseros
- Fácil de extender: añadir tercer factor (minutiae count, quality score)

**Consecuencias:**
- Precisión depende de calidad del embedding (normalización canónica)
- Sin reranking, accuracy es menor (solo L2)
- Pesos 0.7/0.3 son empíricos, requieren validación con Phase 1

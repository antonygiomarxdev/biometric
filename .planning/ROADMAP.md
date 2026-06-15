# Roadmap: Biometric

**Created:** 2025-06-12 | **Updated:** 2026-06-15
**Strategy:** Entregas Verticales (Vertical Slices) agresivas. Arquitectura evolutiva regida por la doctrina "No Legacy" (código obsoleto se destruye).

---

## 🏆 MILESTONE: v1.0 (Completado)
*El MVP Operativo: Flujo legal, extracción matemática y Generative AI.*

- [x] **Phase 01:** Flujo Core Forense (Backend/DB/UI)
- [x] **Phase 02:** Investigación IA (Modelos, Metrics)
- [x] **Phase 03:** Global Compliance & Security Core
- [x] **Phase 04:** GenAI para Burocracia Policial
- [x] **Phase 05:** Clean Architecture & Coverage
- [x] **Phase 06:** Testing y QA (100% Core coverage)
- [x] **Phase 07:** Despliegue e Infraestructura
- [x] **Phase 08:** Fingerprint CPU Engine Refactor (Modular Pre/Post Hooks & Fusion)
- [x] **Phase 09 (ARCHIVADA/PIVOT):** Extracción IA con CNN. *Descartada por RAG Matemático.*
- [x] **Phase 10:** RAG Dactilar (Matching Geométrico Vectorial 1-to-N con `pgvector`).

---

## 🚀 MILESTONE: v2.0 Alpha (ACTUAL)
*Escalamiento a 50M+ Huellas mediante Persistencia Políglota y Topología de Grafos.*

- [ ] **Phase 11 (ACTUAL):** Topología de Grafos y "Micelio Dactilar" (NebulaGraph)
  - Extraer el esqueleto de crestas continuas de la huella en lugar de minucias aisladas (librería `sknw`).
  - Modelar la huella como un Grafo Topológico continuo (Minutiae Stars / Ridge Skeleton Graph) tolerante a estiramiento elástico.
  - Implementar persistencia políglota: Desplegar **NebulaGraph** para búsquedas de Isomorfismo de Subgrafos.
  - Destruir y purgar el motor antiguo (Phase 10 RAG Dactilar) siguiendo la doctrina "No Legacy" una vez que el motor de Grafos pruebe superioridad matemática.
- [ ] **Phase 12:** Motor de Búsqueda Híbrido (Coarse-to-Fine)
  - Filtro veloz con embeddings generados a partir de los grafos.
  - Integración de Caching en memoria (Redis/Memcached).
- [ ] **Phase 13:** Reconocimiento Facial (Multimodal)
- [ ] **Phase 14:** Sincronización P2P entre laboratorios regionales.


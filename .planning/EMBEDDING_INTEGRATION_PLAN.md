# Plan: Deep Embedding Integration (AFR-Net + U-Net) into Production

**Created:** 2026-06-22 | **Updated:** 2026-06-22
**Status:** Ready to execute
**Strategy:** Vertical slices. Fresh start — solo embedding, no MCC/Bozorth3 legacy.

---

## Decisions (confirmed in review)

| # | Decisión | Opción |
|---|----------|--------|
| 1 | **Arquitectura** | Solo embedding. Sin MCC, sin Bozorth3. Un solo endpoint. |
| 2 | **GPU/CPU** | GPU ideal, fallback a CPU. El modelo funciona en ambos. |
| 3 | **Qdrant** | Eliminar colección `ridge_graphs` existente. Nueva `fingerprint_embeddings` 512-D cosine. |
| 4 | **API response** | `{candidates: [{person_id, score, image_url, gradcam_b64}]}`. GradCAM heatmap en vez de minutiae. |
| 5 | **Imágenes** | En MinIO, no en DB. El vector Qdrant solo guarda IDs de referencia. |
| 6 | **Enrollment** | Script manual primero para SOCOFing. Endpoint automático después. |
| 7 | **Latentes reales** | Disponibles (ambos tipos: foto de escena y recortados). |
| 8 | **Galería inicial** | SOCOFing Real (6000 imágenes, 600 personas × 10 dedos). |

---

## Current State

**Backend** (`apps/backend/src/`):
- FastAPI + PostgreSQL + Qdrant + MinIO
- Clean Architecture (routers → services → repositories)
- Modelos `Person`, `Fingerprint`, `FingerprintCapture`, `Evidence`, `Case`
- Qdrant collection `ridge_graphs` con 22-D (MCC topology) — se elimina
- MCC/Bozorth3 legacy — se elimina
- `latent_search.py` router existente — se reemplaza

**Deep models** (en `.planning/spikes/06-afrnet-baseline/`):
- `best_model.pt` — AFR-Net (ConvNeXt-Tiny + ViT-Tiny + ArcFace), 34M params, 130MB
- `unet_best.pt` — Latent enhancement U-Net, 7.7M params, 30MB
- 99.70% TAR@FAR=0.01 / 98.87% TAR@FAR=0.001 en Altered-Hard

---

## Target Architecture

```
FastAPI (existing)
  │
  ├─ POST /api/v1/embedding/enroll    (crear después)
  ├─ POST /api/v1/embedding/search    (principal)
  │
  └── EmbeddingService (NEW)
        ├─ Preprocessor (pad+resize 224)
        ├─ U-Net (opcional, ?enhance=true)
        ├─ AFR-Net → 512-D embedding
        └─ GradCAM (explicabilidad)

Qdrant: fingerprint_embeddings (512-D cosine)
  payload: {person_id, capture_id, finger_id, quality}

MinIO: imágenes originales (ya existente)
```

---

## Phases

### Phase 27: Embedding Skeleton

**Goal:** `POST /api/v1/embedding/search` funcional con AFR-Net.

**Pasos:**
1. Copiar `best_model.pt` a `apps/backend/models/`
2. Cargador singleton del modelo en `src/ai/loader.py`
3. `EmbeddingService`: preprocess → embedding → Qdrant search
4. Qdrant `fingerprint_embeddings` 512-D cosine (eliminar ridge_graphs)
5. Script manual `scripts/quick_enroll.py` que procesa SOCOFing y lo mete a Qdrant
6. Router `/api/v1/embedding/search`
7. Respuesta con GradCAM heatmap del probe

**Archivos nuevos:**
- `src/ai/loader.py`
- `src/ai/unet_loader.py`
- `src/services/embedding_service.py`
- `src/db/qdrant_embedding_repository.py`
- `src/api/routers/embedding.py`
- `scripts/quick_enroll.py`

**Archivos modificados:**
- `src/main.py` — registrar router
- `src/api/dependencies.py` — DI embedding service
- `src/core/config.py` — paths de modelos
- `src/core/types.py` — tipos nuevos

**Response format:**
```json
{
  "query_time_ms": 45,
  "probe_image_url": "/api/v1/captures/{id}/image",
  "probe_gradcam_b64": "iVBOR...",
  "enhance_applied": false,
  "candidates": [
    {
      "person_id": "uuid",
      "score": 0.923,
      "full_name": "Juan Pérez",
      "image_url": "/api/v1/captures/{id}/image",
      "capture_id": "uuid",
      "finger_name": "right_index",
      "external_id": "NI-001"
    }
  ]
}
```

**Acceptance:**
- Query <200ms p50 en 6K gallery
- SOCOFing self-match ≥95% R-1
- GradCAM incluido en respuesta
- Script quick_enroll.py enrolla 6000 imágenes

**Tiempo:** 2-3 días

---

### Phase 28: U-Net Enhancement

**Goal:** `?enhance=true` aplica U-Net antes de embedding.

**Pasos:**
- Copiar `unet_best.pt` a `apps/backend/models/`
- Integrar en `EmbeddingService`
- Toggle por query param

**Tiempo:** 1 día

---

### Phase 29: Latent Robustness

**Goal:** Funciona en latentes reales (no solo SOCOFing).

| Módulo | Qué | Tiempo |
|--------|-----|--------|
| M1 | Segmentación (U-Net seg) | 3 días |
| M2 | Detección de orientación | 1 día |
| M3 | Evaluación en NIST SD27 | 1 día |
| M4 | GAN augmentation (solo si M3 falla) | 1 semana |
| M5 | Quality assessment | 1 día |
| M6 | Multi-finger fusion | 2 días |

**M1 es prioridad.** Sin segmentación, latentes con fondo ruidoso dan embeddings malos.
**M3 es la validación definitiva.** Usar los latentes reales disponibles para medir TAR real.

---

## Enrollment inicial (previo a todo)

Antes de tener endpoint, correr script manual:

```bash
python scripts/quick_enroll.py
```

Procesa SOCOFing Real (6000 imágenes):
1. Carga AFR-Net
2. Preprocesa cada imagen (pad+resize 224)
3. Extrae embedding 512-D
4. Guarda en Qdrant con payload {person_id, capture_id, finger_name}

Se corre 1 vez y listo. Después el endpoint de enrollment se construye sobre la misma lógica.

---

## Success Metrics

- **Phase 27:** API <200ms p50, self-match ≥95%
- **Phase 28:** Hard TAR@0.001 >98% (validado en spike)
- **Phase 29:** TAR en latentes reales ≥85% (ideal: >95%)
- **Producción:** 5-10× más rápido que pipeline anterior

---

## Timeline

**Week 1:**
- Lunes-Martes: Phase 27 (skeleton + enrollment batch)
- Miércoles: Phase 28 (U-Net toggle)
- Jueves-Viernes: M1 segmentación

**Week 2:**
- Lunes: M2 orientación + M5 quality
- Martes: M3 evaluar en latentes reales
- Miércoles-Viernes: Ajustes según resultados

**Total estimado:** ~10 días hábiles

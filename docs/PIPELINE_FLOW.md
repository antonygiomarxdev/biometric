# Flujo del Pipeline Biométrico

Descripción del flujo de procesamiento actual del sistema AFIS Forense.

> Última actualización: Junio 2026 (Fase A — Bozorth3 linker corregido)
> Ver base científica completa en [`FINGERPRINT_SCIENCE.md`](./FINGERPRINT_SCIENCE.md)

---

## Flujo completo: Enrolamiento

```mermaid
graph TD
    A[📷 Imagen de huella\nBMP / PNG / JPEG] --> B[Normalización\n+ escala de grises]
    B --> C[Estimación del\nOrientation Field\nof_filter.py]
    C --> D[Enhancement con\nFiltros de Gabor\ngabor.py]
    D --> E[Umbralizado\nBinarización Otsu]
    E --> F[Esqueletonización\nZhang-Suen thinning\nskeletonize_step.py]
    F --> G[Detección de Minutiae\nCrossing Number CN\ncrossing_number.py]
    G --> H[Filtro de Spurious\n+ mínimo de minutiae\nspurious_filter.py]
    H --> I[Extracción de\nPares 5-D\npair_extractor.py]
    I --> J[Vectores 5-D\nΔx, Δy, sinΔθ, cosΔθ, dist]
    J --> K[(Qdrant\nCollección de pares)]
    H --> L[(PostgreSQL\nMinutiae raw)]

    style K fill:#1565c0,color:#fff
    style L fill:#1b5e20,color:#fff
```

---

## Flujo completo: Búsqueda de huella latente

```mermaid
graph TD
    A[🔍 Huella Latente\nescena del crimen] --> B[Mismo pipeline\nde extracción]
    B --> C[Pares probe 5-D]
    C --> D[KNN en Qdrant\nsimilitud coseno\nTop-K hits por par]
    D --> E[Bozorth3 Linker\nbozorth3_linker.py]

    subgraph boz [Bozorth3 Linking — por cada candidato]
        E --> F[_compute_transform\n0, 0, dθ por par]
        F --> G[Union-Find\ncompatibilidad angular]
        G --> H[Componente más grande\nn ≥ 3 y n/total ≥ 0.25]
        H --> I[Score de margen\n margin+1 /2]
    end

    I --> J[Ranking de candidatos\northenados por score]
    J --> K[🧑‍⚖️ Perito forense\nrevisa resultado]

    style E fill:#4a148c,color:#fff
    style J fill:#1565c0,color:#fff
```

---

## Diagrama de módulos y responsabilidades

```mermaid
graph LR
    subgraph processing [apps/backend/src/processing/]
        OF[of_filter.py\nOrientation Field]
        GB[gabor.py\nEnhancement]
        SK[skeletonize_step.py\nZhang-Suen]
        CN[crossing_number.py\nDetector CN]
        SF[spurious_filter.py\nFiltro calidad]
        PE[pair_extractor.py\nDescriptor 5-D]
        MCC[mcc_descriptor.py\nDescriptor 144-D ⚠️ Fase B]
        BL[bozorth3_linker.py\nBozorth3 Linker ✅]
        TPS[tps.py\nThin Plate Spline ⚠️ Fase C]
        OFS[of_similarity.py\nOF Similarity ⚠️ Fase C]
    end

    subgraph services [src/services/]
        MS[mcc_matching_service.py]
    end

    subgraph api [src/api/routers/]
        FP[fingerprints.py]
        LS[latent_search.py]
    end

    OF --> GB --> SK --> CN --> SF --> PE --> BL
    SF --> MCC
    BL --> TPS
    MS --> processing
    FP --> MS
    LS --> MS
```

---

## Invarianza a rotación: cómo funciona el Bozorth3

```mermaid
sequenceDiagram
    participant KNN as KNN Qdrant
    participant Linker as Bozorth3 Linker
    participant UF as Union-Find

    KNN->>Linker: hits[ ] — pares candidatos agrupados por persona
    loop Por cada candidato
        Linker->>Linker: _compute_transform(probe_pair, hit)
        Note over Linker: Retorna (0, 0, dθ) — sólo ángulo
        Linker->>UF: Si |dθ1 - dθ2| ≤ 0.20 rad → union(par1, par2)
        UF-->>Linker: Componente más grande (n pares)
        Linker->>Linker: Guardia: n ≥ 3 AND n/total ≥ 0.25
        Linker->>Linker: score = (margin + 1) / 2
    end
    Linker-->>KNN: ranking de candidatos
```

---

## Comparación: descriptor actual vs Fase B

| Aspecto | Descriptor 5-D (actual) | MCC 144-D (Fase B) |
|---------|------------------------|--------------------|
| Dimensionalidad | 5 | 144 |
| Unidad | Par de minutiae | Minutia individual |
| Invarianza | Parcial (relativa al par) | Total (rotación + traslación) |
| Colisiones en KNN | Alta (geometrías comunes) | Muy baja |
| EER en FVC2006 | ~8–10% estimado | 1.82% (Cappelli 2010) |
| Estado | ⚠️ En uso, deprecar Fase B | ✅ Implementado, conectar |

---

## Estado de fases

| Fase | Descripción | Estado |
|------|-------------|--------|
| **A** | Bozorth3 linker corregido (rotation-only + component guards) | ✅ Completa |
| **B** | Migrar descriptor 5-D → MCC 144-D | 🕒 Pendiente |
| **C** | TPS post-match + OF Similarity en scoring | 🕒 Pendiente |

Ver detalles técnicos y base científica en [`FINGERPRINT_SCIENCE.md`](./FINGERPRINT_SCIENCE.md).

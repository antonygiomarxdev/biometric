# Flujo del Pipeline Biométrico (Clean Code & High Performance)

Este documento describe el flujo de procesamiento implementado en la refactorización.

## Diagrama de Flujo

```mermaid
graph TD
    A[Imagen Input] --> B{GPU Disponible?}
    
    subgraph Enhancement [1. Enhancement Strategy]
        B -- Sí --> C[GpuEnhancer]
        B -- No --> D[CpuEnhancer]
        C --> E[Pre-cálculo de Filtros]
        C --> F[Convolución Vectorizada (CuPy)]
        D --> G[Procesamiento Multiprocess (CPU)]
    end
    
    F --> H[Imagen Esqueletizada]
    G --> H
    
    subgraph Extraction [2. Feature Extraction]
        H --> I[SkeletonMinutiaeExtractor]
        I --> J[Detección de Cruces (CN)]
        J --> K[Filtrado Geométrico]
    end
    
    subgraph Normalization [3. Normalization & Consensus]
        K --> L[MinutiaNormalizer]
        L --> M[Consenso (Eliminar duplicados)]
        M --> N[Centrado y Ordenamiento Canónico]
        N --> O[NormalizedFingerprint]
    end
    
    subgraph Matching [4. Hybrid Matching]
        O --> P[VectorIndex (L2 Search)]
        P --> Q[Top-K Candidates]
        Q --> R[HybridMatcher]
        R --> S[Cálculo Coseno + L2 Ponderado]
        S --> T[MatchResult]
    end
```

## Componentes Clave

### 1. Interfaces Estrictas (`src/core/interfaces.py`)
Garantizan que cualquier implementación cumpla con el contrato (Liskov Substitution Principle).
- `IEnhancer`: Entrada `np.ndarray` -> Salida `np.ndarray`
- `IFeatureExtractor`: Entrada `np.ndarray` -> Salida `List[MinutiaCandidate]`
- `INormalizer`: Entrada `List[MinutiaCandidate]` -> Salida `NormalizedFingerprint`

### 2. Tipos Inmutables (`src/core/types.py`)
Uso de `dataclasses(frozen=True)` para evitar efectos secundarios y garantizar thread-safety.
- `MinutiaCandidate`: Coordenadas, ángulo, tipo, confianza.
- `MatchResult`: Métricas detalladas (L2, Coseno, Combinado).

### 3. Paralelismo (`src/services/fingerprint_service.py`)
- **Modo CPU**: Usa `ProcessPoolExecutor` para distribuir la carga en todos los núcleos.
- **Modo GPU**: Usa procesamiento secuencial (o batch real) para maximizar el throughput de la tarjeta gráfica sin overhead de contexto.

### 4. Matching Híbrido (`src/storage/repository.py`)
Combina la velocidad de `Qdrant` (índice IVFFlat L2) con la precisión de la distancia Coseno.
- **Fase 1**: Recuperación rápida de candidatos.
- **Fase 2**: Re-ranking en memoria usando `score = 0.7*L2_score + 0.3*Cosine_score`.

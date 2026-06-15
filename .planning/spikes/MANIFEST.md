# Spike Manifest

## Idea
Cleanup spike — analyze the biometric codebase to identify unused files, dead code, Spanish names/comments that should be English, and redundant/duplicate code. Produce an inventory for the user to review before taking action.

## Requirements
- Code must be in English (identifiers, comments, docstrings, variable names)
- File names should be in English
- Full type annotations, no `Any`/`any`
- Clean Architecture separation of concerns

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | cleanup-inventory | standard | Given the codebase, when analyzed for dead code and naming issues, then a complete inventory is produced for user review | VALIDATED | cleanup, naming, dead-code, spanish |
| 002 | llm-forensic-security | standard | Estrategias de privacidad y gateway multi-proveedor seguro | VALIDATED | security, llm |

## Spike 003: Modular Extraction Pipeline & Multi-Algorithm Fusion
El pipeline CPU actual sufre de falsos positivos masivos (>200) ante alteraciones (cicatrices, ruido). 
Este spike valida la reestructuración del motor de extracción en un modelo de capas (Pre-hooks -> Core Extractors paralelos -> Post-hooks) y la viabilidad de la fusión (ensamble) de algoritmos.

### Requirements Mapeados
- **Arquitectura Modular**: El proceso debe dividirse estrictamente en Pre-procesamiento (Imágenes/Máscaras), Extracción Cruda (Coordenadas) y Post-procesamiento (Filtrado topológico/calidad).
- **Extracción Híbrida**: El sistema debe poder ejecutar múltiples algoritmos de extracción (ej. CrossingNumber + HarrisCorner) de forma independiente y ensamblar/votar sus resultados.

## Spikes (Phase 3)

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 003a | pre-hooks-quality | standard | Given an altered print, when applying coherence mapping, then scars/cuts are successfully masked | PENDING | cv, heuristics |
| 003b | multi-algo-fusion | comparison | Given multiple raw extractors, when fusing their outputs (voting/proximity), then true minutiae survive while algorithm-specific noise is dropped | PENDING | cv, ensemble |
| 003c | post-hooks-topo | standard | Given fused points, when applying topological rules (spurs, broken ridges), then structural false positives are eliminated | PENDING | cv, heuristics |

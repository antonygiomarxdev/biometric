# Nomenclatura NIST NBIS — Mapeo al Spike

Este documento mapea la terminología del spike a la nomenclatura estándar
de NIST (NBIS — NIST Biometric Image Software). El objetivo es que la
interfaz `DetectionResult` pueda interoperar con otras herramientas del
ecosistema NIST y que la comunicación con peritos sea inequívoca.

## Referencias

- **NBIS (NIST Biometric Image Software)**: `bozorth3`, `mindtct`, `nfiq`
  código fuente en `https://github.com/usnistgov/nbis`
- **NIST Handbook of Fingerprint Examination** (US DOJ, 2011)
- **ANSI/NIST ITL 1-2011**: tipo de record tipo 14 (minucias) y
  campos opcionales como PLR (pattern level record)
- **ISO/IEC 19794-2:2011**: formato de intercambio de minucias
- **IAFIS / NGISong**: convenciones operativas del FBI

## Minucias (NIST NBIS `mindtct` / ISO 19794-2)

| Spike | NIST NBIS | ISO 19794-2 | Notas |
|-------|-----------|-------------|-------|
| `MinutiaType.TERMINATION` | `RIDGE_ENDING` (type 1) | `ridge ending` | Una cresta que termina |
| `MinutiaType.BIFURCATION` | `BIFURCATION` (type 3) | `bifurcation` | Una cresta que se divide en dos |
| `MinutiaType.UNKNOWN` | `OTHER` (no usado) | `other` | Sin correspondencia clara |

**Importante**: NBIS usa "ridge ending", no "termination". El proyecto
actual usa `TERMINATION` por claridad local. El spike mantiene la
nomenclatura del proyecto pero documenta el mapeo.

### Campos por minucia (NBIS mindtct)

| Campo | Tipo | NBIS nombre |
|-------|------|-------------|
| Posición | (x, y) | `x`, `y` |
| Ángulo | float (rad) | `theta` |
| Tipo | int (1 o 3) | `type` |
| Calidad | int (0-100) | `quality[0..2]` (coherence, continuity, clarity) |
| — | — | `appearing` (no usado en spike) |

El spike emite **un solo score de confianza** (`confidence` ∈ [0, 1]).
NBIS emite tres componentes. Esto es una simplificación consciente —
la descomposición NBIS está fuera del scope del spike.

## Singularidades

| Spike | NIST NBIS | Notas |
|-------|-----------|-------|
| `SingularityKind.CORE` | `core point` | Centro del patrón (loop/whorl) |
| `SingularityKind.DELTA` | `delta point` | Confluencia triangular |

### Detección: Poincaré + DORIC

El spike reimplementa el algoritmo de `preproc`/`classify` de NBIS:
- Suavizado Gaussiano del OF con `sigma=2.0` (estándar)
- Índice de Poincaré en malla de bloques
- DORIC validation con `radius=10` bloques, `n_samples=16`
- Threshold RMS `0.15` rad (NBIS default)

**Diferencia con NBIS**: NBIS devuelve el **único** core y delta más
fuertes por huella. El spike devuelve **todos** los validados — esto
es lo que el contexto forense de tu padre invalida (los 4 cores
"fantasma" en el mismo lugar). Por eso el NMS es necesario.

## Patrón (Henry classification)

| Spike | NBIS | Henry | Minucias esperadas |
|-------|------|-------|---------------------|
| `plain_arch` | `ARCH_PLAIN` | Plain arch | 0-0 |
| `tented_arch` | `ARCH_TENTED` | Tented arch | 0-1 |
| `loop` | `LOOP` | Loop (radial/ulnar) | 1-1 |
| `whorl` | `WHORL` | Whorl (plain/double) | 2-2 |
| `loop_missing_delta` | (incompleto) | Loop degradado | 1-0 |
| `whorl_or_loop` | (mixto) | Whorl con deltas débiles | 2-0/1 |
| `ambiguous_arch` | (ruido) | Arch con artefactos | 0-2+ |
| `over_detected` | (sobre-detecc.) | NMS insuficiente | 3+/2+ |

**Limitación actual del spike**: el patrón se **infiere del conteo
después de NMS**. NBIS usa clasificadores entrenados (mínimos
cuadrados sobre plantillas de curvatura) que son más robustos. Esto
queda como **spike separado** (Pregunta 3, fuera de scope aquí).

## NFIQ 2.0 (calidad)

NIST Fingerprint Image Quality 2.0 devuelve un score entero 0-100:
- 0-24: poor
- 25-49: fair
- 50-74: good
- 75-100: excellent

El spike **no implementa NFIQ 2.0**. La calidad por minucia
(`confidence`) es un score local, no global. Una integración con
NFIQ 2.0 sería un spike aparte.

## ENFSI / NRC verbal scale (matching)

Para el dictamen pericial, el matching se reporta en una escala verbal:

| ENFSI level | NIST LR range | Significado |
|-------------|---------------|-------------|
| Identification | LR > 10^6 | Identificación individual |
| Very strong support | 10^4 < LR ≤ 10^6 | Soporte muy fuerte |
| Strong support | 10^2 < LR ≤ 10^4 | Soporte fuerte |
| Moderate support | 10 < LR ≤ 10^2 | Soporte moderado |
| Weak support | 1 < LR ≤ 10 | Soporte débil |
| Inconclusive | LR ≈ 1 | Inconcluso |
| Exclusion | LR < 1 | Exclusión |

El spike **no toca el matching**, solo mejora la entrada. La integración
con esta escala es responsabilidad del módulo matcher (Bozorth3 pairs).

## Resumen: lo que el spike emite vs NIST

```
Spike DetectionResult            →  NIST NBIS concept
─────────────────────────────────────────────────────────
list[ValidatedMinutia]           →  minutiae record (M1)
Singularity (x,y,kind,poincare)  →  core/delta point
pattern_area_mask                →  ROI (region of interest)
quality_zones                    →  NFIQ 2.0 block scores
metadata.inferred_pattern        →  NBIS classify output
```

El `DetectionResult` es **más rico que M1** (incluye zone, ridge
trace, overlap flag) pero **menos formal** que NBIS (no usa
`quality[0..2]` ni el formato binario M1). Si en el futuro se
quiere interoperar con bozorth3, basta con reducir el `DetectionResult`
al subset M1.

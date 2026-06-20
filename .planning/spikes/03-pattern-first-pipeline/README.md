# Spike 03: Pattern-First Pipeline

**Inicio:** 2026-06-19
**Tiempo estimado:** 5–7 días
**Estado:** En progreso

## Objetivo

Reordenar la caja negra del Spike 02 para que la **clasificación de
patrón sea el primer paso**, no el último. Demostrar que con el orden
correcto (pattern → type-aware rules → minutiae) la detección de
singularidades y minucias mejora sobre el spike 02.

Este spike es la **corrección arquitectónica** identificada al cerrar
el spike 02: el orden de operaciones estaba invertido. La
clasificación del patrón (arch/loop/whorl) determina qué reglas
aplicar; no al revés.

## Motivación (del cierre del Spike 02)

El spike 02 detectó 3/3 loops incorrectamente como `loop_missing_delta`
(1-0 en lugar de 1-1). Causa raíz: el `sigma=2.0` del OF suaviza
demasiado la señal del delta en regiones de alta curvatura (loops),
pero no en regiones de baja curvatura (arches). Si el sistema supiera
**antes** de detectar singularidades que es un loop, podría usar
`sigma=1.0` solo en la zona del delta.

Esto es exactamente lo que hace NBIS: clasifica el patrón primero y
aplica reglas específicas. El spike 02 hacía lo contrario: detectaba
minucias primero e infería el patrón del conteo post-hoc.

## Preguntas a responder

1. **¿La clasificación de patrón funciona como filtro previo?**
   - Un clasificador simple (basado en curvature + singularidades)
     ¿determina arch/loop/whorl con suficiente precisión sobre
     SOCOFing Real?

2. **¿Las reglas type-aware mejoran la detección de deltas en loops?**
   - Con `sigma=1.0` específico para deltas en loops, ¿los 3 loops
     del spike 02 pasan a 1-1?

3. **¿Facing direction (radial/ulnar) es detectable?**
   - Para un loop, ¿podemos decir automáticamente si la recurva
     abre a la izquierda (radial) o a la derecha (ulnar)?

4. **¿La zona confiable type-aware mejora la calidad?**
   - Loop: zona cerca del core, en el lado de la recurva
   - Arch: zona central de crestas horizontales
   - Whorl: zona entre los dos cores
   - ¿Esto descarta minucias falsas sistemáticamente?

## Lo que el spike REUSA del Spike 02

- `types_spike.py` (contrato `DetectionResult`) — completo
- `detector_spike.py` (capa 1) — completo
- NMS, ridge tracing, overlap detection — completo
- Zone classification — completo (se extiende con type-aware variants)
- Visualizer — completo (se añade panel de facing direction)
- `NIST_NOMENCLATURE.md` — completo

## Lo que el spike AGREGA (nuevo)

### 1. Clasificador de patrón (real, no inferido)

Inputs: orientation field + singularidades post-NMS.

Algoritmo (simplificado, basado en el spike SOTA):
- Calcular curvature global del OF (media de `|∇θ|`)
- Bajo curvature → arch
- Alta curvature + 1 core + 1 delta → loop
- Alta curvature + 2 cores + 2 deltas → whorl
- Output: `PatternType` enum + confidence

### 2. Reglas type-aware

Switch sobre `PatternType`:
- `ARCH_PLAIN`: `sigma=2.0` (default), descartar cualquier core
  detectado, mantener 0 deltas
- `ARCH_TENTED`: `sigma=2.0`, mantener máximo 1 delta
- `LOOP`: `sigma=1.0` para deltas (recurrir a `sigma=2.0` solo
  para cores), cap a 1-1 esperado
- `WHORL`: `sigma=2.0`, cap a 2-2 esperado, validar que los 2
  cores son realmente distintos (no NMS fallido)

### 3. Facing direction (radial/ulnar)

Para loops solamente:
- Si es loop, el delta está en un lado y el core en el otro
- Calcular vector `delta → core`
- Si la proyección sobre el eje de la mano va hacia radial (pulgar) o
  ulnar (meñique)
- Output: `FacingDirection` enum (`RADIAL` | `ULNAR` | `UNKNOWN`)

NOTA: el lado "radial vs ulnar" depende de qué mano es (izquierda o
derecha). Sin saber la mano, solo podemos decir "slant" (hacia
izquierda o hacia la derecha). Esto es lo que vamos a hacer.

### 4. Zona confiable type-aware

En vez de una sola zona (border / interior / near_singularity), el
patrón define qué significa "interior confiable":

- `LOOP`: zona es la región dentro del bounding box core-delta,
  en el lado de la recurva
- `ARCH`: zona es la franja central horizontal del fingerprint
- `WHORL`: zona es la región entre los dos cores

## Estructura

```
03-pattern-first-pipeline/
├── README.md                # este archivo
├── types_spike.py           # extensión del contrato (PatternType, FacingDirection)
├── pattern_classifier.py    # NUEVO: clasificador real
├── facing_direction.py      # NUEVO: detección de slant
├── type_aware_rules.py      # NUEVO: reglas según patrón
├── pattern_first_detector.py # NUEVO: orquestador con orden correcto
├── visualize_spike.py       # extendido con panel de facing
├── sample_outputs/
└── REPORT.md                # al final
```

## Anti-patrones (del LESSONS_LEARNED, aplican igual)

- ❌ **No añadir thresholds sin datos.** El `sigma=1.0` para deltas en
  loops viene de la observación en el spike 02 (los deltas se perdían
  con `sigma=2.0`). El threshold está **calibrado por medición**, no
  inventado.
- ❌ **No cargar research code.** El spike termina en go/no-go.
- ❌ **No patches especulativos.** Si el orden pattern-first no mejora,
  se descarta el approach completo.
- ❌ **Dev = prod.** El spike usa la MISMA pipeline base.

## Métrica de éxito

**Mínima** (sin esto, NO-GO):
- 5/5 imágenes Real clasificadas correctamente (arch/loop/whorl)
- Los 3 loops del spike 02 (100, 102, 103) ahora reportan 1-1
- Facing direction se detecta al menos para los 3 loops

**Deseable** (si se cumple, GO):
- Singularity detection más estable (variance entre los mismos
  sujetos en distintas imágenes menor)
- Zone classification coincide con la inspección visual del perito

## Cómo correr

```bash
# Visualización con leyenda extendida
uv run --directory apps/backend python \
  .planning/spikes/03-pattern-first-pipeline/visualize_spike.py
```

## Referencias

- `../02-black-box-minutiae-detector/REPORT.md` — cierre del spike 02
- `../02-black-box-minutiae-detector/NIST_NOMENCLATURE.md` — nomenclatura
- `../02-black-box-minutiae-detector/validation_spike.py` — primitivas
  reusables (NMS, ridge trace, overlap, zone)
- `.planning/research/LATENT_AFIS_SOTA.md` — Módulo 4 (singularidades)

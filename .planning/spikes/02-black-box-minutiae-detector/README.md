# Spike 02: Black-Box Minutiae Detector con Validación Contextual

**Inicio:** 2026-06-19
**Tiempo estimado:** 5–7 días
**Estado:** En progreso

## Objetivo

Validar que la arquitectura de "caja negra" propuesta para el detector de minucias
mejora la calidad del output sin cambiar el contrato del matcher.

El sistema actual tiene 3 implementaciones de skeletonización, 2 de Crossing Number,
y 2 de filtro de minucias falsas, sin una interfaz estable. El síntoma operacional
es "a veces funciona, a veces no" — la salida es inconsistente y no se puede
distinguir una minucia confiable de una falsa solo mirando el output.

Este spike NO toca el pipeline de producción. Es una exploración aislada que
corre sobre las mismas funciones que producción y agrega una capa de
validación contextual inspirada en la práctica forense.

## Preguntas a responder

1. **¿La arquitectura de caja negra es viable?** — Un solo `MinutiaeDetector`
   con output `DetectionResult` reemplaza la maraña actual de implementaciones?

2. **¿La validación contextual reduce las falsas?** — Ridge tracing + overlap
   detection + zone classification ¿reducen visiblemente el ruido en la imagen?

3. **¿La detección de overlap (Y real vs cruce) es factible?** — Distinguir
   bifurcaciones genuinas de cruces de crestas (overlaps) usando la vecindad local.

4. **¿Mejora el matching?** — Con el matcher Bozorth3 actual, ¿Rank-1 sube en
   SOCOFing Altered-Easy cuando se usan las minucias validadas con confianza
   como peso?

**Fuera de scope de este spike** (van en un spike separado):

- Clasificación de patrón (arch/loop/whorl)
- Facing direction (radial/ulnar)
- Ridge count core-delta
- Level 3 features (poros, contornos)
- Reemplazo del matcher actual

## Criterios de éxito

### Mínimos (sin estos, NO-GO)

- Las visualizaciones muestran menos ruido que el baseline (verificación visual del perito)
- El contrato `DetectionResult` se implementa completo y se puede consumir
- La implementación corre sobre el pipeline real de producción (no es código paralelo)

### Deseables (si se cumplen, GO)

- El matcher Bozorth3 con minutiae validadas sube Rank-1 sobre SOCOFing Altered-Easy
- El output del spike tiene todos los campos del `DetectionResult` con datos reales

## Contexto forense (de la investigación y la entrevista con perito)

- **Capa 1 (detección cruda):** todo CN=1 y CN=3 del skeleton, sin filtrar.
  Estas son CANDIDATAS, no minucias.
- **Capa 2 (validación contextual):** para cada candidata, evaluar su contexto
  local antes de emitirla como confiable:
  - **Ridge tracing:** ¿la cresta tiene continuidad clara? Si se desvanece
    en pocos píxeles, es ruido o cruce, no terminación.
  - **Overlap detection:** ¿las 3 ramas del "Y" tienen continuidad, o una
    se desvanece? Si se desvanece, es cruce de crestas, no bifurcación.
  - **Zone classification:** ¿está en zona confiable (lejos del borde, lejos
    de cicatrices) o en zona de baja calidad?
  - **Singularity proximity:** minucias adyacentes al core/delta son más
    estables (anclas), pero las que están pegadas al delta pueden ser artefactos.

## Anti-patrones (del LESSONS_LEARNED)

- ❌ **No añadir thresholds sin datos.** Cada threshold se calibra contra el
  baseline (medición antes), no se inventa.
- ❌ **No cargar research code.** El spike termina en go/no-go. No queda
  "por si acaso" en el repo.
- ❌ **No patches especulativos.** Si la validación contextual no funciona
  visualmente, se descarta el approach completo.
- ❌ **Dev = prod.** El spike usa la MISMA pipeline base que producción
  (`MccMatchingService._run_quality_pipeline`).

## Estructura

```
02-black-box-minutiae-detector/
├── README.md              # este archivo
├── types_spike.py         # contrato: ValidatedMinutia, Singularity, DetectionResult
├── detector_spike.py      # capa 1: detector de candidatas
├── validation_spike.py    # capa 2: validación contextual
├── compare_spike.py       # baseline vs new (before/after)
├── visualize_spike.py     # visualizador extendido
├── sample_outputs/        # las imágenes para inspección visual
└── REPORT.md              # go / no-go / change-approach (al final)
```

## Cómo correr

```bash
# Una imagen
uv run python .planning/spikes/02-black-box-minutiae-detector/visualize_spike.py \
  apps/backend/static/SOCOFing/Real/100__M_Left_index_finger.BMP

# Comparación before/after sobre el set default
uv run python .planning/spikes/02-black-box-minutiae-detector/compare_spike.py
```

## Referencias

- `.planning/research/LATENT_AFIS_SOTA.md` — Módulos 1–5
- `docs/LESSONS_LEARNED.md` — anti-patrones a evitar
- `apps/backend/scripts/visualize_minutiae.py` — visualizador del baseline
- `apps/backend/src/services/mcc_matching_service.py` — pipeline real

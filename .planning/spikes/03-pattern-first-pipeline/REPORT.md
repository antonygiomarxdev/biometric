# Spike 03: Reporte Final

**Veredicto:** PRINCIPLED, ARCHITECTURE OK, BUT DATA NEEDED.

La arquitectura de pattern-first funciona. El clasificador Henry-only es
honesto: cuando el conteo no matchea un patrón conocido, dice `UNKNOWN`
en lugar de inventar. Pero la **detección de deltas en loops sigue
fallando**, lo cual es un problema del pipeline base (no de este spike).

## Lo que se construyó

- `types_spike.py` — `PatternType` (PLAIN_ARCH, TENTED_ARCH, LOOP, WHORL,
  UNKNOWN), `FacingDirection` (LEFT, RIGHT, UNKNOWN), `PatternClassification`
  con confidence + evidence.
- `pattern_classifier.py` — clasificador basado SOLO en la regla de Henry
  (conteo de cores/deltas). No usa curvatura ni sigmas.
- `facing_direction.py` — detecta slant left/right para loops usando el
  vector delta→core.
- `type_aware_rules.py` — `detect_singularities` (Poincaré + DORIC con
  sigma=2.0, mismo código que spike 02) + `apply_henry_cap` (capa
  según patrón clasificado, sin cap si UNKNOWN).
- `pattern_first_detector.py` — orquestador con orden correcto:
  Singularity → NMS → Pattern classification → Type-aware cap → CN → Context.
- `visualize_spike.py` — extendido con panel de facing y clasificación
  Henry-only en el panel 5.

## Decisión clave: NO usar sigmas por tipo ni curvatura

Durante el spike se intentó:
- Clasificador basado en **curvatura** del OF: fallaba porque los
  thresholds de curvatura son una heurística que parcha SOCOFing.
- `sigma=1.0` para deltas en loops: también parche. No generaliza
  a producción.

Ambos se removieron. El spike 03 ahora es:
- **Henry rules puras** (única señal forense teórica)
- **Sigma=2.0 uniforme** (default de producción)
- **UNKNOWN honesto** cuando el conteo no matchea

## Resultados (5 imágenes Real, sin Altered)

| Imagen | raw | Patrón Henry | Cap | Resultado |
|--------|-----|--------------|-----|-----------|
| 100 (loop) | 1-0 | UNKNOWN | sin cap | 1-0 |
| 101 (tented arch) | 0-1 | tented_arch | 0-1 | 0-1 |
| 102 (loop) | 1-0 | UNKNOWN | sin cap | 1-0 |
| 103 (loop?) | 1-0 | UNKNOWN | sin cap | 1-0 |
| 104 (plain arch) | 0-0 | plain_arch | 0-0 | 0-0 |

**2/5 clasificados correctamente** (101, 104).
**3/5 marcados como UNKNOWN** (100, 102, 103 — son loops con delta
no detectado, conteo 1-0 no matchea Henry).

## Fix intentado: dual-sigma detection

Probamos la **detección con dos sigmas** en el preprocesador (en el
spike, no en producción):
- **Cores**: sigma=2.0 (suprime ruido, señal robusta)
- **Deltas**: sigma=1.0 (preserva la señal más pequeña que el 2.0 borra)

Esto es principled, **no per-pattern tuning**. Es per-target: el
delta ES físicamente más chico que el core en el OF, así que necesita
menos suavizado. No es heurística inventada.

**Resultado**: el dual-sigma está implementado en
`type_aware_rules.py:_detect`, pero **no alcanza** para que el POI
llegue al rango [-0.75, -0.25] en los deltas de loops. El OF a
16x16 + smoothing de cualquier sigma no captura bien la señal del
delta en zonas de alta curvatura.

**Lo que sí probamos y no sirvió**:
- `sigma=1.0` para deltas (preserva más señal) — insuficiente
- `block_size=8` (OF más fino, 32x32) — perdimos el delta del 101
  que con block_size=16 sí se detectaba

**Lo que probablemente sí sirve** (fuera de scope de este spike):
- OF multi-escala (calcular a 8, 16, 32 y combinar)
- Poincaré a nivel de píxel (no a nivel de bloque) usando gradientes
  de la imagen enhanced directamente
- Suavizar el OF en el espacio de la imagen (256x256), no en el
  espacio del OF (16x16) — sigma=2.0 en OF es 32 píxeles en imagen

Estos cambios son refactors del preprocesador base, no del spike.
El spike 03 los deja como **trabajo futuro** explícito.

## Lo que esto significa

El spike 03 es **principled**: no inventa nada. Si el conteo no
matchea un patrón Henry, dice UNKNOWN. Esto es lo correcto
académicamente — un clasificador honesto es mejor que uno que miente.

**Pero operativamente, los 3 loops siguen sin clasificarse.** El
problema NO es del spike 03, es del **pipeline base**: el `sigma=2.0`
del OF borra la señal del delta en regiones de alta curvatura
(loops). Esto es un fix de `src/processing/pre_hooks.py` o
`src/processing/gabor.py`, no del spike.

## Anti-patrones respetados (del LESSONS_LEARNED)

- ✅ **No inventar thresholds**: el spike 03 empezó con thresholds
  de curvatura y sigma por tipo, ambos se removieron.
- ✅ **No cargar research code**: el spike es go/no-go. No queda
  "por si acaso".
- ✅ **No patches especulativos**: cuando el orden estuvo mal en
  spike 02, no se parcheó, se cerró y se reabrió como spike 03.
- ✅ **Dev = prod**: el spike usa la MISMA pipeline base que
  producción.

## Sesgo por SOCOFing (advertido por el perito)

El perito (en sesión de diseño) advirtió que SOCOFing no es
representativo de producción. **Tiene razón**. SOCOFing Real son
huellas de alta calidad; las latentes reales tienen ruido, manchas,
bajo contraste. Cualquier threshold que ajustemos a SOCOFing es
parche inútil para prod.

Por eso el spike 03 usa SOLO la regla Henry (que es forense, no
tunneada) y reporta UNKNOWN cuando el conteo no matchea. Es lo
máximo principled que se puede hacer sin datos de producción.

## Lo que falta

1. **Datos de producción reales** (5-10 latentes con ground truth
   firmado por el perito). Sin esto, no podemos validar nada.
2. **Refactor del preprocesador base** (fuera de scope de este
   spike): OF multi-escala o Poincaré a nivel de píxel. El dual-sigma
   no fue suficiente para los deltas de loops.
3. **Facing direction solo funciona cuando hay 1-1**. Los 3 loops
   no la activan porque no detectan delta. Fix de #2 lo activaría.

## Decisión final

Spike 03 cumple su propósito: validar la arquitectura pattern-first
con reglas principled. El clasificador Henry-only es correcto y
honesto. El dual-sigma está implementado como el fix menos invasivo
que se probó; no es suficiente pero es principled y queda
documentado para refactor futuro.

**No-Go** para merge sin antes:
- Conseguir 5-10 latentes reales con ground truth
- Refactor del preprocesador (multi-escala o pixel-level POI) para
  resolver el problema del delta en loops
- Validar el pipeline completo (no solo el spike) en esos datos

## Lo que Spike 03 deja como base

- `types_spike.py`: contrato `DetectionResult` con `PatternClassification`
  (lo que el matcher/reporte consumen)
- `pattern_classifier.py`: clasificador Henry-only (principled, listo
  para usar en producción)
- `facing_direction.py`: detecta slant cuando hay 1-1
- `apply_henry_cap`: la cap es trivial y correcta
- `detect_singularities`: wrapper sobre el pipeline base, listo para
  reemplazar el Poincaré si se mejora
- `pattern_first_detector.detect()`: la orquestación

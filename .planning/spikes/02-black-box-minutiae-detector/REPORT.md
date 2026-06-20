# Spike 02: Reporte Final

**Verdict:** CLOSED. Aprendimos lo que necesitábamos. La arquitectura de
caja negra es viable, pero el **orden de operaciones es incorrecto**.
La clasificación de patrón debe ir ANTES de la detección de minucias,
no después.

## Lo que se construyó

Una implementación completa de la caja negra `MinutiaeDetector` con dos
capas: detección cruda (mismo pipeline que producción) y validación
contextual (NMS, context check, ridge tracing, overlap detection,
zone classification, pattern area mask). Sin tocar el código de
producción.

Archivos:

- `types_spike.py` — contrato `DetectionResult`, `ValidatedMinutia`,
  `Singularity`, `QualityZone`, enums `Zone` y `SingularityKind`.
- `detector_spike.py` — capa 1, envuelve el pipeline real de
  `MccMatchingService._run_quality_pipeline`.
- `validation_spike.py` — capa 2, NMS + context check (Henry rules)
  + ridge tracing + overlap + zone classification + pattern area.
- `visualize_spike.py` — 6 paneles + leyenda NIST por imagen,
  comparativa before/after NMS+context, summary JSON.
- `NIST_NOMENCLATURE.md` — mapeo de tipos a NBIS / ISO 19794-2 / ENFSI.

## Lo que funcionó

1. **NMS en singularidades**: 4 cores detectados en 100/102 → 1 real.
   Implementación estándar (sort by |PI|, suppress within radius).
2. **Context check (Henry rules)**: detecta correctamente `plain_arch`
   (104: 0-0), `tented_arch` (101: 0-1). Sin ella, el sistema no
   sabe qué patrón es.
3. **Zone classification**: border / interior / near_core / near_delta
   con colores distintos en la visualización. El perito ve de un
   vistazo qué minucias están en zona confiable.
4. **Ridge tracing**: promedio de 14-17 px por minucia. Distingue
   terminaciones de cruces cortos.
5. **Overlap detection**: 1-6 cruces por imagen marcados con X roja.
6. **Pattern area mask**: identifica correctamente la región del
   fingerprint vs fondo.
7. **Nomenclatura NIST**: leyenda usa ridge ending (type 1),
   bifurcation (type 3), NBIS core/delta. Aceptable para peritos.

## Lo que NO funcionó (los hallazgos)

### 1. Orden de operaciones incorrecto

El spike detecta minucias PRIMERO, luego infiere patrón del conteo de
singularidades. **Esto está al revés**. En NBIS y ACE-V, el patrón se
determina primero, y las reglas type-aware se aplican antes de
clasificar minucias.

Síntoma: 100 y 102 son loops con 1-1 esperado, pero el sistema los
reporta como `loop_missing_delta` (1-0). El delta no se detecta porque
el sigma=2.0 del OF suaviza demasiado la señal del delta en los loops
(no así en arches, donde la OF es más uniforme).

Si el sistema supiera de antemano que es un loop, podría usar un
`sigma=1.0` para detectar mejor el delta.

### 2. Facing direction no implementada

Tu padre mencionó "a favor / en contra del reloj" (facing radial/ulnar).
Esto requiere saber:
- Si es un loop
- Dónde está el core relativo al delta
- Hacia dónde abre la recurva

No se implementó. Es feature de spike 03.

### 3. Ridge count core-delta no implementado

Feature estándar forense (cuántas crestas entre core y delta). Útil
para matching (Bozorth3) y para reporte. No se implementó. Es
feature de spike 04.

### 4. Delta detection en loops falla consistentemente

En las 3 imágenes de loop (100, 102, 103), el sistema encuentra 0
delitas. En arch (101) sí encuentra 1. La diferencia: en loops el OF
tiene alta curvatura en la zona del delta, y el sigma=2.0 lo borra.

Fix propuesto: usar `sigma=1.0` para delta en loops, `sigma=2.0` para
arch. Esto es un branch type-aware, que requiere el orden correcto
(patrón primero → luego sigma).

### 5. Patrón inferido del conteo, no clasificado

`infer_pattern_from_count()` devuelve el patrón que matchea el conteo.
Pero esto es **post-hoc**: si el conteo es erróneo (4 cores en lugar
de 1), la inferencia es errónea. La clasificación debe venir de la
geometría del OF + curvature, no del conteo.

## Métricas (5 imágenes Real)

| Imagen | raw cores | raw deltas | NMS | +context | Patrón correcto |
|--------|-----------|------------|-----|----------|-----------------|
| 100 (loop) | 4 | 0 | 1 | 1-0 | No (falta delta) |
| 101 (tented arch) | 0 | 1 | 0 | 0-1 | Sí |
| 102 (loop) | 4 | 0 | 1 | 1-0 | No (falta delta) |
| 103 (loop?) | 1 | 0 | 1 | 1-0 | No (falta delta) |
| 104 (plain arch) | 0 | 0 | 0 | 0-0 | Sí |

**2/5 aciertan en el patrón completo** (101 y 104, los arches). Los
3 loops fallan por delta faltante.

## Anti-patrones aplicados (del LESSONS_LEARNED)

- ✅ **No añadir thresholds sin datos**: todos los thresholds (NMS
  radius, PI ranges, DORIC radius) vienen de la literatura NBIS.
- ✅ **No cargar research code**: el spike termina con go/no-go claro.
- ✅ **No patches especulativos**: cuando el orden está mal, no se
  parchea, se cierra y se reabre con el orden correcto.
- ✅ **Dev = prod**: el spike usa la MISMA pipeline base que
  producción (`MccMatchingService._run_quality_pipeline`).

## Transición a Spike 03

**No-Go** para merge a producción con la arquitectura actual. La idea
de la caja negra es correcta, pero el orden debe ser:

```
Spike 02 (cerrado)
  Minutiae → Context → Pattern inferred (post-hoc)

Spike 03 (siguiente)
  Pattern classification → Type-aware rules → Minutiae → Context
```

**Lo que Spike 03 debe responder**:
1. ¿La clasificación de patrón funciona como filtro previo?
2. ¿Las reglas type-aware (sigma diferente, cap de singularidades
   según Henry) mejoran la detección de deltas en loops?
3. ¿Facing direction (radial/ulnar) es detectable automáticamente?
4. ¿La zona confiable cambia según el patrón (loop: cerca del core;
   arch: zona central; whorl: entre los dos cores)?

**Lo que Spike 03 NO debe incluir** (van en Spike 04):
- Ridge count core-delta
- Distortion compensation
- NFIQ 2.0 global quality
- Multi-touch handling

**Lo que Spike 03 debe reusar del Spike 02**:
- `types_spike.py` completo
- `detector_spike.py` completo
- Ridge tracing, overlap detection
- Zone classification
- Visualizer (extender con panel de facing)
- NIST_NOMENCLATURE.md

## Decisión final

Spike 02 cumple su propósito: validar la viabilidad de la caja negra.
Las reglas implementadas (NMS, context check, ridge trace, overlap)
son correctas y se reusan. El cambio crítico es el ORDEN: pattern-first
en lugar de minutiae-first. Eso es spike 03.

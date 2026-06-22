# Spike 04: DeepPrint Proof-of-Concept

**Inicio:** 2026-06-20
**Tiempo estimado:** 5–7 días
**Estado:** En progreso

## Objetivo

Validar si el modelo pre-entrenado **DeepPrint** (Rohwedder et al.,
BIOSIG 2023) puede reemplazar el pipeline clásico de detección de
minucias + matching Bozorth3. El spike **NO entrena un modelo** — usa
el pre-entrenado de 512-D disponible en Google Drive.

Si el resultado es positivo, justifica una fase de integración completa.
Si es negativo o marginal, descartamos DL como camino y volvemos a
mejorar el pipeline clásico.

## Motivación

Los spikes 02 y 03 validaron que:
- La arquitectura de caja negra con validación contextual funciona.
- El orden pattern-first + Henry rules + zone classification es correcto.
- El problema fundamental es la **detección de singularidades** en el
  preprocesador: con OF de 16x16 + smoothing Gaussiano, el delta en
  loops no se detecta. El dual-sigma y el block_size=8 no lo resuelven.

DeepPrint (y otros modelos de extracción de embeddings) son entrenados
sobre millones de huellas y aprenden a producir representaciones
robustas a ruido, manchas y baja calidad — exactamente el problema
que el pipeline clásico no resuelve.

## Preguntas a responder

1. **¿El modelo pre-entrenado funciona sobre SOCOFing Real?**
   - Generar embeddings 512-D para las 600 personas.
   - Verificar que el matching por coseno produce resultados
     consistentes (la misma persona tiene embeddings similares).

2. **¿El matching por coseno sobre embeddings 512-D supera a Bozorth3
   pairs en Rank-1 / Rank-5 sobre SOCOFing Altered-Easy?**
   - Baseline: 95% Rank-1 en limpio, 80% en Zcut (del LESSONS_LEARNED).
   - DeepPrint debe igualar o superar, especialmente en Altered.

3. **¿El decoder ISO 19794-2 produce minucias válidas?**
   - El repo de tim-rohwedder incluye un decoder.
   - Necesitamos verificar que las minucias decodificadas son
     utilizables por el perito para el dictamen.

4. **¿Cuál es la latencia por query?**
   - El sistema actual es ~1.5s (Bozorth3).
   - DeepPrint tiene inferencia de GPU pero el matching por coseno
     es O(N) — diferente perfil de costo.

5. **¿El modelo pre-entrenado generaliza a latentes reales?**
   - Si tenés 5-10 latentes con ground truth, podemos validar.
   - Si no, el POC termina en SOCOFing.

## Lo que el spike USA de los anteriores

- **Spike 02 + 03** validación: la detección clásica tiene un techo.
  Esto justifica el POC.
- `visualize_minutiae.py`: para comparar visualmente las minucias
  decodificadas con las del pipeline clásico.
- Estructura de validación visual (paneles, leyenda, summary).

## Lo que el spike NO hace

- ❌ **No entrena un modelo.** Solo usa el pre-entrenado de 512-D.
- ❌ **No modifica producción.** Todo en `.planning/spikes/04-...`.
- ❌ **No reemplaza el matcher.** El spike es solo evaluación; el
  reemplazo sería una fase aparte.
- ❌ **No decide arquitectura final.** Si DeepPrint funciona,
  planeamos; si no, volvemos al pipeline clásico.

## Estructura

```
04-deepprint-poc/
├── README.md                # este archivo
├── CONVENTIONS.md           # decisiones de scope (qué se prueba, qué no)
├── NBS_NOMENCLATURE.md      # mapeo de DeepPrint outputs a NBIS / ISO
├── setup_spike.py           # descarga modelo pre-entrenado, setup
├── extract_embeddings.py   # genera embeddings 512-D para SOCOFing
├── match_cosine.py          # matching por coseno, top-K
├── decode_minutiae.py       # decoder ISO 19794-2 (si funciona)
├── compare_baseline.py      # Rank-1 contra el baseline (Bozorth3)
├── visualize_spike.py       # visualización de embeddings + minutias
├── sample_outputs/
│   ├── benchmark_results.json
│   ├── per_image_embeddings/  # opcional, para inspección
│   └── visualizations/
└── REPORT.md                # al final: go / no-go / next steps
```

## Datos

- **SOCOFing Real** (600 personas) — para validar que el modelo
  pre-entrenado funciona y Rank-1 comparable a baseline.
- **SOCOFing Altered-Easy** (CR, Obl, Zcut) — para validar en
  condiciones similares a producción.
- **Latentes del perito** (5-10, con ground truth) — si están
  disponibles. Validación final.

## Métrica de éxito

**Mínima** (sin esto, NO-GO):
- El modelo pre-entrenado carga y produce embeddings.
- El matching por coseno sobre embeddings tiene Rank-1 >= 80% sobre
  SOCOFing Real (igualar al baseline).
- Las minucias decodificadas son visualmente razonables (no ruido).

**Deseable** (si se cumple, GO):
- Rank-1 >= 95% sobre Altered-Easy (vs 80% baseline).
- Rank-1 >= 80% sobre latentes reales del perito.
- Latencia de inferencia aceptable (<500 ms por query con GPU).

## Anti-patrones (del LESSONS_LEARNED)

- ❌ **No comparar sin baseline.** El spike debe correr también el
  pipeline actual sobre los mismos datos y comparar Rank-1 / Rank-5.
- ❌ **No patches sin datos.** Si el modelo pre-entrenado no
  generaliza, NO ajustamos — volvemos al pipeline clásico.
- ❌ **Dev = prod.** Si el POC valida, la fase de integración debe
  usar el MISMO modelo (mismo .pth, mismas dimensiones).

## Cómo correr

```bash
# 1. Setup: clonar el repo upstream y descargar el modelo
uv run python .planning/spikes/04-deepprint-poc/setup_spike.py

# 2. Generar embeddings para SOCOFing
uv run python .planning/spikes/04-deepprint-poc/extract_embeddings.py

# 3. Matchear por coseno y comparar con baseline
uv run python .planning/spikes/04-deepprint-poc/compare_baseline.py

# 4. Visualizar
uv run python .planning/spikes/04-deepprint-poc/visualize_spike.py
```

## Referencias

- **Repo upstream**: https://github.com/tim-rohwedder/fixed-length-fingerprint-extractors
- **Paper**: Rohwedder et al., "Benchmarking fixed-length Fingerprint
  Representations across different Embedding Sizes and Sensor Types",
  BIOSIG 2023. https://arxiv.org/abs/2307.08615
- **DeepPrint original**: Joshi et al., "DeepPrint: A Probabilistic
  Model for Minutiae Extraction", ICB 2018.
- `docs/LESSONS_LEARNED.md` — anti-patrones a evitar
- `../02-black-box-minutiae-detector/REPORT.md` — qué validó el spike 02
- `../03-pattern-first-pipeline/REPORT.md` — qué validó el spike 03

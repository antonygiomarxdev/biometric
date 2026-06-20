# Proceso de Matching — Paso a Paso

Visión detallada de cómo el sistema identifica una huella latente contra la base de datos.

> Base científica completa en [`FINGERPRINT_SCIENCE.md`](./FINGERPRINT_SCIENCE.md)
> Diagramas de flujo generales en [`PIPELINE_FLOW.md`](./PIPELINE_FLOW.md)

---

## Etapa 1 — Qué es una minucia

Una **minucia** es un punto singular en el patrón de crestas de la piel: donde una cresta termina o se bifurca. Son los puntos que el sistema extrae y compara.

```
 Imagen de huella (sección)

  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~   ← terminación (CN=1)  ◆
  ~~~~~~~~~~~~~~~ ~~~~~~~~~~~~~~~~
  ~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~   ← bifurcación (CN=3)  ▲
  ~~~~~~~~~~~~~~ ~ ~~~~~~~~~~~~~~
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Cada minucia se guarda como:  m = (x, y, θ)
  θ = orientación local de la cresta en ese punto
```

Estándar: ISO/IEC 19794-2:2005, ANSI/NIST-ITL 1-2011.

---

## Etapa 2 — Pipeline de extracción visual

```
 IMAGEN ORIGINAL          ORIENTATION FIELD       GABOR ENHANCED
 ┌────────────┐          ┌────────────┐       ┌────────────┐
 │ ~~~~~~~~~~~│          │ →→→→→→→→│       │ ___________│
 │ ~~~ ~~~~~~~~│          │ →→→→→→→→│       │ ___  _______│
 │ ~~  ~~~~~~~~│          │ ↗↗↗↗↗↗↗↗│       │ __  ________│
 │ ~~~~~~~~~~~│          │ ↗↗↗↗↗↗↗↗│       │ ___________│
 └────────────┘          └────────────┘       └────────────┘
  Escala de grises          Dirección de crestas      Crestas realzadas
  (ruido, manchas)          por bloque (Sobel)        (Gabor orientado)

 ESQUELETO                 MINUTIAE DETECTADAS
 ┌────────────┐          ┌────────────┐
 │  ────────│          │  ───────◆  │  ◆ = terminación
 │  ─  ─────│          │  ─  ─────│  ▲ = bifurcación
 │   \ ─────│          │   \ ─────│
 │  ──\  ──│          │  ──▲  ──│
 └────────────┘          └────────────┘
  Zhang-Suen 1px             Crossing Number
  (adelgazamiento)           CN=1 ◆, CN=3 ▲
```

---

## Etapa 3 — Descriptor de par 5-D (actual)

Para cada par de minucias `(mᵢ, mⱼ)` dentro de la misma huella:

```
        mᵢ (xᵢ, yᵢ, θᵢ)              mⱼ (xⱼ, yⱼ, θⱼ)
            ●────────────────────●
            ↑              d = dist(mᵢ, mⱼ)

  vector = (Δx, Δy, sin(Δθ), cos(Δθ), d)
           └─────────────────────────┘
                  5 dimensiones

  Limitación: geometrías simples (≈0.2 unidades, ~45°)
  existen en casi cualquier dedo → muchos falsos hits en KNN
```

---

## Etapa 4 — KNN en Qdrant

Cada par probe se busca en Qdrant por similitud coseno. Retorna los `top-K` pares más similares de toda la base de datos, agrupados por persona.

```
  Par probe: v_probe = (Δx=0.1, Δy=0.05, sin=0.7, cos=0.7, d=0.15)

  Qdrant retorna:
  ┌─────────────────────────────────────────────┐
  │ person_id   sim    mi_angle  query_pair_idx │
  ├─────────────────────────────────────────────┤
  │ persona_A   0.97   45°       2             │
  │ persona_A   0.94   46°       5             │
  │ persona_B   0.96   15°       2             │  ← mismo mi_angle que otros pares?
  │ persona_C   0.93   45°       7             │
  └─────────────────────────────────────────────┘
```

---

## Etapa 5 — Bozorth3 Linker: cómo filtra los falsos positivos

Este es el corazón del sistema. Para cada persona candidata:

### Paso 5.1 — Calcular dθ por par

```
  Para cada hit del KNN:

  dθ = mi_angle_hit - mi_angle_probe

  Ejemplo:
  ┌───────────────────────────────────────┐
  │ Par # │ probe θ │ hit θ │ dθ   │
  ├───────────────────────────────────────┤
  │  2   │  75°    │ 45°   │ -30° │
  │  5   │ 120°    │ 90°   │ -30° │  ← mismo dθ = COMPATIBLES
  │  7   │  30°    │ 12°   │ -18° │  ← dθ diferente = INCOMPATIBLES
  └───────────────────────────────────────┘
```

### Paso 5.2 — Union-Find agrupa pares compatibles

```
  Pares con |dθ₁ - dθ₂| ≤ 0.20 rad (11.5°) → se unen

  Antes del Union-Find:
  [Par2: dθ=-30°] [Par5: dθ=-30°] [Par7: dθ=-18°] [Par9: dθ=-31°]

  Después:
  Componente A: { Par2, Par5, Par9 }  ← todos votan por rotación -30°
  Componente B: { Par7 }              ← solitario, descartado (n<3)

  El componente más grande = mejor alineación global = candidato genuino
```

### Paso 5.3 — Guardias de calidad

```
  n = tamaño del componente más grande
  total = total de hits disponibles para esta persona

  Validar:
    n ≥ 3              (mínimo de pares consistentes)
    n / total ≥ 0.25   (al menos 25% del total concuerda)

  Si no pasa: resultado = None (descartado)
```

### Paso 5.4 — Score

```
  Para 1 solo candidato:
    score = min(1.0, n / saturation)   (saturation = 30)

  Para múltiples candidatos:
    margin = (n_ganador - n_mejor_fp) / max(n_ganador, 1)
    score  = (margin + 1) / 2

  Ejemplos:
    n=15, best_fp=3  → margin=0.80  → score=0.90  (match claro)
    n=5,  best_fp=4  → margin=0.20  → score=0.60  (incierto)
    n=3,  best_fp=6  → margin=-1.0  → score=0.00  (no match)
```

---

## Etapa 6 — Por qué las latentes son especialmente difíciles

```
  ROLLED (enrolamiento)          LATENTE (forense)
  ─────────────────             ─────────────────
  ┌──────────┐               ┌──────────┐
  │ ● ● ● ● ● │               │  ● ●   │  ← solo esquina
  │ ● ● ● ● ● │               │ ● ● ●  │  ← imagen completa
  │ ● ● ● ● ● │               └──────────┘
  │ ● ● ● ● ● │
  └──────────┘
  Imagen completa             Solo zona de contacto
  Posición controlada        Rotación y posición aleatorias
  Alta calidad                Ruido, manchas, presión variable
  dtheta_tol: 0.10 rad        dtheta_tol: 0.20 rad (NIST IR 8215)
```

Por esto la implementación usa `dtheta_tol=0.20 rad` y las guardias de componente. Sin estas protecciones, el linker produce matches en zonas random de la huella.

---

## Etapa 7 — Hoja de ruta: mejoras pendientes

### Fase B — MCC 144-D (reemplaza descriptor 5-D)

```
  Descriptor 5-D actual:          MCC 144-D (Fase B):

  Par (mᵢ, mⱼ)                    Cilindro 3D centrado en mᵢ
  vector 5D simple                 ┌─────────────────────┐
  colisiona con muchos            │ 12 sectores × 4 anillos  │
  pares de otros dedos            │ × 12 orientaciones       │
                                   │ = 144 dimensiones        │
                                   └─────────────────────┘
                                   alineado a θ_t = invariante
                                   a rotación y traslación

  Impacto: EER 1.82% vs ~8-10% (FVC2006, Cappelli 2010)
```

### Fase C — TPS post-match

```
  Después del Bozorth3 linker, se ajusta un modelo de deformación
  elástica (Thin Plate Spline) sobre los pares del componente ganador:

  Correspondencias genuinas:        Correspondencias falsas:
  ●─────●  (distorsión real ≤15px)   ●──────────────●  (incoherentes)
  E_TPS pequeño → penalización baja  E_TPS grande → penalización alta

  S_final = S_bozorth3 × (1 - E_TPS / E_max)
```

---

*Para detalles científicos completos con fórmulas y referencias, ver [`FINGERPRINT_SCIENCE.md`](./FINGERPRINT_SCIENCE.md).*

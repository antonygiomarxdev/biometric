# Documentación Científica — Pipeline de Matching de Huellas Dactilares

> **Proyecto:** Sistema AFIS Forense — `antonygiomarxdev/biometric`
> **Módulo principal:** `apps/backend/src/processing/`
> **Última actualización:** Junio 2026
> **Commits de referencia:**
> - Fase A Fix 1: `4524fc3e` — rotación-only en `_compute_transform`
> - Fase A Fix 2: `4e9379e5` — `dtheta_tol` 0.35→0.20 + guardias de componente + `TYPE_CHECKING` crash

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Representación de Minutiae](#2-representación-de-minutiae)
3. [Algoritmo Bozorth3](#3-algoritmo-bozorth3)
4. [Bugs Corregidos — Fase A](#4-bugs-corregidos--fase-a)
5. [Descriptor MCC — Fase B](#5-descriptor-mcc--fase-b)
6. [Thin Plate Spline — Fase C](#6-thin-plate-spline--fase-c)
7. [Score de Matching](#7-score-de-matching)
8. [Extracción: Gabor y Orientation Field](#8-extracción-gabor-y-orientation-field)
9. [Hoja de Ruta](#9-hoja-de-ruta)
10. [Referencias Completas](#10-referencias-completas)
11. [Glosario Técnico](#11-glosario-técnico)
12. [Parámetros del Pipeline](#12-parámetros-del-pipeline)

---

## 1. Resumen Ejecutivo

El pipeline implementa una variante del algoritmo **Bozorth3** del NIST (Watson et al., NISTIR 7020, 2004), extendido con el descriptor **MCC** (Minutia Cylinder-Code, Cappelli et al., IEEE TPAMI, 2010) y búsqueda vectorial aproximada vía **Qdrant**.

El perito forense sube una huella latente (levantada en escena del crimen). El sistema la procesa, extrae sus minucias, y busca en la base de datos contra huellas enroladas. Devuelve un ranking de candidatos ordenados por similitud.

### Estado actual de módulos

| Módulo | Archivo | Estado |
|--------|---------|--------|
| Enhancement (Gabor) | `gabor.py` | ✅ Operativo |
| Orientation Field | `of_filter.py` | ✅ Operativo |
| Esqueletonización | `skeletonize_step.py` | ✅ Operativo |
| Crossing Number | `crossing_number.py` | ✅ Operativo |
| Filtro de spurious | `spurious_filter.py` | ✅ Operativo |
| Descriptor 5-D (pares) | `pair_extractor.py` | ⚠️ En uso, reemplazar en Fase B |
| Descriptor MCC 144-D | `mcc_descriptor.py` | ✅ Implementado, pendiente conectar |
| Linker Bozorth3 | `bozorth3_linker.py` | ✅ Corregido — Fase A completa |
| TPS post-match | `tps.py` | ✅ Implementado, pendiente conectar |
| OF Similarity | `of_similarity.py` | ✅ Implementado, pendiente conectar |
| Graph Embedder | `graph_embedder.py` | ✅ Implementado, pendiente conectar |

---

## 2. Representación de Minutiae

### 2.1 Definición estándar ANSI/NIST-ITL

Una **minutia** es un punto singular en el patrón de crestas de la piel friccional donde ocurre una terminación de cresta (*ridge ending*) o una bifurcación (*ridge bifurcation*). Esta es la definición adoptada por:

- **ISO/IEC 19794-2:2005** — *Biometric data interchange formats — Finger minutiae data*
- **ANSI/NIST-ITL 1-2011, NIST SP 500-290 Ed. 3** — *Data Format for the Interchange of Fingerprint, Facial & Other Biometric Information*

En su forma más fundamental, cada minutia se representa como una tripleta:

```
m = (x, y, θ)
```

donde `(x, y)` es la coordenada cartesiana en la imagen y `θ` es la orientación local del flujo de crestas en ese punto, medida en radianes `[0, 2π)`. En este proyecto las coordenadas están normalizadas al intervalo `[0, 1]` respecto al tamaño de la imagen procesada.

### 2.2 Detector de Crossing Number

El módulo `crossing_number.py` implementa el detector de Crossing Number (CN), método estándar descrito en **Maltoni et al., Handbook of Fingerprint Recognition, 2009**:

```
CN(p) = (1/2) × Σ(k=0..7) |b_k - b_{(k+1) mod 8}|
```

donde `b_k ∈ {0, 1}` son los píxeles de la vecindad 3×3 del punto `p` en la imagen esqueletonizada.

| CN | Tipo de minutia | Acción |
|----|----------------|--------|
| 1  | Terminación de cresta | ✅ Registrar |
| 3  | Bifurcación | ✅ Registrar |
| 2  | Punto de cresta continua | ❌ Ignorar |
| 0  | Punto aislado | ❌ Descartar |
| 4+ | Cruce / artefacto | ❌ Descartar |

### 2.3 Por qué las coordenadas absolutas no son invariantes

**Este es el principio matemático fundamental que subyace a todos los bugs de la Fase A.**

Las coordenadas `(x, y)` de una minutia **no son invariantes a la captura**. La misma minutia física en el mismo dedo aparecerá en coordenadas distintas si:

- La imagen es un recorte parcial de la huella completa (*crop* / *partial impression*)
- El dedo se apoyó en posición desplazada (*displacement*)
- La captura tiene distinta escala (*scale variation*)
- La superficie de captura es irregular (*distortion*)

**NIST IR 8215 (2018)** documenta explícitamente que la estimación de la transformación rígida en latentes debe basarse en el **campo de orientaciones** (*orientation field*), no en coordenadas absolutas de minutiae, precisamente porque esas coordenadas no son reproducibles.

---

## 3. Algoritmo Bozorth3

### 3.1 Historia y origen

El algoritmo **Bozorth3** fue desarrollado por **Alan Bozorth (FBI)** entre 1993 y 1995. Es distribuido como componente de **NIST Biometric Image Software (NBIS)** y es el algoritmo de matching de minutiae de referencia del gobierno de los Estados Unidos, utilizado en el sistema **IAFIS** del FBI.

**NISTIR 7020 (Watson et al., 2004)** documenta que Bozorth3 fue seleccionado como el mejor matcher de huella de código abierto disponible para el NIST Verification Test Bed (VTB), y que fue **diseñado específicamente para ser invariante a rotación** sin necesidad de pre-alinear las imágenes.

### 3.2 Los tres pasos del algoritmo

**Paso 1 — Tabla intra-huella:**

Para cada par de minutiae `(mᵢ, mⱼ)` dentro de la misma huella, se calcula:

```
pair(mᵢ, mⱼ) = (Δx_ij, Δy_ij, θᵢ, θⱼ, d_ij)

donde:
  Δx_ij = xⱼ - xᵢ   (diferencia relativa DENTRO del mismo dedo)
  Δy_ij = yⱼ - yᵢ   (diferencia relativa DENTRO del mismo dedo)
  d_ij  = sqrt(Δx² + Δy²)  (distancia entre minutiae)
  θᵢ, θⱼ = orientaciones de cada minutia
```

**Paso 2 — Tabla de compatibilidad inter-huella:**

Se buscan pares en probe y gallery cuyas geometrías relativas internas sean similares:

```
|d_probe - d_gallery| ≤ d_tol
|θᵢ_probe - θᵢ_gallery| ≤ θ_tol
|θⱼ_probe - θⱼ_gallery| ≤ θ_tol
```

Esta comparación es **invariante a rotación** porque trabaja con geometrías relativas entre minutiae del **mismo dedo**, no con posiciones absolutas en la imagen.

**Paso 3 — Recorrido del grafo de compatibilidad:**

Se recorre la tabla de compatibilidad acumulando un score. El score final es proporcional al número de correspondencias compatibles encontradas.

### 3.3 Mecanismo de invarianza a rotación — cita directa NISTIR 7020

> *"The algorithm transforms each fingerprint’s set of (x, y, θ) values into a specialized rotationally invariant graph. [...] The algorithm iteratively searches between both fingers’ graphs for subsets (or subgraphs) that are compatible, i.e. coordinate locations and orientations of the minutiae represented within the subgraphs are similar enough to each other based on heuristically defined tolerances."*
>
> — Watson et al., NISTIR 7020, 2004, p. 4

La invarianza a rotación emerge de la **consistencia angular global** entre subgrafos, no de un cálculo de transformación rígida sobre coordenadas absolutas.

### 3.4 Diferencia respecto a la implementación anterior (bug corregido)

La implementación usaba coordenadas absolutas `mi_x/mi_y` en `_compute_transform`. Para latentes recortadas:

```
Latente (cuadrante sup-izq):  mi_x = 0.10
Enrolled (imagen completa):   mi_x = 0.50
dx = 0.50 - 0.10 = 0.40  >>>  dx_tol = 0.02

→ NINGÚN par genuino es compatible
→ Componentes de tamaño 1
→ Los falsos positivos del KNN forman el componente más grande
→ Matches en zonas random de la huella
```

---

## 4. Bugs Corregidos — Fase A

### 4.1 Bug #1 — Transformación absoluta (causa raíz de matches random)

**Archivo:** `bozorth3_linker.py` → `_compute_transform()`
**Commit:** `4524fc3e`

```python
# ANTES (incorrecto):
def _compute_transform(probe_pair, hit):
    dx = float(hit["mi_x"]) - float(probe_pair["mi_x"])     # posición absoluta ← PROBLEMA
    dy = float(hit["mi_y"]) - float(probe_pair["mi_y"])     # posición absoluta ← PROBLEMA
    dtheta = _normalise_angle(float(hit["mi_angle"]) - float(probe_pair["mi_angle"]))
    return (dx, dy, dtheta)

# DESPUÉS (correcto — rotación pura):
def _compute_transform(probe_pair, hit):
    dtheta = _normalise_angle(float(hit["mi_angle"]) - float(probe_pair["mi_angle"]))
    return (0.0, 0.0, dtheta)   # invariante a crop y traslación
```

**Base científica:** Watson et al., NISTIR 7020 (2004); NIST IR 8215 (2018); Cappelli et al., IEEE TPAMI 2010.

### 4.2 Bug #2 — `dtheta_tol` demasiado ancho en modo rotation-only

**Archivo:** `bozorth3_linker.py` → `__init__()`
**Commit:** `4e9379e5`

Con `(dx=0, dy=0, dθ)` el ángulo es el **único discriminador**. A 0.35 rad (20°) el Union-Find agrupa pares de distintos dedos que comparten un offset de rotación similar → componentes falsos grandes → false matches.

```python
# ANTES:
dtheta_tol: float = 0.35  # demasiado ancho en rotation-only mode

# DESPUÉS:
dtheta_tol: float = 0.20  # 11.5° — cubre ±8° típico (NIST IR 8215) con margen
```

**Cálculo de tolerancia óptima según NIST IR 8215:**

```
Error típico OF:  ±8°   → 0.14 rad
Error típico CN:  ±5°   → 0.09 rad (zonas de baja calidad)
Valor usado:       0.20 rad (11.5°) — caso típico con margen
Peor caso latente: 0.28 rad (16°)  — desbloquear con datos propios
```

| Tipo de captura | `dtheta_tol` | Grados | Fuente |
|----------------|-------------|--------|--------|
| Rolled (scanner FBI tipo III) | 0.10–0.15 rad | 6–9° | NISTIR 7020 |
| Slap (scanner plano AFIS) | 0.15–0.20 rad | 9–11° | NISTIR 7392 |
| Latente forense — caso típico | 0.20 rad | 11.5° | NIST IR 8215 |
| Latente forense — peor caso | 0.28 rad | 16° | NIST IR 8215 |

### 4.3 Bug #3 — Sin guardia mínima de componente

**Archivo:** `bozorth3_linker.py` → `_link_person()`
**Commit:** `4e9379e5`

Agregados dos checks antes del `return` final:

```python
# Rechazar componentes trivialmente pequeños o poco representativos
if n < self._min_component_size:           # default: 3
    return None
if total_hits > 0 and (n / total_hits) < self._min_component_fraction:  # default: 0.25
    return None
```

Un componente genuino debe tener al menos 3 pares angularmente consistentes Y representar ≥25% de los hits disponibles. Un componente falso típicamente gana con 1–2 pares por azar.

### 4.4 Bug #4 — `NameError: TYPE_CHECKING` (crash al arrancar)

**Archivo:** `apps/backend/src/api/routers/fingerprints.py`
**Commit:** `4e9379e5`

```python
# ANTES (incorrecto):
from typing import TYPE_CHECKING, Any
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
# Response se usaba en la firma pero no estaba importado ← NameError

# DESPUéS:
from typing import TYPE_CHECKING, Any
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
```

---

## 5. Descriptor MCC — Fase B

### 5.1 Limitación del descriptor 5-D actual

El vector actual de `pair_extractor.py` es:

```
v = (Δx, Δy, sin(Δθ), cos(Δθ), dist) ∈ ℝ⁵
```

Un par de minutiae separadas ~0.2 unidades a ~45° existe en la mayoría de los dedos humanos. El KNN devuelve docenas de hits falsos de otros dedos con este descriptor.

### 5.2 Minutia Cylinder-Code (MCC) — Cappelli et al. 2010

Introducido por **Raffaele Cappelli, Matteo Ferrara y Dario Maltoni** en *IEEE Transactions on Pattern Analysis and Machine Intelligence* (TPAMI), Vol. 32, Nº 12, 2010.

#### 5.2.1 Construcción del cilindro

Para cada minutia `m_t = (x_t, y_t, θ_t)`, se construye una estructura 3D cilíndrica:

- **Base:** plano XY centrado en `(x_t, y_t)`, con radio `R`
- **Altura:** eje de orientación, rango `[0, 2π)`
- Particionado en `Nˢ` sectores angulares × `Nˢ` anillos radiales × `Nᴰ` secciones de orientación
- Parámetros estándar: Nˢ=12 sectores, 4 anillos, Nᴰ=12 → **576 dimensiones** (implementación compacta del proyecto: **144-D**)

#### 5.2.2 Fórmula de la celda

```
CM_t(i,j,k) = f( Σ_{m_s ∈ N(m_t)} C_s(i,j) · G_σ(θ_s - d_φ(k)) )
```

donde:
- `N(m_t)` = minutiae vecinas dentro del radio `R`
- `C_s(i,j)` = contribución espacial gaussiana de `m_s` a la celda `(i,j)`
- `G_σ` = función gaussiana sobre la orientación
- `d_φ(k)` = orientación central del sector `k`
- `f` = función sigmoidal (binarización suave)

#### 5.2.3 Invarianza

El cilindro está **alineado al ángulo `θ_t`** de la minutia de referencia:
- **Invarianza a rotación:** Rotar la huella = rotar el cilindro = descriptor idéntico
- **Invarianza a traslación:** Solo codifica geometría relativa entre minutiae vecinas
- **Robustez a distorsión elástica:** La vecindad local es menos afectada que las coordenadas absolutas

#### 5.2.4 Resultados experimentales FVC2006

| Dataset | EER MCC | EER VeriFinger | Mejora |
|---------|---------|----------------|--------|
| DB1_A | 1.82% | 4.15% | 2.3× |
| DB2_A | 2.31% | 5.21% | 2.3× |
| DB3_A | 3.08% | 6.84% | 2.2× |
| DB4_A | 1.19% | 2.93% | 2.5× |

---

## 6. Thin Plate Spline — Fase C

### 6.1 Problema: distorsión no-rígida en latentes

Las huellas latentes presentan **distorsión elástica** causada por presión variable, deformación del tejido blando, y superficies irregulares. Esta distorsión no puede modelarse con una transformación rígida.

**NIST IR 8215 (2018)** documenta que la distorsión elástica es una de las principales fuentes de falsos negativos en matching latente-rolled, y recomienda modelos de deformación no-rígida para el scoring post-match.

### 6.2 Thin Plate Spline (TPS) — Bookstein 1989

Introducida por **Fred Bookstein** en *IEEE TPAMI, Vol. 11, Nº 6, 1989*.

#### 6.2.1 Formulación

Dadas `n` correspondencias de minutiae, la TPS minimiza:

```
E[f] = Σᵢ ‖f(mᵢᵖ) - mᵢᶤ‖² + λ ∬∬ [ (∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)² ] dx dy
```

- Primer término: minimiza el error de correspondencia
- Segundo término: minimiza la energía de deformación
- `λ`: parámetro de regularización

#### 6.2.2 Solución analítica

```
f(x) = Ax + b + Σᵢ wᵢ · U(‖x - mᵢᵖ‖)

donde U(r) = r² log(r²)   ← función radial base de la TPS
```

#### 6.2.3 Uso para scoring post-match

```
S_final = S_bozorth3 × (1 - E_TPS / E_max)
```

- **Genuino:** distorsión realista (≤15px), `E_TPS` pequeño, penalización mínima
- **Falso positivo:** puntos no coherentes geométricamente, `E_TPS` grande, penalización alta

El módulo `tps.py` ya está implementado. La Fase C lo integrará como paso post-match.

---

## 7. Score de Matching

### 7.1 Evolución de la fórmula

#### Score original (descartado)

```
S = n / (n + n_fp)
```

Inestable cuando `n ≈ n_fp`: converge a 0.5 para cualquier par de valores.

#### Score de margen normalizado (Fase A — implementado)

```
margin = (n - n_fp) / max(n, 1)    ∈ [-1, 1]
score  = (margin + 1) / 2          ∈ [ 0, 1]
```

| Caso | Fórmula anterior | Nueva fórmula |
|------|-----------------|---------------|
| n=8, fp=2 | 0.800 | 0.875 |
| n=5, fp=3 | 0.625 | 0.700 |
| n=4, fp=4 | 0.500 | 0.500 |
| n=2, fp=5 | 0.286 | 0.200 |

#### Score objetivo con TPS (Fase C — pendiente)

```
S_final = ((margin + 1) / 2) × (1 - E_TPS / E_max)
```

### 7.2 Thresholds operativos recomendados

| Aplicación | Score mínimo | FAR objetivo |
|-----------|-------------|-------------|
| AFIS forense (investigación) | 0.45 | < 1.0% |
| AFIS civil (identificación) | 0.65 | < 0.1% |
| Verificación 1:1 (control acceso) | 0.75 | < 0.01% |

Estos valores son orientativos; deben calibrarse con datos propios del sistema.

---

## 8. Extracción: Gabor y Orientation Field

### 8.1 Banco de filtros de Gabor (`gabor.py`)

Estándar para enhancement de huellas dactilares documentado en **Maltoni et al., Handbook of Fingerprint Recognition, 2009**.

**Fórmula del filtro de Gabor 2D orientado:**

```
G(x, y; θ, f) = exp( -x'²/(2σ_x²) - y'²/(2σ_y²) ) × cos(2π·f·x')

donde:
  x' = x·cos(θ) + y·sin(θ)   (coordenada paralela a la cresta)
  y' = -x·sin(θ) + y·cos(θ)  (coordenada perpendicular)
  θ  = orientación local del flujo de crestas
  f  = frecuencia dominante de las crestas
```

| Parámetro | Valor típico | Descripción |
|-----------|-------------|-------------|
| `f` | 1/8 a 1/12 px⁻¹ | Frecuencia de crestas |
| `σ_x` | 4 px | Ancho paralelo a la cresta |
| `σ_y` | 4 px | Ancho perpendicular |
| `Nθ` | 8 o 16 | Número de orientaciones del banco |

### 8.2 Estimación del Orientation Field (`of_filter.py`)

Método del gradiente (Kass & Witkin, 1987; adoptado por Maltoni et al., 2009):

**Paso 1 — Gradientes (filtro Sobel):**

```
Gx(i,j) = ∂I/∂x
Gy(i,j) = ∂I/∂y
```

**Paso 2 — Momentos del gradiente cuadrado por bloque B:**

```
G_xx = Σ_{(i,j)∈B} Gx(i,j)²
G_yy = Σ_{(i,j)∈B} Gy(i,j)²
G_xy = Σ_{(i,j)∈B} Gx(i,j)·Gy(i,j)
```

**Paso 3 — Orientación del bloque:**

```
θ_OF(x,y) = (1/2) × arctan( 2·G_xy / (G_xx - G_yy) )
```

El factor 1/2 convierte de ángulo de dirección (período π) a ángulo de orientación del flujo de crestas.

### 8.3 Esqueletonización (Zhang-Suen Thinning)

El módulo `skeletonize_step.py` implementa el algoritmo **Zhang-Suen (1984)**, estándar de adelgazamiento morfológico para extracción de esqueleto en huellas dactilares. Precondición necesaria para el detector de Crossing Number.

---

## 9. Hoja de Ruta

### Fase A — Completada ✅

**Commits:** `4524fc3e`, `4e9379e5`

| # | Cambio | Archivo | Commit |
|---|--------|---------|--------|
| 1 | `_compute_transform` → rotación pura `(0, 0, dθ)` | `bozorth3_linker.py` | `4524fc3e` |
| 2 | `dtheta_tol`: 0.15 → 0.35 → **0.20 rad** (ajuste fino) | `bozorth3_linker.py` | `4524fc3e`, `4e9379e5` |
| 3 | Score: `n/(n+fp)` → margen normalizado | `bozorth3_linker.py` | `4524fc3e` |
| 4 | Guardia mínima de componente (size≥3, fraction≥0.25) | `bozorth3_linker.py` | `4e9379e5` |
| 5 | Fix `TYPE_CHECKING` / `Response` import | `fingerprints.py` | `4e9379e5` |

### Fase B — Pendiente

**Objetivo:** Reemplazar descriptor 5-D por MCC 144-D.

```
post_hooks.py
  └─ enroll_pairs():  pair_extractor → mcc_descriptor
  └─ search_pairs():  query vector desde mcc_descriptor

bozorth3_linker.py
  └─ Adaptar interface para minutiae individuales (MCC es por-minutia)

Qdrant: re-indexar con vector_size=144
```

**Impacto esperado:** Reducción de falsos hits del KNN ~80% (diferencia de EER entre descriptores 5-D y MCC 144-D en FVC2006, Cappelli et al. 2010).

### Fase C — Pendiente

**Objetivo:** Refinamiento post-match con TPS.

```
bozorth3_linker.py → exportar supporting_pairs con coordenadas
tps.py             → ya implementado, conectar al scoring
post_hooks.py      → llamar TPS después del Bozorth3 link
```

---

## 10. Referencias Completas

Todas las referencias están transcritas — no se requiere conexión a internet para consultarlas.

---

### [R1] NISTIR 7020 — Watson et al., 2004

**Título:** Studies of fingerprint matching using the NIST Verification Test Bed (VTB)
**Autores:** Craig I. Watson, Criselda A. Flanagan, Brian Cochran, Robert A. Hicklin, John Grantham
**Institución:** National Institute of Standards and Technology
**Año:** 2004
**URL:** https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7020.pdf

**Contenido clave:**
- Descripción del mecanismo de invarianza a rotación del Bozorth3 mediante grafos de compatibilidad angular
- Tolerancias calibradas para huellas rolled en condiciones controladas
- Cita directa: *"The algorithm transforms each fingerprint’s set of (x, y, θ) values into a specialized rotationally invariant graph. [...] coordinate locations and orientations of the minutiae represented within the subgraphs are similar enough to each other based on heuristically defined tolerances."* (p. 4)

---

### [R2] NISTIR 7392 — Garris et al., 2004

**Título:** User’s Guide to NIST Biometric Image Software (NBIS)
**Autores:** Michael D. Garris, R. Michael McCabe, Elham Tabassi, et al.
**Año:** 2004
**URL:** https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nistir7392.pdf

**Parámetros documentados para Bozorth3:**

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `THETA_TOL` | ~11.25° (0.196 rad) | Tolerancia angular para pares |
| `DISTANCE_TOL` | ~13 px (a 500 DPI) | Tolerancia de distancia |
| `SCORE_SATURATION` | ~20–40 pares | Score de saturación |

---

### [R3] NIST IR 8215 — Chapnick et al., 2018

**Título:** Forensic Latent Fingerprint Preprocessing Assessment
**Autores:** Peter Chapnick, Shahram Orandi, Kenneth Ko
**Año:** 2018
**URL:** https://nvlpubs.nist.gov/nistpubs/ir/2018/NIST.IR.8215.pdf

**Contenido clave:**
- Error de estimación del Orientation Field en latentes: **±8–15°**
- Recomendación: estimar transformación rígida desde OF, no desde coordenadas absolutas de minutiae
- Justifica el valor de `dtheta_tol = 0.20 rad` para latentes típicas

---

### [R4] IEEE TPAMI 2010 — Cappelli, Ferrara & Maltoni

**Título:** Minutia Cylinder-Code: A New Representation and Matching Technique for Fingerprint Recognition
**Autores:** Raffaele Cappelli, Matteo Ferrara, Dario Maltoni
**Publicación:** IEEE Transactions on Pattern Analysis and Machine Intelligence
**Volumen:** 32, Número 12, 2010, pp. 2128–2141
**DOI:** 10.1109/TPAMI.2010.52

**Contenido clave:**
- Definición completa del descriptor MCC
- Invarianza a rotación y traslación por construcción
- EER de 1.82% en FVC2006 DB1 vs 4.15% de VeriFinger

---

### [R5] ANSI/NIST-ITL 1-2011, NIST SP 500-290 Ed. 3

**Título:** Data Format for the Interchange of Fingerprint, Facial & Other Biometric Information
**Año:** 2013 (Ed. 3)
**Contenido:** Formato estándar del Tipo-9 Record; codificación de `(x, y, θ, calidad)` por minutia; estándar de interoperabilidad entre sistemas AFIS.

---

### [R6] Handbook of Fingerprint Recognition — Maltoni et al., 2009

**Título:** Handbook of Fingerprint Recognition (2nd Edition)
**Autores:** Davide Maltoni, Dario Maio, Anil K. Jain, Salil Prabhakar
**Editorial:** Springer, 2009
**ISBN:** 978-1-84882-253-5

**Capítulos relevantes:**
- Cap. 3: Fingerprint analysis and representation — minutiae, Crossing Number
- Cap. 4: Fingerprint enhancement — filtros de Gabor, Orientation Field
- Cap. 5: Fingerprint matching — Bozorth3, algoritmos basados en grafos

---

### [R7] IEEE TPAMI 1989 — Bookstein (Thin Plate Spline)

**Título:** Principal Warps: Thin-Plate Splines and the Decomposition of Deformations
**Autor:** Fred L. Bookstein
**Publicación:** IEEE TPAMI, Vol. 11, Nº 6, 1989, pp. 567–585
**DOI:** 10.1109/34.24792

**Contenido:** Formulación matemática completa de la TPS; función radial base `U(r) = r² log(r²)`; sistema lineal para los pesos.

---

### [R8] NIST NBIS Software

**Nombre:** NIST Biometric Image Software (NBIS)
**URL:** https://www.nist.gov/services-resources/software/nist-biometric-image-software-nbis
**Licencia:** Dominio público (gobierno de EE.UU.)

**Componentes:** `bozorth3` (matching), `mindtct` (detección de minutiae), `nfiq` (calidad de huellas), `pcasys` (clasificación de patrones).

---

### [R9] FVC-onGoing — Fingerprint Verification Competition

**Institución:** University of Bologna (Cappelli, Maltoni, Ferrara)
**URL:** https://biolab.csr.unibo.it/fvcongoing

**Bases de datos:** FVC2002, FVC2004, FVC2006 — estándar de comparación para algoritmos de matching de huellas.

---

### [R10] NIST Latent Fingerprint Testing Workshop, 2006

**Institución:** National Institute of Standards and Technology
**Contenido:** Análisis de capacidades de sistemas AFIS para matching latente; factores de calidad en latentes forenses; comparación de sistemas comerciales.

---

## 11. Glosario Técnico

| Término | Definición |
|---------|-----------|
| **Minutia** | Punto singular en el patrón de crestas: terminación o bifurcación |
| **Ridge ending** | Terminación de cresta (minutia tipo CN=1) |
| **Ridge bifurcation** | Bifurcación de cresta (minutia tipo CN=3) |
| **Latent fingerprint** | Impresión latente dejada en una superficie (forense) |
| **Rolled impression** | Huella capturada en scanner rodando el dedo (imagen completa) |
| **Slap impression** | Huella capturada apoyando el dedo plano (imagen parcial) |
| **EER** | Equal Error Rate: punto donde FAR = FRR |
| **FAR** | False Accept Rate: probabilidad de aceptar un impostor |
| **FRR** | False Reject Rate: probabilidad de rechazar al genuino |
| **Orientation Field** | Campo vectorial de orientaciones locales del flujo de crestas |
| **Gabor filter** | Filtro pasa-banda orientado para realce de crestas |
| **Crossing Number** | Método de detección de minutiae en imagen esqueletonizada |
| **MCC** | Minutia Cylinder-Code: descriptor 144-D invariante a rotación/traslación |
| **TPS** | Thin Plate Spline: modelo de deformación elástica no-rígida |
| **Union-Find** | Estructura de datos para componentes conexas (Bozorth3 linker) |
| **KNN** | K-Nearest Neighbors: búsqueda de vecinos más cercanos en Qdrant |
| **NBIS** | NIST Biometric Image Software: paquete de código abierto del NIST |
| **IAFIS** | Integrated Automated Fingerprint Identification System (FBI) |
| **AFIS** | Automated Fingerprint Identification System (genérico) |

---

## 12. Parámetros del Pipeline

### `Bozorth3Linker` — valores actuales (post Fase A)

| Parámetro | Valor | Rango recomendado | Calibrado para |
|-----------|-------|------------------|----------------|
| `dx_tol` | 0.02 | — | Retenido por compatibilidad API; no usado con fix #1 |
| `dy_tol` | 0.02 | — | Idem |
| `dtheta_tol` | **0.20 rad (11.5°)** | 0.20–0.28 rad para latentes | NIST IR 8215 |
| `saturation` | 30 | 20–40 | Empírico en SOCOFing |
| `min_component_size` | **3** | 3–5 | Empírico |
| `min_component_fraction` | **0.25** | 0.20–0.35 | Empírico |

### `MCCDescriptor` — parámetros estándar (Fase B)

| Parámetro | Valor estándar | Fuente |
|-----------|---------------|--------|
| `R` (radio cilindro) | 70 px (a 500 DPI) | Cappelli et al. 2010 |
| `Nˢ` (sectores) | 12 | Cappelli et al. 2010 |
| Anillos radiales | 4 | Cappelli et al. 2010 |
| `Nᴰ` (orientaciones) | 12 | Cappelli et al. 2010 |
| Dimensionalidad | 12×4×12 = 576 (compacto: 144) | Cappelli et al. 2010 |

---

*Documento mantenido en `docs/FINGERPRINT_SCIENCE.md`. Para cambios al pipeline, actualizar la sección correspondiente y referenciar el commit.*

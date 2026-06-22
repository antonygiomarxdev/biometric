# Convenciones del Spike 04 (DeepPrint POC)

## Alcance acotado: lo que el spike prueba y lo que NO prueba

### Qué prueba
1. **Viabilidad técnica**: ¿el modelo pre-entrenado carga, corre y
   produce embeddings?
2. **Rank-1 comparable**: ¿el matching por coseno sobre embeddings
   iguala o supera el baseline de Bozorth3 sobre los MISMOS datos?
3. **Decodificación a minucias**: ¿las minucias decodificadas son
   utilizables (no son ruido)?

### Qué NO prueba
1. **Latencia de producción**: el POC no mide latencia bajo carga
   concurrente. Una métrica para una fase posterior.
2. **Escalabilidad a 1M+**: el POC corre sobre SOCOFing (600
   personas). La escalabilidad es una métrica de la fase de
   integración.
3. **Throughput de Qdrant con 512-D**: el POC usa comparación
   in-memory. La integración con Qdrant es para después.
4. **Compatibilidad con el dictamen pericial**: el POC no evalúa si
   las minucias decodificadas son forensicamente significativas. Eso
   requiere revisión por el perito humano.

## Decisiones de implementación

### ¿Qué modelo usar?
- **DeepPrint_Tex_512**: solo la rama de textura, embedding 512-D.
- **DeepPrint_TexMinu_512**: textura + minutiae (más complejo).
- **DeepPrint_LocTexMinu_512**: localización + textura + minutiae.

**Decisión**: empezamos con `DeepPrint_Tex_512` por simplicidad.
Si el matching funciona, probamos las variantes más complejas.

### ¿Qué tamaño de embedding?
- 192, 256, 512, 1024 (tested in the paper)
- El paper recomienda 512.

**Decisión**: 512-D (el recomendado por el paper).

### ¿Preprocesamiento?
- El paper entrena con augmentations: rotación ±15°, traslación ±25 px,
  gain/contrast.
- Para inference, solo resize a 299x299 (input size de Inception v4).

**Decisión**: el POC usa SOLO el resize, sin augmentations. Las
augmentations son para training.

### ¿Comparación con baseline?
- El baseline actual (Bozorth3 pairs) corre sobre los MISMOS datos.
- El POC debe correr el baseline como subproceso y comparar Rank-1.

**Decisión**: el POC implementa su propio matching por coseno y
compara con el baseline llamando al matcher de producción (vía
API o reusando el código).

## Convenciones de archivo

- **Embeddings**: guardados como `numpy.ndarray` con shape `(N, 512)`
  en `.npy` files.
- **Resultados**: JSON con Rank-1, Rank-5, EER, latencia.
- **Visualizaciones**: PNG con el embedding proyectado (t-SNE o PCA)
  y las minucias decodificadas superpuestas.

## Lo que NO se mete al POC (decidido)

- **MCC / Bozorth3 fine-tuning**: no tocamos el matcher actual.
- **Qdrant integration**: no escribimos en Qdrant. Solo comparación
  in-memory.
- **Perito UI**: no tocamos el frontend.
- **Configuración del sistema**: no tocamos `core/config.py`.
- **Migrations**: no tocamos la base de datos.

## Riesgos identificados

1. **Licencia LGPL-3.0**: si DeepPrint se usa en producción, hay
   que revisar la compatibilidad con la licencia del proyecto. Esto
   se evalúa al final del spike si los resultados son positivos.

2. **Pre-entrenado en SFinGe sintético + MCYT**: el modelo no vio
   latentes nicaragüenses. La generalización es incierta. El POC
   con SOCOFing Real es la primera prueba.

3. **GPU requirement**: el POC requiere GPU. El proyecto tiene
   `docker-compose.gpu.yml`, pero el POC se corre en la workstation
   del perito (con GPU). Sin GPU, el POC no se puede hacer.

4. **Decoder ISO 19794-2**: el repo upstream lo incluye pero no
   sabemos si produce minucias utilizables. Hay que validarlo
   visualmente.

## Plan de ejecución (días)

- **Día 1**: setup, descarga del modelo pre-entrenado, primer test
  con 1 imagen.
- **Día 2**: extracción de embeddings para todo SOCOFing Real
  (600 personas), comparación con baseline.
- **Día 3**: extracción para Altered-Easy, comparación.
- **Día 4**: decodificación ISO 19794-2, validación visual de
  minucias decodificadas.
- **Día 5**: latentes del perito (si están disponibles), reporte
  final, decisión go/no-go.
- **Día 6-7**: buffer para issues no anticipados.

## Criterio de salida (cierre del spike)

El spike se cierra con un `REPORT.md` que dice:
- **GO**: el POC demuestra Rank-1 comparable o superior, latencia
  aceptable, minucias decodificadas utilizables. La fase de
  integración se planifica.
- **NO-GO**: el POC no demuestra ventajas sobre el pipeline clásico.
  Volvemos a mejorar el preprocesador o exploramos otra arquitectura
  (FingerNet, MinutiaeNet).
- **CHANGE-APPROACH**: el POC revela issues específicos que requieren
  cambios antes de re-evaluar (ej: cambiar tamaño de embedding,
  cambiar modelo, entrenar).

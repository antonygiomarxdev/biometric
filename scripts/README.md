# Scripts de Prueba - SOCOFing Dataset

## Descargar el Dataset

1. Descargar SOCOFing desde Kaggle:
   ```
   https://www.kaggle.com/datasets/ruizgara/socofing
   ```

2. Extraer en la carpeta `data/`:
   ```
   data/
   └── SOCOFing/
       ├── Real/
       │   ├── 1__M_Left_index_finger.BMP
       │   ├── 1__M_Left_middle_finger.BMP
       │   └── ...
       └── Altered/
           ├── Altered-Easy/
           ├── Altered-Medium/
           └── Altered-Hard/
   ```

## Uso

### 1. Ver Estadísticas del Dataset

```bash
# Local
python scripts/load_socofing.py stats data/SOCOFing

# Docker
docker-compose exec api python scripts/load_socofing.py stats /data/SOCOFing
```

### 2. Registrar Huellas

```bash
# Registrar 100 huellas del subset "Real"
python scripts/load_socofing.py register data/SOCOFing --limit 100

# Registrar todas las huellas (⚠️ puede tomar tiempo)
python scripts/load_socofing.py register data/SOCOFing

# Con Docker
docker-compose exec api python scripts/load_socofing.py register /data/SOCOFing --limit 100

# Continuar si hay errores
python scripts/load_socofing.py register data/SOCOFing --limit 100 --skip-errors
```

**Tiempos estimados:**
- 100 huellas: ~5-10 minutos
- 1000 huellas: ~50-90 minutos
- 6000 huellas completas: ~5-8 horas

### 3. Identificar Huellas

```bash
# Identificar 10 huellas aleatorias
python scripts/load_socofing.py identify data/SOCOFing --num-queries 10

# Identificar 50 huellas
python scripts/load_socofing.py identify data/SOCOFing --num-queries 50

# Con Docker
docker-compose exec api python scripts/load_socofing.py identify /data/SOCOFing --num-queries 10
```

## Ejemplo de Flujo Completo

```powershell
# 1. Iniciar servicios
docker-compose -f docker-compose.dev.yml up -d

# 2. Copiar dataset al volumen de Docker (si es necesario)
docker cp data/SOCOFing biometric_api_dev:/data/

# 3. Ver estadísticas
docker-compose exec api python scripts/load_socofing.py stats /data/SOCOFing

# 4. Registrar subset pequeño para pruebas (50 huellas)
docker-compose exec api python scripts/load_socofing.py register /data/SOCOFing --limit 50 --skip-errors

# 5. Verificar registros
docker-compose exec api python -m src.api.cli status

# 6. Identificar algunas huellas
docker-compose exec api python scripts/load_socofing.py identify /data/SOCOFing --num-queries 20

# 7. Ver métricas
curl http://localhost:8000/metrics
```

## Resultados Esperados

### Accuracy Esperado
- **Same finger, same person**: >95%
- **Different finger, same person**: ~0% (correcto, no debería hacer match)
- **Different person**: ~0% (correcto)

### Performance
- **Extracción**: 150-300ms por huella
- **Registro**: 200-400ms por huella
- **Identificación**: 10-50ms por búsqueda (depende del tamaño del índice)

## Troubleshooting

### Error: "No se pudieron extraer minutiae"
Algunas imágenes de SOCOFing pueden tener baja calidad. Usa `--skip-errors`:
```bash
python scripts/load_socofing.py register data/SOCOFing --skip-errors
```

### Performance muy lento
- Verificar que PostgreSQL tiene suficiente RAM
- Ajustar `vector_index_lists` en config para tu tamaño de dataset
- Considerar procesar en batches más pequeños

### Memoria insuficiente
```bash
# Registrar en lotes
python scripts/load_socofing.py register data/SOCOFing --limit 100
python scripts/load_socofing.py register data/SOCOFing --limit 200
# etc...
```

## Análisis Avanzado

Puedes extender el script para análisis más detallados:

```python
# Ejemplo: Analizar calidad de extracción
for img_path, metadata in images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    fingerprint = fingerprint_service.process_image(img)
    
    print(f"{metadata['person_id']}: {len(fingerprint.minutiae)} minutiae")
```

## Estructura de Nombres SOCOFing

Formato: `{ID}__{Gender}_{Hand}_{Finger}.BMP`

Ejemplos:
- `1__M_Left_index_finger.BMP` → Person 1, Male, Left hand, Index finger
- `150__F_Right_thumb.BMP` → Person 150, Female, Right hand, Thumb

El script parsea automáticamente esta información.

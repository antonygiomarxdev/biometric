# 🚀 Test Rápido con SOCOFing

## Setup Ultra Rápido

### 1. Instalar dependencia adicional
```bash
pip install tqdm
# o con uv
uv pip install tqdm
```

### 2. Configurar ruta del dataset

Abre `test_socofing.py` y ajusta esta línea si tu dataset está en otra ubicación:

```python
DATASET_PATH = "data/SOCOFing/Real"  # ← Cambia esto
```

### 3. Ejecutar

```bash
python test_socofing.py
```

## 🎯 Opciones del Script

Cuando ejecutes, verás:

```
¿Qué quieres hacer?
  1. Indexar dataset (extraer features y guardar)
  2. Buscar huella (asume que ya indexaste)
  3. Indexar Y buscar (flujo completo)
```

### Opción 3 (Recomendada para primera vez)

1. Extrae features de las primeras 100 huellas (configurable con `MAX_REGISTER`)
2. Las indexa en PostgreSQL + Qdrant
3. Te deja buscar interactivamente

### Modo Búsqueda Interactiva

Después de indexar, entras en modo búsqueda:

```
1. Buscar imagen aleatoria del dataset    ← Elige una imagen random
2. Buscar imagen específica (ruta)        ← Especifica una imagen
3. Ver estadísticas                       ← Cuántas huellas hay indexadas
4. Salir
```

## 📊 Qué Esperar

### Al Indexar
```
📁 Encontradas 100 imágenes en data/SOCOFing/Real
🚀 Iniciando indexación...

Indexando: 100%|████████████| 100/100 [08:24<00:00, 5.04s/it]

✅ Indexación completada!
   - Exitosas: 95
   - Errores: 5
   - Tiempo: 504.2s (5.31s por huella)
```

### Al Buscar
```
🔍 Buscando: data/SOCOFing/Real/1__M_Left_index_finger.BMP
   Extrayendo features...
   ✅ Extraídas 42 minutiae
   Buscando en el índice...

============================================================
✅ MATCH ENCONTRADO!
   Persona: Person P0001 - M Left index finger
   ID: P0001
   Score: 1.0000
   Distancia: 0.0000
============================================================
```

## ⚙️ Configuración

### Cambiar número de huellas a indexar

En `test_socofing.py`:
```python
MAX_REGISTER = 100  # Cambia esto (None = todas las huellas)
```

### Usar todo el dataset

```python
MAX_REGISTER = None  # ⚠️ ~6000 huellas, puede tomar 6-8 horas
```

### Performance esperado

| Huellas | Tiempo Indexación | Tiempo Búsqueda |
|---------|-------------------|-----------------|
| 100     | ~5-10 min         | <50ms           |
| 500     | ~30-50 min        | <100ms          |
| 1000    | ~1-1.5 hrs        | <150ms          |
| 6000    | ~6-8 hrs          | <300ms          |

## 🧪 Ejemplo Completo

```bash
# 1. Instalar tqdm
pip install tqdm

# 2. Verificar que PostgreSQL está corriendo
docker-compose -f docker-compose.dev.yml up -d postgres

# 3. Ejecutar script
python test_socofing.py

# 4. Elegir opción 3 (indexar y buscar)

# 5. Esperar a que indexe...

# 6. En el modo búsqueda, elegir opción 1 varias veces
#    para probar con diferentes huellas aleatorias
```

## 🎯 Validar que Funciona

Deberías ver:
- **Score > 0.95** cuando busca la misma huella
- **Match correcto** del person_id
- **Latencia < 200ms** en búsquedas

## ❌ Troubleshooting

### Error: "No se encuentra el dataset"
```python
# En test_socofing.py, cambia:
DATASET_PATH = "tu/ruta/a/SOCOFing/Real"
```

### Error: "No module named 'tqdm'"
```bash
pip install tqdm
```

### Muy lento al indexar
- Normal, cada huella toma ~5s (procesamiento intensivo)
- Reduce `MAX_REGISTER` a 50 para pruebas rápidas
- El tiempo se invierte solo una vez

### Base de datos no responde
```bash
# Asegúrate que PostgreSQL está corriendo
docker-compose -f docker-compose.dev.yml up -d postgres

# O inicia todo
docker-compose -f docker-compose.dev.yml up -d
```

## 📈 Mejorar Performance

Si quieres más velocidad:

1. **Reducir resolución de imagen** (en `src/core/config.py`):
   ```python
   image_resize_width=300  # Default: 350
   ```

2. **Usar menos minutiae** (trade-off accuracy vs speed)

3. **Procesar en paralelo** (futuro)

---

**¿Listo para probar?** 🚀

```bash
python test_socofing.py
```

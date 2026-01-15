# 🚀 GPU QuickStart - 5 Minutos

## ✅ Estado Actual

Tu sistema **YA tiene soporte GPU** integrado con CuPy. Solo necesitas:

1. **CUDA Toolkit** (si no lo tienes)
2. **CuPy** instalado

---

## 🔧 Instalación (Windows)

### Paso 1: Verificar CUDA

```powershell
nvidia-smi
```

**Si funciona** → Ya tienes CUDA, salta al Paso 2

**Si NO funciona** → Instalar CUDA:
1. Descargar: https://developer.nvidia.com/cuda-downloads
2. Instalar (siguiente → siguiente → instalar)
3. Reiniciar terminal

---

### Paso 2: Instalar CuPy

```powershell
# Asegúrate de estar en el directorio del proyecto
cd c:\Users\ksante\dev\biometric

# Instalar con uv
uv sync

# Verificar instalación
python -c "import cupy as cp; print('✅ GPU:', cp.cuda.Device(0).compute_capability)"
```

**Output esperado:**
```
✅ GPU: (8, 9)  # RTX 4070
```

---

## 🧪 Probar GPU

### 1. Ver estado de GPU

```powershell
python -c "from src.core.gpu_utils import print_gpu_status; print_gpu_status()"
```

### 2. Benchmark CPU vs GPU

```powershell
# Comparación automática
python benchmark_gpu.py --mode compare --iterations 3
```

**Output esperado:**
```
======================================================================
🔥 Benchmark: CPU
======================================================================
  Iteración 1/3: 2.34s - Minutiae: 42
  ...
  Promedio: 2.40s

======================================================================
🔥 Benchmark: GPU
======================================================================
  Iteración 1/3: 1.15s - Minutiae: 42
  ...
  Promedio: 1.20s

🔥 Speedup: 2.00x
   ✅ ¡GPU es 2.0x más rápida!
======================================================================
```

### 3. Procesar dataset completo con GPU

```powershell
# Indexar 20 imágenes con GPU
python test_socofing.py
# Seleccionar opción 1 (indexar)
```

El sistema **automáticamente usa GPU** si está disponible.

---

## 🎮 Control Manual

### Forzar CPU (sin GPU)

```powershell
$env:FORCE_CPU="1"
python identify_single.py "data\SOCOFing\Real\1__M_Left_index_finger.BMP"
```

### Ver backend actual

```python
from src.core.gpu_utils import GPUConfig
print(GPUConfig.get_backend())  # "GPU (CuPy)" o "CPU (NumPy)"
```

---

## 📊 Performance Esperado

| Operación | CPU | GPU (CuPy) | Speedup |
|-----------|-----|------------|---------|
| 1 imagen | ~2-3s | ~1-1.5s | **2-2.5x** |
| 20 imágenes | ~40-60s | ~20-30s | **2x** |
| 100 imágenes | ~4-5 min | ~2-2.5 min | **2x** |

---

## ❓ Problemas Comunes

### "No module named 'cupy'"

```powershell
uv pip install cupy-cuda12x
```

### "CUDA not found"

Instalar CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

### GPU no se detecta

```powershell
# Verificar drivers
nvidia-smi

# Verificar CuPy
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

---

## 🎯 ¿Qué Sigue?

✅ **Ya tienes GPU funcionando** → Usa el sistema normalmente
- `test_socofing.py` usa GPU automáticamente
- `identify_single.py` usa GPU automáticamente
- `benchmark_gpu.py` para comparar performance

⏳ **Para máximo rendimiento** (5-10x en lugar de 2-3x):
- Ver `GPU_SETUP.md` para compilar OpenCV con CUDA
- Requiere ~1 hora pero da speedup adicional
- Recomendado solo para producción a gran escala

---

## 📚 Más Información

- **Guía completa**: Ver `GPU_SETUP.md`
- **Benchmarks**: `python benchmark_gpu.py --help`
- **Documentación CuPy**: https://docs.cupy.dev/

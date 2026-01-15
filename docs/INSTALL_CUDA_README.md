# 🚀 Instalador Semi-Automático de CUDA

## ¿Qué hace este script?

1. ✅ **Detecta** si tienes GPU NVIDIA
2. ✅ **Verifica** si CUDA está instalado
3. ✅ **Busca** instalaciones de CUDA no configuradas
4. ✅ **Descarga** el instalador de CUDA 12.6 (~3.5GB)
5. ✅ **Abre** el instalador para que lo completes
6. ✅ **Verifica** que CuPy puede usar la GPU

---

## 🎯 Uso Rápido

```powershell
python install_cuda.py
```

El script te guiará paso a paso.

---

## 📋 Requisitos

- ✅ GPU NVIDIA (RTX 4070 en tu caso)
- ✅ Drivers NVIDIA instalados (`nvidia-smi` debe funcionar)
- ✅ ~4GB de espacio en disco
- ✅ Conexión a internet
- ✅ Permisos de administrador (para la instalación)

---

## 🔄 Flujo del Script

### Escenario 1: CUDA no instalado

```
1. Detecta que necesitas CUDA
2. Ofrece descargarlo automáticamente
3. Descarga el instalador (~10-30 min)
4. Abre el instalador
5. TÚ completas la instalación
6. Verifica que funciona
```

### Escenario 2: CUDA instalado pero no en PATH

```
1. Detecta CUDA en C:\Program Files\...
2. Te muestra los comandos para agregarlo al PATH
3. Espera a que lo agregues
4. Verifica que funciona
```

### Escenario 3: CUDA ya funciona

```
1. Detecta que nvcc funciona
2. Verifica que CuPy puede usarlo
3. ¡Listo!
```

---

## ⚙️ Opciones del Script

Cuando ejecutas, tienes 3 opciones:

1. **Descarga e instalación semi-automática** (Recomendado)
   - Descarga CUDA 12.6 automáticamente
   - Abre el instalador
   - Tú completas el wizard

2. **Descarga manual**
   - Abre el navegador en la página de NVIDIA
   - Descargas e instalas manualmente

3. **Salir**
   - Usa CPU por ahora
   - Instala CUDA más tarde

---

## 🎮 Después de Instalar

1. **Reinicia la terminal/PowerShell** (importante)

2. Verifica que funciona:
   ```powershell
   nvcc --version
   ```

3. Prueba el sistema:
   ```powershell
   python test_socofing.py
   ```

   Debería mostrar:
   ```
   ⚙️  Backend: GPU (CuPy)
   ```

---

## ❓ Troubleshooting

### Error: "nvcc no encontrado" después de instalar

**Solución:** CUDA no está en el PATH

```powershell
# Verificar si existe
dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\"

# Si existe, ejecutar install_cuda.py de nuevo
# El script detectará y te ayudará a agregarlo al PATH
```

### Error: "CuPy no puede usar CUDA"

**Solución 1:** Reiniciar terminal
```powershell
# Cierra y abre de nuevo PowerShell
```

**Solución 2:** Reinstalar CuPy
```powershell
uv pip install --force-reinstall cupy-cuda12x
```

### Error: "La descarga falló"

**Solución:** Descarga manual
```powershell
# El script abrirá el navegador automáticamente
# O ve directamente a:
# https://developer.nvidia.com/cuda-downloads
```

---

## 📊 Tiempos Esperados

| Etapa | Tiempo |
|-------|--------|
| Detección | < 5 seg |
| Descarga (~3.5GB) | 10-30 min |
| Instalación | 10-20 min |
| **Total** | **20-50 min** |

---

## 🆚 CPU vs GPU

### Con CPU (actual)
- 1 imagen: ~2-3s
- 100 imágenes: ~4-5 min

### Con GPU (después de instalar)
- 1 imagen: ~1-1.5s
- 100 imágenes: ~2-2.5 min

**Speedup: ~2x** 🚀

---

## 💡 Recomendación

**Para empezar YA:** Usa CPU (sin instalar CUDA)
```powershell
python test_socofing.py
# Funcionará automáticamente con CPU
```

**Para producción:** Instala CUDA cuando necesites procesar miles de imágenes
```powershell
python install_cuda.py
```

---

## 🔗 Links Útiles

- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- CuPy Documentation: https://docs.cupy.dev/
- GPU Setup Guide: Ver `GPU_SETUP.md`

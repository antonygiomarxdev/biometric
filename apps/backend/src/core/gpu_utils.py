"""GPU acceleration utilities with automatic CPU fallback."""

import logging
import os
from typing import Any, Union, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

import numpy as np

# Type definitions
ArrayLike = Union[np.ndarray, "cp.ndarray"] # type: ignore

# Intentar importar CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    
    # Verificar que hay GPU disponible
    try:
        # Try a simple GPU operation to verify CUDA works
        test_array = cp.array([1, 2, 3])
        _ = cp.sum(test_array)  # Force compilation
        cp.cuda.Device(0).compute_capability
        GPU_AVAILABLE = True
        logger.info("✅ GPU detectada y CuPy disponible")
    except Exception as e:
        # CuPy instalado pero CUDA no disponible (ej: falta CUDA Toolkit)
        GPU_AVAILABLE = False
        CUPY_AVAILABLE = False
        logger.warning(f"⚠️ CuPy instalado pero CUDA no disponible: {type(e).__name__}")
        logger.info("ℹ️ Usando CPU (NumPy) - Para usar GPU instala CUDA Toolkit")
        # Replace cp with numpy for fallback
        import numpy as cp # type: ignore
        
except ImportError:
    import numpy as cp  # Fallback a numpy # type: ignore
    CUPY_AVAILABLE = False
    GPU_AVAILABLE = False
    logger.info("ℹ️ CuPy no disponible, usando CPU (NumPy)")


class GPUConfig:
    """Global GPU configuration."""
    
    # Puede ser forzado por variable de entorno
    _force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    _enabled = GPU_AVAILABLE and not _force_cpu
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Returns True if GPU is enabled."""
        return cls._enabled
    
    @classmethod
    def enable(cls):
        """Enable GPU if available."""
        if GPU_AVAILABLE:
            cls._enabled = True
            logger.info("✅ GPU habilitada")
        else:
            logger.warning("⚠️ No se puede habilitar GPU: no disponible")
    
    @classmethod
    def disable(cls):
        """Disable GPU (force CPU)."""
        cls._enabled = False
        logger.info("ℹ️ GPU deshabilitada, usando CPU")
    
    @classmethod
    def get_backend(cls) -> str:
        """Returns the current backend."""
        if cls._enabled:
            return "GPU (CuPy)"
        return "CPU (NumPy)"
    
    @classmethod
    def get_device_info(cls) -> dict[str, Any]:
        """Device information."""
        if not cls._enabled or not GPU_AVAILABLE:
            return {
                "backend": "CPU",
                "device": "NumPy",
                "available": False
            }
        
        try:
            device = cp.cuda.Device(0)
            mem_info = device.mem_info
            compute_cap = device.compute_capability
            
            # Try to get device name (may not be available in all versions)
            device_name = "NVIDIA GPU"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                device_name = pynvml.nvmlDeviceGetName(handle).decode()
            except:
                pass  # If pynvml is not available, use generic name
            
            return {
                "backend": "GPU",
                "device": device_name,
                "compute_capability": compute_cap,
                "memory_total": f"{mem_info[1] / 1024**3:.2f} GB",
                "memory_free": f"{mem_info[0] / 1024**3:.2f} GB",
                "available": True
            }
        except Exception as e:
            logger.error(f"Error obteniendo info GPU: {e}")
            return {
                "backend": "GPU",
                "device": "Unknown",
                "error": str(e),
                "available": False
            }


def get_array_module(arr: Optional[ArrayLike] = None):
    """
    Returns the appropriate module (cupy or numpy) based on configuration.
    
    If an array is passed, it automatically detects its type.
    """
    if arr is not None:
        # Detect the array's module
        if CUPY_AVAILABLE and hasattr(arr, "__module__") and arr.__module__ == "cupy":
             return cp
        return np
    
    # Use global configuration
    if GPUConfig.is_enabled():
        return cp
    return np


def to_gpu(arr: ArrayLike) -> ArrayLike:
    """Move array to GPU if enabled."""
    if not GPUConfig.is_enabled():
        return arr
    
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr)
    return arr


def to_cpu(arr: ArrayLike) -> np.ndarray:
    """Move array to CPU (converts CuPy to NumPy)."""
    if CUPY_AVAILABLE and hasattr(arr, "device"): # Check for cupy array
        return cp.asnumpy(arr)
    return np.asarray(arr)


def ensure_cpu(arr: ArrayLike) -> np.ndarray:
    """Ensure the array is on CPU as NumPy."""
    return to_cpu(arr)


def ensure_gpu(arr: np.ndarray) -> ArrayLike:
    """Ensure the array is on GPU if enabled."""
    return to_gpu(arr)


class GPUArray:
    """Wrapper that handles GPU/CPU arrays transparently."""
    
    def __init__(self, data: ArrayLike):
        xp = get_array_module()
        if isinstance(data, np.ndarray) and GPUConfig.is_enabled():
            self._data = xp.asarray(data)
        else:
            self._data = data
        self._xp = get_array_module(self._data)
    
    @property
    def data(self) -> ArrayLike:
        """Returns the array (GPU or CPU)."""
        return self._data
    
    @property
    def xp(self):
        """Returns the module (cupy or numpy)."""
        return self._xp
    
    def to_cpu(self) -> np.ndarray:
        """Converts to NumPy on CPU."""
        return to_cpu(self._data)
    
    def to_gpu(self) -> "GPUArray":
        """Converts to CuPy on GPU."""
        if not GPUConfig.is_enabled():
            return self
        self._data = to_gpu(self._data)
        self._xp = get_array_module(self._data)
        return self
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def dtype(self):
        return self._data.dtype
    
    def __repr__(self):
        backend = "GPU" if self._xp.__name__ == "cupy" else "CPU"
        return f"GPUArray(shape={self.shape}, dtype={self.dtype}, backend={backend})"


# Aliases para compatibilidad
xp = get_array_module()  # Default module


def print_gpu_status():
    """Print GPU status."""
    info = GPUConfig.get_device_info()
    
    print("\n" + "="*70)
    print("🖥️  CONFIGURACIÓN GPU")
    print("="*70)
    
    if info["available"]:
        print(f"✅ Backend:   {info['backend']}")
        print(f"✅ Dispositivo: {info['device']}")
        if "compute_capability" in info:
            print(f"✅ Compute:   {info['compute_capability']}")
        if "memory_total" in info:
            print(f"✅ Memoria:   {info['memory_free']} / {info['memory_total']}")
    else:
        print(f"❌ Backend:   {info['backend']}")
        print(f"❌ GPU no disponible")
        if "error" in info:
            print(f"   Error: {info['error']}")
    
    print("="*70 + "\n")


# Log inicial
if GPU_AVAILABLE and GPUConfig.is_enabled():
    device_info = GPUConfig.get_device_info()
    logger.info(f"🚀 GPU inicializada: {device_info.get('device', 'Unknown')}")
elif CUPY_AVAILABLE:
    logger.info("⚠️ CuPy disponible pero GPU no detectada")
else:
    logger.info("ℹ️ Usando CPU (CuPy no instalado)")

#!/usr/bin/env python3
"""
Script de desarrollo para uvicorn con hot reload mejorado.
Configurado para Windows con watchfiles.
"""
import os
import sys
from pathlib import Path

import uvicorn

# Obtener el directorio base del proyecto (donde está este script)
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"

# Cambiar al directorio del backend para que los imports funcionen
os.chdir(BASE_DIR)

# En Windows, usar rutas absolutas para mejor compatibilidad
reload_dirs = [str(SRC_DIR.resolve())]

# Agregar el directorio actual también para capturar cambios en dev.py
reload_dirs.append(str(BASE_DIR.resolve()))

if __name__ == "__main__":
    print(f"[INFO] Observando cambios en: {reload_dirs}")
    print(f"[INFO] Directorio de trabajo: {os.getcwd()}")
    print(f"[INFO] Directorio base: {BASE_DIR}")

    # Configuración para desarrollo con hot reload mejorado
    uvicorn.run(
        "src.api.rest:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=reload_dirs,
        reload_includes=["*.py"],
        reload_excludes=["*.pyc", "__pycache__", "*.pyo", "*.pyd", "*.egg-info"],
        log_level="info",
        reload_delay=0.25,  # Delay más corto para respuesta más rápida
    )

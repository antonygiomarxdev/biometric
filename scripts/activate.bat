@echo off
REM Script de activación del entorno virtual para CMD
REM Uso: activate.bat

echo 🔧 Configurando entorno virtual con uv...

REM Verificar si uv está instalado
where uv >nul 2>&1
if errorlevel 1 (
    echo ❌ uv no está instalado. Instálalo con:
    echo    pip install uv
    echo    o visita: https://github.com/astral-sh/uv
    exit /b 1
)

REM Sincronizar dependencias (crea .venv si no existe)
echo 📦 Sincronizando dependencias...
call uv sync

REM Verificar que .venv existe
if not exist ".venv" (
    echo ❌ No se pudo crear el entorno virtual
    exit /b 1
)

REM Activar el entorno virtual
if exist ".venv\Scripts\activate.bat" (
    echo ✅ Activando entorno virtual...
    call .venv\Scripts\activate.bat
    
    echo.
    echo 🎉 Entorno virtual activado!
    echo    Python: 
    python --version
    echo.
) else (
    echo ❌ Script de activación no encontrado
    exit /b 1
)

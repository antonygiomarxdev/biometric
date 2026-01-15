@echo off
echo ===================================================
echo   BioSecure Gov - Sistema Biometrico
echo ===================================================
echo.
echo Seleccione una opcion:
echo 1. Iniciar Todo con Docker (Recomendado)
echo 2. Iniciar Solo Backend (Local)
echo 3. Iniciar Solo Frontend (Local)
echo 4. Ejecutar Tests
echo 5. Generar Cliente API (Frontend)
echo 6. Limpiar Archivos Temporales
echo.

set /p op=Opcion: 

if "%op%"=="1" goto docker
if "%op%"=="2" goto backend
if "%op%"=="3" goto frontend
if "%op%"=="4" goto tests
if "%op%"=="5" goto clientgen
if "%op%"=="6" goto clean

goto end

:docker
echo Iniciando con Docker Compose...
docker-compose up --build
goto end

:backend
echo Iniciando Backend Local...
cd apps\backend
uv run uvicorn src.api.rest:app --reload --host 0.0.0.0 --port 8000
goto end

:frontend
echo Iniciando Frontend Local...
cd apps\frontend
npm run dev
goto end

:tests
echo Ejecutando Tests...
cd apps\backend
uv run pytest
goto end

:clientgen
echo Generando Cliente API...
cd apps\backend
uv run python export_openapi.py
move openapi.json ..\frontend\
cd ..\frontend
npm run gen:client
goto end

:clean
echo Limpiando...
docker-compose down
cd apps\backend
rmdir /s /q __pycache__ 2>nul
cd ..\..
echo Limpieza completada.
goto end

:end
pause

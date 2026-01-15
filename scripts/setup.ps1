# Script de setup inicial del proyecto
# Ejecuta: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "🚀 CONFIGURACIÓN INICIAL DEL PROYECTO" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# 1. Verificar uv
Write-Host "1️⃣ Verificando uv..." -ForegroundColor Yellow
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ uv no está instalado" -ForegroundColor Red
    Write-Host ""
    Write-Host "Instala uv con uno de estos métodos:" -ForegroundColor Yellow
    Write-Host "  • pip install uv" -ForegroundColor White
    Write-Host "  • winget install astral-sh.uv" -ForegroundColor White
    Write-Host "  • https://github.com/astral-sh/uv#installation" -ForegroundColor White
    exit 1
}

$uvVersion = uv --version
Write-Host "✅ uv instalado: $uvVersion" -ForegroundColor Green

# 2. Sincronizar dependencias
Write-Host ""
Write-Host "2️⃣ Sincronizando dependencias..." -ForegroundColor Yellow
uv sync

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Error sincronizando dependencias" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dependencias instaladas" -ForegroundColor Green

# 3. Verificar Python
Write-Host ""
Write-Host "3️⃣ Verificando Python..." -ForegroundColor Yellow
$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $pythonPath) {
    $pythonVersion = & $pythonPath --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python no encontrado en el entorno virtual" -ForegroundColor Red
    exit 1
}

# 4. Verificar PostgreSQL (opcional)
Write-Host ""
Write-Host "4️⃣ Verificando PostgreSQL..." -ForegroundColor Yellow
$pgRunning = docker ps 2>$null | Select-String -Pattern "postgres" -Quiet
if ($pgRunning) {
    Write-Host "✅ PostgreSQL corriendo en Docker" -ForegroundColor Green
} else {
    Write-Host "⚠️  PostgreSQL no está corriendo" -ForegroundColor Yellow
    Write-Host "   Para iniciarlo: docker-compose -f docker-compose.dev.yml up -d postgres" -ForegroundColor Gray
}

# 5. Mostrar instrucciones
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "✅ CONFIGURACIÓN COMPLETA" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""
Write-Host "📝 Próximos pasos:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Activar el entorno virtual:" -ForegroundColor White
Write-Host "   PowerShell: . .\activate.ps1" -ForegroundColor Gray
Write-Host "   CMD:        activate.bat" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Iniciar PostgreSQL (si no está corriendo):" -ForegroundColor White
Write-Host "   docker-compose -f docker-compose.dev.yml up -d postgres" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Probar el sistema:" -ForegroundColor White
Write-Host "   python test_socofing.py" -ForegroundColor Gray
Write-Host ""
Write-Host "💡 En VS Code/Cursor, el entorno se activa automáticamente" -ForegroundColor Cyan
Write-Host ""

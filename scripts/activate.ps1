# Script de activación del entorno virtual para PowerShell
# Uso: . .\activate.ps1

$ErrorActionPreference = "Stop"

Write-Host "🔧 Configurando entorno virtual con uv..." -ForegroundColor Cyan

# Verificar si uv está instalado
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ uv no está instalado. Instálalo con:" -ForegroundColor Red
    Write-Host "   pip install uv" -ForegroundColor Yellow
    Write-Host "   o visita: https://github.com/astral-sh/uv" -ForegroundColor Yellow
    exit 1
}

# Sincronizar dependencias (crea .venv si no existe)
Write-Host "📦 Sincronizando dependencias..." -ForegroundColor Cyan
uv sync

# Verificar que .venv existe
$venvPath = Join-Path $PSScriptRoot ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "❌ No se pudo crear el entorno virtual" -ForegroundColor Red
    exit 1
}

# Activar el entorno virtual
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    Write-Host "✅ Activando entorno virtual..." -ForegroundColor Green
    & $activateScript
    
    Write-Host ""
    Write-Host "🎉 Entorno virtual activado!" -ForegroundColor Green
    Write-Host "   Python: $(python --version)" -ForegroundColor Gray
    Write-Host "   Ubicación: $venvPath" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "❌ Script de activación no encontrado en: $activateScript" -ForegroundColor Red
    exit 1
}

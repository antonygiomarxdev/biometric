# Script de inicio rápido para el sistema biométrico (PowerShell)

Write-Host "🚀 Iniciando Sistema Biométrico de Huellas Dactilares" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Verificar dependencias
Write-Host ""
Write-Host "📦 Verificando dependencias..." -ForegroundColor Yellow

$dockerInstalled = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerInstalled) {
    Write-Host "❌ Docker no está instalado. Por favor instalar Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Docker encontrado" -ForegroundColor Green

# Crear directorio de datos
Write-Host ""
Write-Host "📁 Creando directorio de datos..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data" | Out-Null
Write-Host "✅ Directorio creado" -ForegroundColor Green

# Modo de ejecución
Write-Host ""
Write-Host "Selecciona el modo de ejecución:" -ForegroundColor Cyan
Write-Host "  1) Development (con hot-reload)"
Write-Host "  2) Production"
$mode = Read-Host "Opción [1]"
if ([string]::IsNullOrWhiteSpace($mode)) { $mode = "1" }

if ($mode -eq "1") {
    Write-Host ""
    Write-Host "🔧 Iniciando en modo DEVELOPMENT..." -ForegroundColor Yellow
    docker-compose -f docker-compose.dev.yml up --build
}
elseif ($mode -eq "2") {
    Write-Host ""
    Write-Host "🏭 Iniciando en modo PRODUCTION..." -ForegroundColor Yellow
    
    # Solicitar credenciales de DB
    $db_user = Read-Host "Usuario DB [postgres]"
    if ([string]::IsNullOrWhiteSpace($db_user)) { $db_user = "postgres" }
    
    $db_password = Read-Host "Password DB [postgres]" -AsSecureString
    $db_password_plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($db_password))
    if ([string]::IsNullOrWhiteSpace($db_password_plain)) { $db_password_plain = "postgres" }
    
    $env:DB_USER = $db_user
    $env:DB_PASSWORD = $db_password_plain
    
    docker-compose -f docker-compose.prod.yml up -d --build
    
    Write-Host ""
    Write-Host "✅ Servicios iniciados en background" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ver logs: docker-compose -f docker-compose.prod.yml logs -f" -ForegroundColor Cyan
    Write-Host "Detener: docker-compose -f docker-compose.prod.yml down" -ForegroundColor Cyan
}
else {
    Write-Host "❌ Opción inválida" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "✅ Sistema iniciado exitosamente" -ForegroundColor Green
Write-Host ""
Write-Host "📍 API disponible en: http://localhost:8000" -ForegroundColor Cyan
Write-Host "📖 Documentación: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "🧪 Para probar el CLI:" -ForegroundColor Yellow
Write-Host "   docker-compose exec api python -m src.api.cli --help" -ForegroundColor White

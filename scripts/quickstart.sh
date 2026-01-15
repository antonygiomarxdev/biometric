#!/bin/bash
# Script de inicio rápido para el sistema biométrico

set -e

echo "🚀 Iniciando Sistema Biométrico de Huellas Dactilares"
echo "=================================================="

# Verificar dependencias
echo ""
echo "📦 Verificando dependencias..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado. Por favor instalar Docker."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado. Por favor instalar Docker Compose."
    exit 1
fi

echo "✅ Docker y Docker Compose encontrados"

# Crear directorio de datos
echo ""
echo "📁 Creando directorio de datos..."
mkdir -p data
echo "✅ Directorio creado"

# Modo de ejecución
echo ""
echo "Selecciona el modo de ejecución:"
echo "  1) Development (con hot-reload)"
echo "  2) Production"
read -p "Opción [1]: " mode
mode=${mode:-1}

if [ "$mode" = "1" ]; then
    echo ""
    echo "🔧 Iniciando en modo DEVELOPMENT..."
    docker-compose -f docker-compose.dev.yml up --build
elif [ "$mode" = "2" ]; then
    echo ""
    echo "🏭 Iniciando en modo PRODUCTION..."
    
    # Solicitar credenciales de DB
    read -p "Usuario DB [postgres]: " db_user
    db_user=${db_user:-postgres}
    
    read -sp "Password DB [postgres]: " db_password
    db_password=${db_password:-postgres}
    echo ""
    
    export DB_USER=$db_user
    export DB_PASSWORD=$db_password
    
    docker-compose -f docker-compose.prod.yml up -d --build
    
    echo ""
    echo "✅ Servicios iniciados en background"
    echo ""
    echo "Ver logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "Detener: docker-compose -f docker-compose.prod.yml down"
else
    echo "❌ Opción inválida"
    exit 1
fi

echo ""
echo "✅ Sistema iniciado exitosamente"
echo ""
echo "📍 API disponible en: http://localhost:8000"
echo "📖 Documentación: http://localhost:8000/docs"
echo ""
echo "🧪 Para probar el CLI:"
echo "   docker-compose exec api python -m src.api.cli --help"

# BioSecure Gov - Sistema Biométrico de Huellas Dactilares

Este repositorio contiene un sistema completo de identificación biométrica diseñado para uso gubernamental y forense. Implementa una arquitectura moderna de monorepo con separación clara entre backend (Python/FastAPI) y frontend (React/TypeScript).

## Estructura del Proyecto

El proyecto está organizado como un monorepo:

- **`apps/backend/`**: API REST en Python con FastAPI. Contiene toda la lógica de procesamiento de huellas, extracción de minucias y algoritmos de coincidencia (matching).
- **`apps/frontend/`**: Aplicación web moderna en React con TypeScript y Tailwind CSS. Proporciona una interfaz visual para escaneo, registro e identificación.
- **`data/`**: Directorio compartido para almacenamiento de datos (imágenes de huellas, etc.).
- **`docs/`**: Documentación técnica detallada del proyecto.

## Requisitos Previos

- Docker y Docker Compose
- Python 3.12+ (para desarrollo local del backend)
- Node.js 20+ (para desarrollo local del frontend)
- `uv` (gestor de paquetes de Python)

## Inicio Rápido

### Windows (Recomendado)
Simplemente ejecute el script interactivo:
```cmd
run.bat
```
Este script le permitirá iniciar todo el sistema con Docker, o ejecutar componentes individuales localmente.

### Docker Manual
```bash
make docker-up
# O:
docker-compose up --build
```

Esto levantará:
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Base de Datos**: PostgreSQL con pgvector (puerto 5434)

## Desarrollo Local

### Backend

```bash
make install       # Instalar dependencias
make api           # Iniciar servidor de desarrollo
make test          # Ejecutar pruebas
```

### Frontend

```bash
make frontend-install # Instalar dependencias
make frontend         # Iniciar servidor de desarrollo
```

## Documentación

Consulte el directorio `docs/` para guías detalladas:
- `QUICKSTART.md`: Guía de inicio rápido.
- `ARCHITECTURE.md`: Detalles de la arquitectura del sistema.
- `GPU_SETUP.md`: Configuración para aceleración por GPU.

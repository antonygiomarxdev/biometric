# 🚀 Guía de Inicio Rápido

## Instalación con uv (Recomendado)

### 1. Instalar uv
```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clonar y configurar proyecto
```bash
git clone <repo-url>
cd biometric

# Crear entorno virtual y instalar dependencias
uv venv
source .venv/bin/activate  # Linux/macOS
# o
.venv\Scripts\activate     # Windows

# Instalar dependencias
uv pip install -e .

# Dependencias de desarrollo
uv pip install -e ".[dev]"
```

### 3. Configurar base de datos

#### Opción A: PostgreSQL local
```bash
# Instalar PostgreSQL 15+
# Crear base de datos
createdb fingerprint

# Configurar conexión
cp .env.example .env
# Editar .env con tus credenciales

# Inicializar tablas
python -m src.api.cli init-db
```

#### Opción B: Docker (Más fácil)
```bash
# Iniciar PostgreSQL + API
make docker-dev

# O manualmente
docker-compose -f docker-compose.dev.yml up
```

## Uso Rápido

### CLI

```bash
# Extraer minutiae de una imagen
python -m src.api.cli extract /path/to/fingerprint.bmp

# Registrar una huella
python -m src.api.cli register /path/to/fingerprint.bmp \
  --person-id P001 \
  --name "Juan Pérez" \
  --document 12345678

# Identificar una huella
python -m src.api.cli identify /path/to/fingerprint.bmp

# Ver estado del sistema
python -m src.api.cli status
```

### API REST

```bash
# Iniciar servidor
make api
# O
uvicorn src.api.rest:app --reload

# La API estará en http://localhost:8000
# Documentación en http://localhost:8000/docs
```

#### Ejemplos de requests

```bash
# Extraer
curl -X POST http://localhost:8000/extract \
  -F "file=@/path/to/fingerprint.bmp"

# Registrar
curl -X POST http://localhost:8000/register \
  -F "file=@/path/to/fingerprint.bmp" \
  -F "person_id=P001" \
  -F "name=Juan Pérez" \
  -F "document=12345678"

# Identificar
curl -X POST http://localhost:8000/identify \
  -F "file=@/path/to/fingerprint.bmp"
```

## Testing

```bash
# Todos los tests
make test

# Con cobertura
make test-cov

# Solo performance benchmarks
make test-perf
```

## Comandos útiles (Makefile)

```bash
make help          # Ver todos los comandos disponibles
make install       # Instalar dependencias
make dev          # Instalar deps de desarrollo
make lint         # Ejecutar linter
make format       # Formatear código
make docker-dev   # Docker development
make docker-prod  # Docker production
make clean        # Limpiar archivos temporales
```

## Troubleshooting

### Error: "Command not found: python"
Usa `python3` en lugar de `python`

### Error: "Module not found"
```bash
# Reinstalar dependencias
uv pip install -e .
```

### Error de conexión a PostgreSQL
```bash
# Verificar que PostgreSQL está corriendo
# Docker
docker ps | grep postgres

# Local (Linux)
sudo systemctl status postgresql
```

### Performance lento
- Verifica que FAISS está instalado correctamente
- Usa índice IVF para >10K huellas (cambiar en config.py)
- Considera usar máquina con más CPU/RAM

## Próximos pasos

1. Lee [README.md](README.md) para documentación completa
2. Revisa [ARCHITECTURE.md](ARCHITECTURE.md) para entender el diseño
3. Explora la API en http://localhost:8000/docs
4. Ejecuta los tests de performance: `make test-perf`

## Obtener imágenes de prueba

Dataset recomendado: [SOCOFing](https://www.kaggle.com/datasets/ruizgara/socofing)

```bash
# Descargar y extraer en ./data/SOCOFing/
```

¡Listo para usar! 🎉

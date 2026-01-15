# Guía de Contribución

## Configuración de Desarrollo

### Opción 1: Con uv (Recomendado)

```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sincronizar dependencias
uv sync

# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Ejecutar tests
uv run pytest

# Ejecutar linter
uv run ruff check src/

# Ejecutar type checker
uv run mypy src/
```

### Opción 2: Con pip tradicional

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
pip install -e ".[dev]"

# Ejecutar tests
pytest

# Linter
ruff check src/

# Type checker
mypy src/
```

## Estructura del Proyecto

```
src/
├── core/           # Configuración, modelos, métricas
├── processing/     # Procesamiento de imágenes
├── storage/        # Base de datos y vectores
├── services/       # Lógica de negocio
└── api/           # REST y CLI

tests/             # Tests unitarios e integración
migrations/        # Migraciones SQL
```

## Flujo de Trabajo

### 1. Crear Branch

```bash
git checkout -b feature/descripcion-breve
```

### 2. Desarrollo

```bash
# Ejecutar API en desarrollo
uvicorn src.api.rest:app --reload

# O con Docker
docker-compose -f docker-compose.dev.yml up
```

### 3. Tests

```bash
# Tests unitarios
pytest tests/test_*.py -v

# Tests de integración
pytest tests/test_integration.py -v

# Tests de performance
pytest tests/test_performance.py -v

# Con cobertura
pytest --cov=src --cov-report=html
```

### 4. Code Quality

```bash
# Formatear código
ruff check --fix src/

# Verificar tipos
mypy src/

# Pre-commit manual
pytest && ruff check src/ && mypy src/
```

### 5. Commit

```bash
# Formato de commits
git commit -m "tipo: descripción breve

Explicación detallada (opcional)
"

# Tipos válidos:
# feat: Nueva funcionalidad
# fix: Corrección de bug
# perf: Mejora de performance
# refactor: Refactorización
# test: Añadir tests
# docs: Documentación
# chore: Tareas de mantenimiento
```

### 6. Push y PR

```bash
git push origin feature/descripcion-breve

# Crear PR en GitHub/GitLab con descripción clara
```

## Guidelines de Código

### Python Style

- PEP 8 compliance
- Type hints obligatorios en funciones públicas
- Docstrings en formato Google
- Líneas máximo 100 caracteres

```python
def process_fingerprint(
    image: np.ndarray,
    fingerprint_id: str = None
) -> Fingerprint:
    """Procesa una imagen de huella completa.
    
    Args:
        image: Imagen en escala de grises
        fingerprint_id: ID opcional para la huella
        
    Returns:
        Objeto Fingerprint con minutiae extraídas
        
    Raises:
        ValueError: Si la imagen es inválida
    """
    pass
```

### Tests

- Cobertura mínima: 80%
- Tests unitarios rápidos (<100ms cada uno)
- Tests de integración aislados
- Fixtures reutilizables en `conftest.py`

```python
def test_feature_description():
    """Test claro y descriptivo."""
    # Arrange
    input_data = create_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.expected_property == expected_value
```

### Performance

- Funciones críticas deben tener decorador `@timed`
- Evitar loops innecesarios
- Usar NumPy para operaciones vectoriales
- Perfilar antes de optimizar

```python
@timed("critical_operation")
def critical_function():
    # Implementation
    pass
```

## Añadir Nueva Funcionalidad

### 1. Añadir nuevo endpoint REST

```python
# src/api/rest.py

@app.post("/new_endpoint", response_model=ResponseModel)
async def new_endpoint(input: InputModel):
    """Descripción del endpoint."""
    try:
        result = service.do_something(input)
        return ResponseModel(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Añadir comando CLI

```python
# src/api/cli.py

@cli.command()
@click.argument("param")
@click.option("--flag", help="Descripción")
def new_command(param, flag):
    """Descripción del comando."""
    try:
        result = service.do_something(param, flag)
        click.echo(f"✓ Resultado: {result}")
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
```

### 3. Añadir nuevo servicio

```python
# src/services/new_service.py

class NewService:
    """Descripción del servicio."""
    
    def __init__(self, dependency):
        self.dependency = dependency
    
    @timed("service_operation")
    def do_something(self, input: InputType) -> OutputType:
        """Método principal."""
        # Implementation
        pass

# Instancia global
new_service = NewService()
```

## Debugging

### Local

```python
# Añadir breakpoints
import pdb; pdb.set_trace()

# O con ipdb (más features)
import ipdb; ipdb.set_trace()
```

### Docker

```bash
# Ver logs
docker-compose logs -f api

# Ejecutar bash en contenedor
docker-compose exec api bash

# Ejecutar comandos Python
docker-compose exec api python -c "from src.core.config import config; print(config.database_url)"
```

## Troubleshooting Común

### Problema: Tests fallan con error de DB

```bash
# Solución: Resetear DB de tests
rm -f test.db
pytest
```

### Problema: Performance lento

```bash
# Perfilar código
python -m cProfile -o profile.stats -m pytest tests/test_performance.py
python -m pstats profile.stats
```

### Problema: Imports no funcionan

```bash
# Instalar en modo editable
pip install -e .

# O con uv
uv pip install -e .
```

## Recursos

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/)
- [pgvector Docs](https://github.com/pgvector/pgvector)
- [pytest Docs](https://docs.pytest.org/)

## Preguntas

Si tienes preguntas o encuentras bugs:

1. Revisa [Issues](../../issues) existentes
2. Crea un nuevo Issue con detalles
3. Contacta al equipo

¡Gracias por contribuir! 🚀

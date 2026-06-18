.PHONY: help install dev test lint format clean docker-up docker-down docker-build api frontend client-gen
help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
install: ## Instala todas las dependencias (backend + frontend + turbo)
	npm install
	cd apps/backend && uv sync
	cd apps/frontend && npm install
install-backend: ## Instala dependencias del backend
	cd apps/backend && uv sync
install-frontend: ## Instala dependencias del frontend
	cd apps/frontend && npm install
dev: ## Inicia back y front en paralelo con Turborepo (recomendado)
	npm run dev
dev-backend: ## Inicia solo el backend en modo desarrollo
	cd apps/backend && uv run uvicorn src.api.rest:app --reload --host 0.0.0.0 --port 8000
dev-frontend: ## Inicia solo el frontend en modo desarrollo
	cd apps/frontend && npm run dev
test: ## Ejecuta los tests del backend
	cd apps/backend && uv run pytest
test-cov: ## Ejecuta tests con cobertura (backend)
	cd apps/backend && uv run pytest --cov=src --cov-report=html --cov-report=term
lint: ## Ejecuta linter (backend)
	cd apps/backend && uv run ruff check src/ tests/
format: ## Formatea el código (backend)
	cd apps/backend && uv run ruff format src/ tests/
type-check: ## Verifica tipos con mypy + pyright (backend)
	cd apps/backend && uv run mypy src/ && uv run pyright src/
check: ## Lint + type-check + test (backend)
	cd apps/backend && uv run ruff check src/ && uv run mypy src/ && uv run pyright src/ && uv run pytest
install-hooks: ## Instala pre-commit hooks
	cd apps/backend && uv tool install pre-commit && pre-commit install
docker-up: ## Inicia todo el sistema en Docker (Backend + Frontend + DB + MinIO)
	docker compose up
docker-up-detached: ## Inicia todo el sistema en Docker en segundo plano
	docker compose up -d
docker-down: ## Detiene los contenedores
	docker compose down
docker-build: ## Reconstruye las imágenes Docker
	docker compose build
docker-logs: ## Muestra los logs de los contenedores
	docker compose logs -f
clean: ## Limpia archivos temporales
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -f .coverage
api: ## Inicia API en modo desarrollo local (sin docker)
	cd apps/backend && uv run uvicorn src.api.rest:app --reload --host 0.0.0.0 --port 8000
frontend: ## Inicia Frontend en modo desarrollo local
	cd apps/frontend && npm run dev
client-gen: ## Genera cliente API para el Frontend (OpenAPI)
	cd apps/backend && uv run python export_openapi.py
	mv apps/backend/openapi.json apps/frontend/
	cd apps/frontend && npm run gen:client

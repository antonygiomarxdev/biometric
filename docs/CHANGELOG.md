# Changelog

Todos los cambios notables de este proyecto serán documentados aquí.

## [1.0.0] - 2026-01-12

### ✨ Añadido
- Sistema completo de extracción de minutiae usando OpenCV
- Comparación e identificación de huellas con FAISS
- API REST completa con FastAPI
- CLI para operaciones batch y testing
- Base de datos PostgreSQL con SQLAlchemy ORM
- Índice vectorial FAISS para búsqueda rápida
- Sistema de métricas y performance monitoring
- Tests unitarios y de integración
- Benchmarks de performance
- Docker Compose para dev y prod
- Documentación completa (README, ARCHITECTURE, QUICKSTART)
- Soporte para uv package manager
- Makefile con comandos útiles

### 🏗️ Arquitectura
- Monolito modular con separación clara de responsabilidades
- Módulos: core, processing, storage, services, api
- Sin terminología DDD innecesaria
- Enfocado en Clean Code y performance

### 📊 Performance
- Extracción: ~150-300ms por imagen
- Comparación: ~10-50ms por búsqueda
- Throughput: 20 img/s para extracción

### 🔧 Configuración
- Variables de entorno para dev/prod
- Configuración centralizada en core/config.py
- Parámetros ajustables para diferentes casos de uso

### 📝 Documentación
- README.md completo con ejemplos
- ARCHITECTURE.md con diseño detallado
- QUICKSTART.md para inicio rápido
- Comentarios en código siguiendo Clean Code
- Swagger UI automático en /docs

### 🧪 Testing
- Tests unitarios para todos los módulos
- Tests de integración del pipeline completo
- Benchmarks de performance
- Fixtures y configuración de pytest
- Cobertura de código configurada

### 🐳 Docker
- Dockerfile optimizado multi-stage
- docker-compose.dev.yml para desarrollo
- docker-compose.prod.yml para producción
- Health checks y restart policies
- Volúmenes persistentes

### 📦 Dependencias Principales
- Python 3.12+
- FastAPI 0.104+
- SQLAlchemy 2.0+
- FAISS 1.7.4+
- OpenCV 4.8+
- NumPy 1.24+
- PostgreSQL 15+

## [Unreleased]

### 🔮 Planeado (Post-MVP)
- GPU acceleration con FAISS-GPU
- Compilación con Numba/Cython para loops críticos
- Cache distribuido con Redis
- Procesamiento batch paralelo
- Compresión de vectores con PQ
- Sharding de índices por categorías
- API de webhooks para notificaciones
- Dashboard de métricas en tiempo real
- Multi-tenancy y aislamiento de datos
- Exportación de reportes forenses

---

Formato basado en [Keep a Changelog](https://keepachangelog.com/)

# Etapa base para dependencias comunes
FROM python:3.12-slim AS base

WORKDIR /app

# Instalar dependencias del sistema mínimas y necesarias para OpenCV
# En Debian Trixie (13+), libgl1-mesa-glx fue reemplazado por libgl1
# libgl1 y libglib2.0-0 son necesarios para cv2-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Etapa de construcción de dependencias
FROM base AS builder

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Etapa final optimizada para producción
FROM base AS final

# Copiar paquetes instalados desde builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copiar código fuente
COPY . .

# Variables de entorno para optimización
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Configuración para soportar CUDA si está disponible en el host (Runtime)
# Nota: Para soporte real de GPU, se requiere usar la imagen base nvidia/cuda
# pero mantenemos python-slim para CPU/Hybrid compatible.

# Crear directorios necesarios
RUN mkdir -p /data/logs

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Usar un usuario no root por seguridad (opcional, ajustado para evitar problemas de permisos con /data)
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

CMD ["uvicorn", "src.api.rest:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

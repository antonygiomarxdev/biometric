"""Configuración centralizada del sistema con soporte robusto para .env."""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

# Intentar cargar .env para desarrollo local
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass(frozen=True)
class Config:
    """Configuración global de la aplicación (Inmutable)."""
    
    # Environment
    env: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    
    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5434/fingerprint"
        )
    )
    db_pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    db_max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")))
    
    # Vector Index (pgvector)
    vector_dimension: int = field(default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "256")))
    vector_index_lists: int = field(default_factory=lambda: int(os.getenv("VECTOR_INDEX_LISTS", "100")))
    
    # Processing
    image_resize_width: int = field(default_factory=lambda: int(os.getenv("IMAGE_RESIZE_WIDTH", "350")))
    enhancement_enabled: bool = field(default_factory=lambda: os.getenv("ENHANCEMENT_ENABLED", "true").lower() == "true")
    
    # Performance & Concurrency
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "8")))
    num_workers: int = field(default_factory=lambda: int(os.getenv("NUM_WORKERS", "0"))) # 0 = Auto-detect
    force_cpu: bool = field(default_factory=lambda: os.getenv("FORCE_CPU", "0") == "1")
    
    # Comparison Logic
    match_threshold: float = field(default_factory=lambda: float(os.getenv("MATCH_THRESHOLD", "2000.0")))
    top_k_matches: int = field(default_factory=lambda: int(os.getenv("TOP_K_MATCHES", "5")))
    combined_score_weight_l2: float = field(default_factory=lambda: float(os.getenv("WEIGHT_L2", "0.7")))
    combined_score_weight_cos: float = field(default_factory=lambda: float(os.getenv("WEIGHT_COS", "0.3")))
    
    # Logging & Metrics
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    enable_metrics: bool = field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == "true")

    # MinIO / Object Storage
    minio_endpoint: str = field(default_factory=lambda: os.getenv("MINIO_ENDPOINT", "localhost:9000"))
    minio_access_key: str = field(default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
    minio_secret_key: str = field(default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "minioadmin"))
    minio_bucket: str = field(default_factory=lambda: os.getenv("MINIO_BUCKET", "fingerprints"))
    minio_secure: bool = field(default_factory=lambda: os.getenv("MINIO_SECURE", "false").lower() == "true")

    # Authentication / JWT
    jwt_secret_key: str = field(
        default_factory=lambda: os.getenv(
            "JWT_SECRET_KEY",
            "change-me-in-production-use-a-real-secret-key-32-chars-min",
        )
    )
    jwt_algorithm: str = field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256")
    )
    jwt_access_token_expire_minutes: int = field(
        default_factory=lambda: int(
            os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        )
    )

    def __post_init__(self):
        """Validaciones post-inicialización."""
        if self.combined_score_weight_l2 + self.combined_score_weight_cos != 1.0:
            # Normalizar si no suman 1 (opcional, o lanzar warning)
            pass

# Instancia global de configuración
config = Config()

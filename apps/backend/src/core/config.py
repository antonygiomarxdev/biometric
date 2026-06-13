"""Centralized system configuration with robust .env support."""

import os
from dataclasses import dataclass, field
from typing import Literal, Optional
import logging

from pydantic import SecretStr

logging.basicConfig(level=logging.INFO)

# Try to load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed, skipping .env loading. Make sure to set environment variables in production.")    

@dataclass(frozen=True)
class JurisdictionConfig:
    """
    Global extensibility configuration for legal frameworks.
    Allows the system to adapt to different countries' legal reporting standards.
    """
    country: str = field(default_factory=lambda: os.getenv("JURISDICTION_COUNTRY", "República de Nicaragua"))
    expert_title: str = field(default_factory=lambda: os.getenv("JURISDICTION_EXPERT_TITLE", "Perito Forense"))
    legal_framework: str = field(default_factory=lambda: os.getenv("JURISDICTION_LEGAL_FRAMEWORK", "Código Procesal Penal"))

@dataclass(frozen=True)
class Config:
    """Global application configuration (Immutable)."""
    
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
    
    # AI / ML
    ai_model_dir: str = field(default_factory=lambda: os.getenv("AI_MODEL_DIR", "data/models/"))
    ai_use_gpu: bool = field(default_factory=lambda: os.getenv("AI_USE_GPU", "true").lower() == "true")
    ai_gpu_device_id: int = field(default_factory=lambda: int(os.getenv("AI_GPU_DEVICE_ID", "0")))
    ai_input_size: int = field(default_factory=lambda: int(os.getenv("AI_INPUT_SIZE", "512")))
    ai_confidence_threshold: float = field(default_factory=lambda: float(os.getenv("AI_CONFIDENCE_THRESH", "0.5")))

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

    # Compliance / Privacy Strategy
    compliance_strategy: str = field(
        default_factory=lambda: os.getenv(
            "COMPLIANCE_STRATEGY", "base"
        )
    )

    # Storage Encryption (Fernet/AES-256)
    storage_encryption_key: str = field(
        default_factory=lambda: os.getenv(
            "STORAGE_ENCRYPTION_KEY", ""
        )
    )

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

    # Generative AI / LLM
    llm_provider: Literal["local", "openai"] = field(
        default_factory=lambda: os.getenv(
            "LLM_PROVIDER", "local"  # type: ignore[return-value]
        )
    )
    local_model_name: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_MODEL_NAME", "llama3.1:latest"
        )
    )
    remote_model_name: str = field(
        default_factory=lambda: os.getenv(
            "REMOTE_MODEL_NAME", "gpt-4"
        )
    )
    # AI Tracing (OpenTelemetry / Arize Phoenix)
    enable_ai_tracing: bool = field(
        default_factory=lambda: os.getenv("ENABLE_AI_TRACING", "true").lower() in ("true", "1", "yes")
    )

    openai_api_key: SecretStr = field(
        default_factory=lambda: SecretStr(
            os.getenv("OPENAI_API_KEY", "")
        )
    )

    def __post_init__(self):
        """Post-initialization validations."""
        if self.combined_score_weight_l2 + self.combined_score_weight_cos != 1.0:
            # Normalize if they don't sum to 1 (optional, or raise warning)
            pass

# Global configuration instance
config = Config()

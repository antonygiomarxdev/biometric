"""Centralized system configuration with robust .env support."""

import logging
import os
from dataclasses import dataclass, field
from typing import Literal  # noqa: F401  (kept for backwards-compat)

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
class PipelineConfig:
    """Configuration for the canonical processing chain.

    Each parameter corresponds to a tunable in :func:`build_production_pipeline`.
    Override via env vars for operator tuning without code changes.
    """
    # Skeletonization
    skeleton_min_island_size: int = field(
        default_factory=lambda: int(os.getenv("PIPELINE_SKELETON_MIN_ISLAND_SIZE", "20"))
    )
    # Singularity detection (Poincaré + DORIC)
    singularity_roi_radius: int = field(
        default_factory=lambda: int(os.getenv("PIPELINE_SINGULARITY_ROI_RADIUS", "140"))
    )
    # Fusion filter (cross-extractor consensus)
    fusion_radius: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_FUSION_RADIUS", "8.0"))
    )
    fusion_min_votes: int = field(
        default_factory=lambda: int(os.getenv("PIPELINE_FUSION_MIN_VOTES", "2"))
    )
    # Post-hooks cleanup
    spur_max_distance: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_SPUR_MAX_DISTANCE", "10.0"))
    )
    healer_max_distance: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_HEALER_MAX_DISTANCE", "8.0"))
    )
    border_px: int = field(
        default_factory=lambda: int(os.getenv("PIPELINE_BORDER_PX", "0"))
    )
    border_roi_mode: str = field(
        default_factory=lambda: os.getenv("PIPELINE_BORDER_ROI_MODE", "core")
    )
    # Orientation refiner
    orientation_window: int = field(
        default_factory=lambda: int(os.getenv("PIPELINE_ORIENTATION_WINDOW", "16"))
    )
    orientation_coherence_threshold: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_ORIENTATION_COHERENCE", "0.65"))
    )
    # Confidence filter
    low_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("PIPELINE_LOW_CONFIDENCE", "0.75"))
    )


@dataclass(frozen=True)
class GaborConfig:
    """Gabor enhancement parameters."""
    block_size: int = field(
        default_factory=lambda: int(os.getenv("GABOR_BLOCK_SIZE", "16"))
    )
    window_length: int = field(
        default_factory=lambda: int(os.getenv("GABOR_WINDOW_LENGTH", "32"))
    )
    window_width: int = field(
        default_factory=lambda: int(os.getenv("GABOR_WINDOW_WIDTH", "16"))
    )
    freq_min: float = field(
        default_factory=lambda: float(os.getenv("GABOR_FREQ_MIN", "0.04"))
    )
    freq_max: float = field(
        default_factory=lambda: float(os.getenv("GABOR_FREQ_MAX", "0.33"))
    )
    recoverable_ratio: float = field(
        default_factory=lambda: float(os.getenv("GABOR_RECOVERABLE_RATIO", "0.40"))
    )


@dataclass(frozen=True)
class DoricConfig:
    """DORIC (Differential Orientation Reliability Index for Cores) parameters.

    Used by SingularityDetector to validate candidate core/delta points.
    """
    radius: int = field(
        default_factory=lambda: int(os.getenv("DORIC_RADIUS", "4"))
    )
    n_samples: int = field(
        default_factory=lambda: int(os.getenv("DORIC_N_SAMPLES", "16"))
    )
    rms_threshold: float = field(
        default_factory=lambda: float(os.getenv("DORIC_RMS_THRESHOLD", "0.30"))
    )
    poi_divisor: float = field(
        default_factory=lambda: float(os.getenv("DORIC_POI_DIVISOR", "6.283185307179586"))
    )


@dataclass(frozen=True)
class QdrantIndexConfig:
    """Qdrant HNSW index and payload configuration."""
    hnsw_m: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_HNSW_M", "16"))
    )
    hnsw_ef_construct: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "200"))
    )
    hnsw_full_scan_threshold: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_HNSW_FULL_SCAN_THRESHOLD", "10000"))
    )
    hnsw_payload_m: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_HNSW_PAYLOAD_M", "16"))
    )
    vector_dimension: int = field(
        default_factory=lambda: int(os.getenv("QDRANT_VECTOR_DIMENSION", "256"))
    )


@dataclass(frozen=True)
class SpuriousFilterConfig:
    """Skeleton spurious-structure removal thresholds.

    These are applied AFTER the DPI-scaled base thresholds from
    :func:`get_scaled_thresholds`. They control additional filtering
    for fragments below a confidence floor.
    """
    bridge_length_max: float = field(
        default_factory=lambda: float(os.getenv("SPURIOUS_BRIDGE_LENGTH_MAX", "0.47"))
    )
    fragment_area_min: float = field(
        default_factory=lambda: float(os.getenv("SPURIOUS_FRAGMENT_AREA_MIN", "0.25"))
    )
    min_recoverable_ratio: float = field(
        default_factory=lambda: float(os.getenv("SPURIOUS_MIN_RECOVERABLE_RATIO", "0.7"))
    )
    dpi_scale_floor: float = field(
        default_factory=lambda: float(os.getenv("SPURIOUS_DPI_SCALE_FLOOR", "0.25"))
    )


@dataclass(frozen=True)
class OrientationFieldConfig:
    """Orientation field analysis (Hong-Wan-Jain) parameters."""
    block_size: int = field(
        default_factory=lambda: int(os.getenv("ORIENTATION_BLOCK_SIZE", "16"))
    )
    min_energy: float = field(
        default_factory=lambda: float(os.getenv("ORIENTATION_MIN_ENERGY", "0.001"))
    )
    coherence_threshold: float = field(
        default_factory=lambda: float(os.getenv("ORIENTATION_COHERENCE_THRESHOLD", "0.35"))
    )


@dataclass(frozen=True)
class FusionConfig:
    """Cross-extractor fusion (EnsembleFusionFilter) parameters.

    Members within ``radius`` of each other are clustered and their
    confidences are summed. A bonus proportional to cluster size is added
    to the average confidence of the fused minutia.
    """
    radius: float = field(
        default_factory=lambda: float(os.getenv("FUSION_RADIUS", "8.0"))
    )
    min_votes: int = field(
        default_factory=lambda: int(os.getenv("FUSION_MIN_VOTES", "2"))
    )
    bonus_max: float = field(
        default_factory=lambda: float(os.getenv("FUSION_BONUS_MAX", "0.2"))
    )
    bonus_per_member: float = field(
        default_factory=lambda: float(os.getenv("FUSION_BONUS_PER_MEMBER", "0.1"))
    )


@dataclass(frozen=True)
class ExtractionConfig:
    """Binarisation + minutia extraction parameters."""
    binarized_min_white_ratio: float = field(
        default_factory=lambda: float(os.getenv("EXTRACTOR_BINARIZED_MIN_WHITE", "0.05"))
    )
    graph_weight_floor: float = field(
        default_factory=lambda: float(os.getenv("GRAPH_WEIGHT_FLOOR", "0.01"))
    )


@dataclass(frozen=True)
class EnhancerDefaultsConfig:
    """Default EnhancerConfig values, env-overridable.

    The :class:`~src.processing.enhancers.base.EnhancerConfig` dataclass
    holds these values; the factory function in
    :func:`src.processing.enhancer.create_enhancer` reads from this section
    so operators can tune Gabor/smoothing parameters without redeploy.
    """
    ridge_segment_blksze: int = field(
        default_factory=lambda: int(os.getenv("ENHANCER_RIDGE_SEGMENT_BLKSZE", "16"))
    )
    ridge_segment_thresh: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_RIDGE_SEGMENT_THRESH", "0.1"))
    )
    gradient_sigma: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_GRADIENT_SIGMA", "1.0"))
    )
    block_sigma: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_BLOCK_SIGMA", "7.0"))
    )
    orient_smooth_sigma: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_ORIENT_SMOOTH_SIGMA", "7.0"))
    )
    ridge_freq_blksze: int = field(
        default_factory=lambda: int(os.getenv("ENHANCER_RIDGE_FREQ_BLKSZE", "38"))
    )
    ridge_freq_windsze: int = field(
        default_factory=lambda: int(os.getenv("ENHANCER_RIDGE_FREQ_WINDSZE", "5"))
    )
    min_wave_length: int = field(
        default_factory=lambda: int(os.getenv("ENHANCER_MIN_WAVE_LENGTH", "5"))
    )
    max_wave_length: int = field(
        default_factory=lambda: int(os.getenv("ENHANCER_MAX_WAVE_LENGTH", "15"))
    )
    relative_scale_factor_x: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_RELATIVE_SCALE_X", "0.65"))
    )
    relative_scale_factor_y: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_RELATIVE_SCALE_Y", "0.65"))
    )
    angle_inc: int = field(
        default_factory=lambda: int(os.getenv("ENHANCER_ANGLE_INC", "3"))
    )
    mean_freq_default: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_MEAN_FREQ_DEFAULT", "0.1"))
    )
    white_ratio_min: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_WHITE_RATIO_MIN", "0.1"))
    )
    white_ratio_max: float = field(
        default_factory=lambda: float(os.getenv("ENHANCER_WHITE_RATIO_MAX", "0.9"))
    )


@dataclass(frozen=True)
class MccMatchingConfig:
    """Parameters for NIST Bozorth3 pair matching (Phase 27).

    Cylinders (Phase 21) and triplets (Phase 25) have been removed —
    pairs is the only matcher. See ``docs/adr/009-remove-cylinders.md``
    for the decision.
    """
    # Bozorth3 linker tolerances (Phase 27, Plan 27-01). Calibrated on
    # SOCOFing Altered-Easy CR for 5 subjects (100% top-1 accuracy).
    # 0.02 in normalised coords ≈ 5px at 256×256 (NBIS reference: 12px).
    link_dx_tol: float = field(
        default_factory=lambda: float(os.getenv("MCC_LINK_DX_TOL", "0.02"))
    )
    link_dy_tol: float = field(
        default_factory=lambda: float(os.getenv("MCC_LINK_DY_TOL", "0.02"))
    )
    link_dtheta_tol: float = field(
        default_factory=lambda: float(os.getenv("MCC_LINK_DTHETA_TOL", "0.15"))
    )
    confidence_saturation: int = field(
        default_factory=lambda: int(os.getenv("MCC_CONFIDENCE_SATURATION", "30"))
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("MCC_CONFIDENCE_THRESHOLD", "0.70"))
    )
    # Compute backend: "auto" (default) tries cupy then numpy;
    # "cupy" forces GPU; "numpy" forces CPU cv2. Falls back gracefully.
    compute_backend: str = field(
        default_factory=lambda: os.getenv("MCC_COMPUTE_BACKEND", "auto")
    )


@dataclass(frozen=True)
class Config:
    """Global application configuration (Immutable)."""

    # Environment
    env: str = field(default_factory=lambda: os.getenv("ENV", "development"))

    # Jurisdiction (legal framework config)
    jurisdiction: JurisdictionConfig = field(default_factory=JurisdictionConfig)

    # Database
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://postgres:postgres@localhost:5434/fingerprint"
        )
    )
    db_pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "5")))
    db_max_overflow: int = field(default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "10")))

    @property
    def async_database_url(self) -> str:
        """Async variant of ``database_url``."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("postgresql+psycopg2://"):
            return url.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)
        return url

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

    # Algorithm tunables (Phase 15 follow-up)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    gabor: GaborConfig = field(default_factory=GaborConfig)
    doric: DoricConfig = field(default_factory=DoricConfig)
    qdrant_index: QdrantIndexConfig = field(default_factory=QdrantIndexConfig)
    spurious_filter: SpuriousFilterConfig = field(default_factory=SpuriousFilterConfig)
    orientation_field: OrientationFieldConfig = field(default_factory=OrientationFieldConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    enhancer_defaults: EnhancerDefaultsConfig = field(default_factory=EnhancerDefaultsConfig)
    matching: MccMatchingConfig = field(default_factory=MccMatchingConfig)

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

    # NebulaGraph (Graph Database for Fine Matching)
    nebula_host: str = field(
        default_factory=lambda: os.getenv("NEBULA_HOST", "localhost")
    )
    nebula_port: int = field(
        default_factory=lambda: int(os.getenv("NEBULA_PORT", "9669"))
    )
    nebula_user: str = field(
        default_factory=lambda: os.getenv("NEBULA_USER", "root")
    )
    nebula_password: str = field(
        default_factory=lambda: os.getenv("NEBULA_PASSWORD", "nebula")
    )
    nebula_space: str = field(
        default_factory=lambda: os.getenv("NEBULA_SPACE", "biometric")
    )

    # Generative AI / LLM - ADR-006 Compatible
    llm_api_base: str = field(
        default_factory=lambda: os.getenv("LLM_API_BASE", "http://localhost:11434/v1")
    )
    llm_model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "qwen3:8b")
    )
    llm_api_key: SecretStr | None = field(
        default_factory=lambda: (
            SecretStr(value) if (value := os.getenv("LLM_API_KEY")) else None
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

    def __post_init__(self) -> None:
        """Post-initialization validations."""
        if self.combined_score_weight_l2 + self.combined_score_weight_cos != 1.0:
            # Normalize if they don't sum to 1 (optional, or raise warning)
            pass

# Global configuration instance
config = Config()

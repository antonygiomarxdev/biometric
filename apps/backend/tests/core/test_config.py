"""Tests for core configuration including GenAI settings."""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import SecretStr

from src.core.config import (
    Config,
    DoricConfig,
    EnhancerDefaultsConfig,
    ExtractionConfig,
    FusionConfig,
    GaborConfig,
    OrientationFieldConfig,
    PipelineConfig,
    QdrantIndexConfig,
    SpuriousFilterConfig,
)


class TestConfigLlmDefaults:
    """Default values should apply when no environment variables are set."""

    def test_llm_api_base_defaults_to_local(self) -> None:
        """LLM API base should default to local Ollama."""
        config = Config()
        assert config.llm_api_base == "http://localhost:11434/v1"

    def test_local_model_name_default(self) -> None:
        """Model name should default to 'qwen3:8b'."""
        config = Config()
        assert config.llm_model_name == "qwen3:8b"

    def test_openai_api_key_is_empty_secret(self) -> None:
        """OpenAI API key should default to an empty SecretStr."""
        config = Config()
        assert isinstance(config.openai_api_key, SecretStr)
        assert config.openai_api_key.get_secret_value() == ""


class TestConfigLlmFromEnv:
    """Environment variables should override config defaults."""

    @pytest.fixture(autouse=True)
    def _set_env(self) -> Any:
        """Set LLM environment variables for this test class."""
        env_vars = {
            "LLM_API_BASE": "https://api.openai.com/v1",
            "LLM_MODEL_NAME": "gpt-4o",
            "OPENAI_API_KEY": "sk-test-key-12345",
        }
        for key, value in env_vars.items():
            os.environ[key] = value
        yield
        for key in env_vars:
            os.environ.pop(key, None)

    def test_llm_api_base_from_env(self) -> None:
        """LLM API base should be read from environment."""
        config = Config()
        assert config.llm_api_base == "https://api.openai.com/v1"

    def test_local_model_name_from_env(self) -> None:
        """Model name should be read from environment."""
        config = Config()
        assert config.llm_model_name == "gpt-4o"

    def test_openai_api_key_from_env(self) -> None:
        """OpenAI API key should be read from environment as SecretStr."""
        config = Config()
        assert isinstance(config.openai_api_key, SecretStr)
        assert config.openai_api_key.get_secret_value() == "sk-test-key-12345"


# ---------------------------------------------------------------------------
# Phase 16: algorithm tunables
# ---------------------------------------------------------------------------


class TestPipelineConfigDefaults:
    """PipelineConfig defaults match the values previously hardcoded."""

    def test_skeleton_min_island_size_default(self) -> None:
        cfg = PipelineConfig()
        assert cfg.skeleton_min_island_size == 20

    def test_singularity_roi_radius_default(self) -> None:
        cfg = PipelineConfig()
        assert cfg.singularity_roi_radius == 140

    def test_fusion_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.fusion_radius == 8.0
        assert cfg.fusion_min_votes == 2

    def test_post_hook_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.spur_max_distance == 10.0
        assert cfg.healer_max_distance == 8.0
        assert cfg.border_px == 0
        assert cfg.border_roi_mode == "core"

    def test_orientation_and_confidence_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.orientation_window == 16
        assert cfg.orientation_coherence_threshold == 0.65
        assert cfg.low_confidence_threshold == 0.75


class TestPipelineConfigFromEnv:
    """PipelineConfig reads from env vars at construction time."""

    @pytest.fixture(autouse=True)
    def _set_env(self) -> Any:
        env_vars = {
            "PIPELINE_SKELETON_MIN_ISLAND_SIZE": "42",
            "PIPELINE_SINGULARITY_ROI_RADIUS": "200",
            "PIPELINE_FUSION_RADIUS": "12.5",
            "PIPELINE_FUSION_MIN_VOTES": "3",
            "PIPELINE_LOW_CONFIDENCE": "0.8",
        }
        for k, v in env_vars.items():
            os.environ[k] = v
        yield
        for k in env_vars:
            os.environ.pop(k, None)

    def test_overrides(self) -> None:
        cfg = PipelineConfig()
        assert cfg.skeleton_min_island_size == 42
        assert cfg.singularity_roi_radius == 200
        assert cfg.fusion_radius == 12.5
        assert cfg.fusion_min_votes == 3
        assert cfg.low_confidence_threshold == 0.8


class TestGaborConfig:
    """Gabor enhancement tunables."""

    def test_defaults(self) -> None:
        cfg = GaborConfig()
        assert cfg.block_size == 16
        assert cfg.window_length == 32
        assert cfg.window_width == 16
        assert cfg.freq_min == 0.04
        assert cfg.freq_max == 0.33
        assert cfg.recoverable_ratio == 0.40


class TestDoricConfig:
    """DORIC + Poincaré tunables."""

    def test_defaults(self) -> None:
        cfg = DoricConfig()
        assert cfg.radius == 4
        assert cfg.n_samples == 16
        assert cfg.rms_threshold == 0.30
        # 2π = 6.283185307179586
        assert cfg.poi_divisor == pytest.approx(2 * 3.141592653589793)

    def test_env_override(self) -> None:
        os.environ["DORIC_RMS_THRESHOLD"] = "0.45"
        try:
            cfg = DoricConfig()
            assert cfg.rms_threshold == 0.45
        finally:
            os.environ.pop("DORIC_RMS_THRESHOLD", None)


class TestQdrantIndexConfig:
    """Qdrant HNSW + payload index tunables."""

    def test_defaults(self) -> None:
        cfg = QdrantIndexConfig()
        assert cfg.hnsw_m == 16
        assert cfg.hnsw_ef_construct == 200
        assert cfg.vector_dimension == 256


class TestSpuriousFilterConfig:
    """Spurious-filter tunables."""

    def test_defaults(self) -> None:
        cfg = SpuriousFilterConfig()
        assert cfg.bridge_length_max == 0.47
        assert cfg.fragment_area_min == 0.25
        assert cfg.min_recoverable_ratio == 0.7
        assert cfg.dpi_scale_floor == 0.25


class TestConfigAlgorithmicSectionsPresent:
    """Top-level Config should expose all algorithm sections."""

    def test_all_sections_present(self) -> None:
        cfg = Config()
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.gabor, GaborConfig)
        assert isinstance(cfg.doric, DoricConfig)
        assert isinstance(cfg.qdrant_index, QdrantIndexConfig)
        assert isinstance(cfg.spurious_filter, SpuriousFilterConfig)
        assert isinstance(cfg.orientation_field, OrientationFieldConfig)
        assert isinstance(cfg.fusion, FusionConfig)
        assert isinstance(cfg.extraction, ExtractionConfig)
        assert isinstance(cfg.enhancer_defaults, EnhancerDefaultsConfig)


class TestConstantsMatchConfig:

    def test_gabor_constants_loaded(self) -> None:
        from src.processing.gabor import (
            BLOCK_SIZE,
            FREQ_MAX,
            FREQ_MIN,
            WINDOW_LENGTH,
            WINDOW_WIDTH,
            _RECOVERABLE_RATIO,
        )
        from src.core.config import config
        assert BLOCK_SIZE == config.gabor.block_size
        assert WINDOW_LENGTH == config.gabor.window_length
        assert WINDOW_WIDTH == config.gabor.window_width
        assert FREQ_MIN == config.gabor.freq_min
        assert FREQ_MAX == config.gabor.freq_max
        assert _RECOVERABLE_RATIO == config.gabor.recoverable_ratio

    def test_spurious_filter_constants_loaded(self) -> None:
        from src.processing.spurious_filter import (
            _DPI_SCALE_FLOOR,
            _NON_FACING_SPUR_RELAXATION,
        )
        from src.core.config import config
        assert _NON_FACING_SPUR_RELAXATION == config.spurious_filter.min_recoverable_ratio
        assert _DPI_SCALE_FLOOR == 0.25


# ---------------------------------------------------------------------------
# Phase 16: extended algorithm tunables (orientation, fusion, extraction, enhancer)
# ---------------------------------------------------------------------------


class TestOrientationFieldConfig:
    """OrientationFieldConfig defaults match the values previously hardcoded."""

    def test_defaults(self) -> None:
        cfg = OrientationFieldConfig()
        assert cfg.block_size == 16
        assert cfg.min_energy == pytest.approx(0.001)
        assert cfg.coherence_threshold == 0.35

    def test_orientation_field_analyzer_uses_config(self) -> None:
        """OrientationFieldAnalyzer() with no args should pick config values."""
        from src.processing.pre_hooks import OrientationFieldAnalyzer
        from src.core.config import config
        ofa = OrientationFieldAnalyzer()
        assert ofa.coherence_threshold == config.orientation_field.coherence_threshold
        assert ofa.block_size == config.orientation_field.block_size
        assert ofa.min_energy == config.orientation_field.min_energy

    def test_orientation_field_analyzer_override(self) -> None:
        """Passing explicit values overrides config defaults."""
        from src.processing.pre_hooks import OrientationFieldAnalyzer
        ofa = OrientationFieldAnalyzer(coherence_threshold=0.99)
        assert ofa.coherence_threshold == 0.99


class TestFusionConfig:
    """FusionConfig (EnsembleFusionFilter) defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = FusionConfig()
        assert cfg.radius == 8.0
        assert cfg.min_votes == 2
        assert cfg.bonus_max == 0.2
        assert cfg.bonus_per_member == 0.1

    def test_ensemble_fusion_uses_config(self) -> None:
        from src.processing.post_hooks import EnsembleFusionFilter
        from src.core.config import config
        eff = EnsembleFusionFilter()
        assert eff.radius == config.fusion.radius
        assert eff.min_votes == config.fusion.min_votes
        assert eff._bonus_max == config.fusion.bonus_max
        assert eff._bonus_per_member == config.fusion.bonus_per_member


class TestExtractionConfig:
    """ExtractionConfig (binarisation + graph weight) defaults."""

    def test_defaults(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.binarized_min_white_ratio == 0.05
        assert cfg.graph_weight_floor == 0.01


class TestEnhancerDefaultsConfig:
    """EnhancerDefaultsConfig mirrors EnhancerConfig defaults."""

    def test_defaults(self) -> None:
        cfg = EnhancerDefaultsConfig()
        assert cfg.ridge_segment_blksze == 16
        assert cfg.ridge_segment_thresh == 0.1
        assert cfg.gradient_sigma == 1.0
        assert cfg.block_sigma == 7.0
        assert cfg.orient_smooth_sigma == 7.0
        assert cfg.ridge_freq_blksze == 38
        assert cfg.ridge_freq_windsze == 5
        assert cfg.min_wave_length == 5
        assert cfg.max_wave_length == 15
        assert cfg.relative_scale_factor_x == 0.65
        assert cfg.relative_scale_factor_y == 0.65
        assert cfg.angle_inc == 3
        assert cfg.mean_freq_default == 0.1
        assert cfg.white_ratio_min == 0.1
        assert cfg.white_ratio_max == 0.9

    def test_enhancer_config_from_env(self) -> None:
        """EnhancerConfig.from_env() should mirror EnhancerDefaultsConfig."""
        from src.processing.enhancers.base import EnhancerConfig
        ec = EnhancerConfig.from_env()
        ed = EnhancerDefaultsConfig()
        assert ec.ridge_segment_blksze == ed.ridge_segment_blksze
        assert ec.gradient_sigma == ed.gradient_sigma
        assert ec.relative_scale_factor_x == ed.relative_scale_factor_x
        assert ec.angle_inc == ed.angle_inc

    def test_create_enhancer_uses_from_env(self) -> None:
        """The :func:`create_enhancer` factory should call ``EnhancerConfig.from_env()`` by default.

        We verify this by reading the factory source file directly — the
        conftest mocks the function globally for the rest of the suite,
        so :func:`inspect.getsource` on the imported binding would fail.
        """
        from pathlib import Path
        from src.processing.enhancers.base import EnhancerConfig
        src_path = Path(__file__).parents[2] / "src" / "processing" / "enhancer.py"
        text = src_path.read_text()
        assert "from_env" in text, "create_enhancer should call EnhancerConfig.from_env()"
        # Sanity: EnhancerConfig.from_env() mirrors the config
        assert (
            EnhancerConfig.from_env().gradient_sigma
            == EnhancerConfig().gradient_sigma
        )

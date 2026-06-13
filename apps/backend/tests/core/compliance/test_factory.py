"""Tests for the compliance strategy factory."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.compliance.base import BaseStrategy
from src.core.compliance.extreme import ExtremePrivacyStrategy
from src.core.compliance.factory import (
    get_compliance_strategy,
    get_compliance_strategy_from_config,
)
from src.core.compliance.strategy import IComplianceStrategy


class TestGetComplianceStrategy:
    """Tests for get_compliance_strategy(name)."""

    def test_returns_base_for_base(self) -> None:
        """'base' should return a BaseStrategy instance."""
        strategy = get_compliance_strategy("base")
        assert isinstance(strategy, BaseStrategy)
        assert isinstance(strategy, IComplianceStrategy)

    def test_returns_extreme_for_extreme(self) -> None:
        """'extreme' should return an ExtremePrivacyStrategy instance."""
        strategy = get_compliance_strategy("extreme")
        assert isinstance(strategy, ExtremePrivacyStrategy)
        assert isinstance(strategy, IComplianceStrategy)

    def test_raises_value_error_for_unknown(self) -> None:
        """An unknown strategy name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown compliance strategy"):
            get_compliance_strategy("gdpr")

    def test_raises_value_error_for_empty_string(self) -> None:
        """An empty string should raise ValueError."""
        with pytest.raises(ValueError):
            get_compliance_strategy("")

    def test_raises_value_error_for_typo(self) -> None:
        """A misspelled name should raise ValueError."""
        with pytest.raises(ValueError):
            get_compliance_strategy("Base")  # capital B

    def test_returns_new_instance_each_call(self) -> None:
        """Each call should return a fresh instance."""
        s1 = get_compliance_strategy("extreme")
        s2 = get_compliance_strategy("extreme")
        assert s1 is not s2


class TestGetComplianceStrategyFromConfig:
    """Tests for get_compliance_strategy_from_config(config)."""

    def test_uses_config_attribute(self) -> None:
        """Should read compliance_strategy from the config object."""
        class FakeConfig:
            compliance_strategy: str = "extreme"

        strategy = get_compliance_strategy_from_config(FakeConfig())
        assert isinstance(strategy, ExtremePrivacyStrategy)

    def test_defaults_to_base(self) -> None:
        """Should default to 'base' when config has no compliance_strategy."""
        class BareConfig:
            pass

        strategy = get_compliance_strategy_from_config(BareConfig())
        assert isinstance(strategy, BaseStrategy)

    def test_raises_on_invalid_config_value(self) -> None:
        """Should raise ValueError when config has an invalid strategy name."""
        class FakeConfig:
            compliance_strategy: str = "invalid_strategy"

        with pytest.raises(ValueError, match="Unknown compliance strategy"):
            get_compliance_strategy_from_config(FakeConfig())

    def test_works_with_actual_config(self) -> None:
        """Should work with the real Config class."""
        from src.core.config import Config

        config = Config()
        strategy = get_compliance_strategy_from_config(config)
        assert isinstance(strategy, BaseStrategy)  # default

        # Override and verify
        config_extreme = Config(compliance_strategy="extreme")
        strategy_extreme = get_compliance_strategy_from_config(config_extreme)
        assert isinstance(strategy_extreme, ExtremePrivacyStrategy)

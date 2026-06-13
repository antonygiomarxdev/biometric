"""Compliance strategy package for dynamic jurisdiction-aware privacy enforcement."""

from src.core.compliance.strategy import IComplianceStrategy
from src.core.compliance.base import BaseStrategy
from src.core.compliance.extreme import ExtremePrivacyStrategy
from src.core.compliance.factory import (
    get_compliance_strategy,
    get_compliance_strategy_from_config,
)

__all__ = [
    "IComplianceStrategy",
    "BaseStrategy",
    "ExtremePrivacyStrategy",
    "get_compliance_strategy",
    "get_compliance_strategy_from_config",
]

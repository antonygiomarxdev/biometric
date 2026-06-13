"""Compliance strategy factory — instantiates strategies based on configuration."""

from __future__ import annotations

from src.core.compliance.base import BaseStrategy
from src.core.compliance.extreme import ExtremePrivacyStrategy
from src.core.compliance.strategy import IComplianceStrategy


# Registry of available compliance strategy names to their constructors.
# New strategies register here following the Open/Closed principle —
# add a new entry without modifying existing strategy code.
_STRATEGY_REGISTRY: dict[str, type[IComplianceStrategy]] = {
    "base": BaseStrategy,
    "extreme": ExtremePrivacyStrategy,
}


def get_compliance_strategy(name: str) -> IComplianceStrategy:
    """Return the compliance strategy instance for the given name.

    Args:
        name: The strategy name (e.g., ``"base"``, ``"extreme"``).
              Case-sensitive — must match the registry exactly.

    Returns:
        An instance of the requested strategy.

    Raises:
        ValueError: If the strategy name is not registered.

    Example:
        >>> strategy = get_compliance_strategy("extreme")
        >>> strategy.requires_client_side_encryption()
        True
    """
    strategy_cls = _STRATEGY_REGISTRY.get(name)
    if strategy_cls is None:
        valid = ", ".join(sorted(_STRATEGY_REGISTRY))
        raise ValueError(
            f"Unknown compliance strategy: '{name}'. "
            f"Available strategies: {valid}"
        )
    return strategy_cls()


def get_compliance_strategy_from_config(config: object) -> IComplianceStrategy:
    """Return the compliance strategy from a configuration object.

    The configuration object must have a ``compliance_strategy`` attribute
    (a string, like those returned by ``Config.compliance_strategy``).

    This is the primary entry point for dependency injection — it decouples
    the factory from any specific configuration class.

    Args:
        config: Any object with a ``compliance_strategy`` attribute.

    Returns:
        An instance of the requested strategy.
    """
    name: str = getattr(config, "compliance_strategy", "base")
    return get_compliance_strategy(name)

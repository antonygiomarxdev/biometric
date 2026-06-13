"""
Compliance-aware logging — PII scrubbing via the active compliance strategy.

This module provides two mechanisms for scrubbing PII from log output:

1. **``ComplianceLogFormatter``** — a custom ``logging.Formatter`` that scrubs
   the formatted message before sending it to output. This is the recommended
   approach for new handler configurations.

2. **``PIIFilter``** — a custom ``logging.Filter`` that scrubs the
   ``LogRecord.msg`` attribute at filter time. This is useful for retrofitting
   existing logger configurations without replacing their formatters.

3. **``setup_compliance_logging()``** — a convenience function that applies PII
   scrubbing to the root logger using one or both of the above mechanisms.

All three components obtain the active compliance strategy through
``get_compliance_strategy_from_config()``, which reads the ``COMPLIANCE_STRATEGY``
environment variable. If the compliance system is unavailable, they fall back
to ``BaseStrategy`` (no scrubbing), ensuring logging is never disrupted.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.compliance.strategy import IComplianceStrategy

# ---------------------------------------------------------------------------
# Strategy resolution
# ---------------------------------------------------------------------------

_RESOLVED_STRATEGY: IComplianceStrategy | None = None
"""Cached strategy to avoid re-resolving the config on every instantiation."""


def _resolve_strategy() -> IComplianceStrategy:
    """Resolve the active compliance strategy, falling back to ``BaseStrategy``.

    Uses a module-level cache so the strategy is only resolved once per
    process lifetime. If resolution fails (e.g. config not yet loaded) it
    returns a ``BaseStrategy`` instance — logging should never break.
    """
    global _RESOLVED_STRATEGY

    if _RESOLVED_STRATEGY is not None:
        return _RESOLVED_STRATEGY

    try:
        from src.core.compliance.factory import get_compliance_strategy_from_config

        # Lazy-import config here to avoid circular imports at module load time
        from src.core.config import config  # type: ignore[import-untyped]

        _RESOLVED_STRATEGY = get_compliance_strategy_from_config(config)
    except Exception:
        from src.core.compliance.base import BaseStrategy

        _RESOLVED_STRATEGY = BaseStrategy()

    return _RESOLVED_STRATEGY


def _reset_strategy_cache() -> None:
    """Reset the cached strategy (primarily for testing)."""
    global _RESOLVED_STRATEGY
    _RESOLVED_STRATEGY = None


# ---------------------------------------------------------------------------
# ComplianceLogFormatter
# ---------------------------------------------------------------------------


class ComplianceLogFormatter(logging.Formatter):
    """Log formatter that scrubs PII from messages before output.

    Uses the active ``IComplianceStrategy.scrub_pii()`` to redact sensitive
    data from every log record before writing it. The original ``LogRecord``
    is not mutated — only the formatted string is scrubbed.

    If no *strategy* is provided, one is auto-resolved from the application
    configuration (``COMPLIANCE_STRATEGY`` env var). If resolution fails, the
    formatter falls back to ``BaseStrategy`` (no scrubbing), ensuring logging
    is never disrupted.

    Args:
        strategy: An optional ``IComplianceStrategy`` instance. Pass ``None``
            (default) to auto-resolve from config.
        fmt: Optional ``logging.Formatter``-compatible format string.
        datefmt: Optional date/time format string.
        style: Format style (``'%'``, ``'{'``, or ``'$'``). Defaults to ``'%'``.
    """

    def __init__(
        self,
        strategy: IComplianceStrategy | None = None,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: str = "%",
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._strategy: IComplianceStrategy = strategy or _resolve_strategy()

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, then scrub PII from the final output.

        Calls the parent formatter first, then passes the result through
        ``self._strategy.scrub_pii()``. The original *record* may be mutated
        by the parent formatter (standard CPython behaviour — it sets
        ``record.message`` and ``record.asctime``), but ``record.msg`` is
        preserved as-is.
        """
        formatted: str = super().format(record)
        return self._strategy.scrub_pii(formatted)


# ---------------------------------------------------------------------------
# PIIFilter
# ---------------------------------------------------------------------------


class PIIFilter(logging.Filter):
    """Log filter that scrubs PII from ``LogRecord.msg`` at filter time.

    This is an alternative to ``ComplianceLogFormatter`` for cases where the
    handler's formatter cannot be replaced (e.g. third-party handler libraries).
    It mutates ``record.msg`` *before* any formatter processes it, so the
    scrubbed text flows through to all attached handlers.

    The filter always returns ``True`` — it never suppresses log records; it
    only transforms them.

    Args:
        strategy: An optional ``IComplianceStrategy`` instance. Pass ``None``
            (default) to auto-resolve from config.
        name: Optional filter name (passed to ``logging.Filter.__init__``).
    """

    def __init__(
        self,
        strategy: IComplianceStrategy | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name)
        self._strategy: IComplianceStrategy = strategy or _resolve_strategy()

    def filter(self, record: logging.LogRecord) -> bool:
        """Scrub PII from the record's message and always allow it.

        Args:
            record: The log record to scrub.

        Returns:
            ``True`` always — this filter never suppresses records.
        """
        record.msg = self._strategy.scrub_pii(record.msg)
        return True


# ---------------------------------------------------------------------------
# setup_compliance_logging
# ---------------------------------------------------------------------------


def setup_compliance_logging(
    strategy: IComplianceStrategy | None = None,
    *,
    add_filter: bool = True,
    update_handlers: bool = True,
) -> None:
    """Configure the root logger with PII scrubbing.

    Applies PII scrubbing to the root logger using one or both mechanisms:

    - A ``PIIFilter`` on the root logger itself (intercepts all log records
      regardless of handler).
    - ``ComplianceLogFormatter`` on every existing root-logger handler
      (preserving the original format string).

    If the same mechanism has already been applied (e.g. a ``PIIFilter`` is
    already on the root logger), it is not duplicated.

    Args:
        strategy: Optional compliance strategy. Auto-resolved if omitted.
        add_filter: Whether to add a ``PIIFilter`` to the root logger.
            Defaults to ``True``.
        update_handlers: Whether to replace formatters on existing root
            handlers. Defaults to ``True``.
    """
    resolved: IComplianceStrategy = strategy or _resolve_strategy()
    root: logging.Logger = logging.getLogger()

    if add_filter:
        _install_pii_filter(root, resolved)

    if update_handlers:
        _update_handler_formatters(root, resolved)


def _install_pii_filter(root: logging.Logger, strategy: IComplianceStrategy) -> None:
    """Add a ``PIIFilter`` to *root* if one is not already present."""
    for existing in root.filters:
        if isinstance(existing, PIIFilter):
            return  # already installed
    root.addFilter(PIIFilter(strategy=strategy))


def _update_handler_formatters(
    root: logging.Logger,
    strategy: IComplianceStrategy,
) -> None:
    """Replace each root handler's formatter with ``ComplianceLogFormatter``."""
    for handler in root.handlers:
        current: logging.Formatter | None = handler.formatter
        if isinstance(current, ComplianceLogFormatter):
            continue  # already using compliance formatter

        fmt: str | None = current._fmt if current is not None else None  # type: ignore[attr-defined]
        datefmt: str | None = current.datefmt if current is not None else None

        handler.setFormatter(
            ComplianceLogFormatter(
                strategy=strategy,
                fmt=fmt,
                datefmt=datefmt,
            ),
        )

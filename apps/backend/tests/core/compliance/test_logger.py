"""Tests for the compliance-aware logging formatter and PII filter.

Verifies that:
- ``ComplianceLogFormatter`` scrubs messages based on the active strategy.
- ``PIIFilter`` scrubs ``LogRecord.msg`` at filter time.
- ``setup_compliance_logging`` configures the root logger without disruption.
"""

from __future__ import annotations

import io
import logging

import pytest

from unittest.mock import patch

from src.core.compliance.base import BaseStrategy
from src.core.compliance.extreme import ExtremePrivacyStrategy
from src.core.compliance.logger import (
    ComplianceLogFormatter,
    PIIFilter,
    _reset_strategy_cache,
    setup_compliance_logging,
)


# ---------------------------------------------------------------------------
# ComplianceLogFormatter tests
# ---------------------------------------------------------------------------


class TestComplianceLogFormatter:
    """ComplianceLogFormatter should scrub messages based on active strategy."""

    @pytest.fixture
    def base_formatter(self) -> ComplianceLogFormatter:
        return ComplianceLogFormatter(strategy=BaseStrategy())

    @pytest.fixture
    def extreme_formatter(self) -> ComplianceLogFormatter:
        return ComplianceLogFormatter(strategy=ExtremePrivacyStrategy())

    # -- BaseStrategy (no scrubbing) ---------------------------------------

    def test_base_strategy_leaves_pii_unchanged(
        self,
        base_formatter: ComplianceLogFormatter,
    ) -> None:
        """With BaseStrategy, PII should remain in log output."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="User email: test@example.com",
            args=(),
            exc_info=None,
        )
        result = base_formatter.format(record)
        assert "test@example.com" in result

    # -- ExtremePrivacyStrategy (aggressive scrubbing) ---------------------

    def test_extreme_strategy_scrubs_email(
        self,
        extreme_formatter: ComplianceLogFormatter,
    ) -> None:
        """With ExtremePrivacyStrategy, emails should be redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Contact: john.doe@example.com",
            args=(),
            exc_info=None,
        )
        result = extreme_formatter.format(record)
        assert "john.doe@example.com" not in result
        assert "[REDACTED]" in result

    def test_extreme_strategy_scrubs_uuid(
        self,
        extreme_formatter: ComplianceLogFormatter,
    ) -> None:
        """With ExtremePrivacyStrategy, UUIDs should be redacted."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="User UUID: 550e8400-e29b-41d4-a716-446655440000",
            args=(),
            exc_info=None,
        )
        result = extreme_formatter.format(record)
        assert "550e8400-e29b-41d4-a716-446655440000" not in result
        assert "[REDACTED]" in result

    def test_clean_message_passes_through(
        self,
        extreme_formatter: ComplianceLogFormatter,
    ) -> None:
        """Messages without PII should be unchanged."""
        msg = "Fingerprint match result: confidence 0.95, candidate ID 42"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg=msg,
            args=(),
            exc_info=None,
        )
        result = extreme_formatter.format(record)
        assert result == msg

    def test_does_not_mutate_original_record_msg(
        self,
        extreme_formatter: ComplianceLogFormatter,
    ) -> None:
        """The original record.msg should be preserved after format()."""
        original_msg = "Email: user@test.com"
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg=original_msg,
            args=(),
            exc_info=None,
        )
        extreme_formatter.format(record)
        assert record.msg == original_msg

    def test_preserves_log_metadata_in_output(
        self,
    ) -> None:
        """Format string metadata (levelname, etc.) should be preserved."""
        extreme_formatter = ComplianceLogFormatter(
            fmt="%(levelname)s - %(message)s",
            strategy=ExtremePrivacyStrategy(),
        )
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=42,
            msg="Email: secret@example.com",
            args=(),
            exc_info=None,
        )
        result = extreme_formatter.format(record)
        assert "WARNING" in result
        assert "[REDACTED]" in result
        assert "secret@example.com" not in result

    def test_graceful_fallback_on_none_strategy(self) -> None:
        """With no strategy provided, should fall back gracefully."""
        formatter = ComplianceLogFormatter()
        # Should not raise
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Some log message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert result is not None

    def test_end_to_end_with_handler(self) -> None:
        """Integration test: log through a handler with ComplianceLogFormatter."""
        logger = logging.getLogger("test_e2e_logger")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            ComplianceLogFormatter(strategy=ExtremePrivacyStrategy()),
        )
        logger.addHandler(handler)

        logger.info("User email: user@test.com, UUID: 550e8400-e29b-41d4-a716-446655440000")

        output = stream.getvalue()
        assert "[REDACTED]" in output
        assert "user@test.com" not in output
        assert "550e8400-e29b-41d4-a716-446655440000" not in output

        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# PIIFilter tests
# ---------------------------------------------------------------------------


class TestPIIFilter:
    """PIIFilter should scrub LogRecord.msg at filter time."""

    @pytest.fixture
    def extreme_filter(self) -> PIIFilter:
        return PIIFilter(strategy=ExtremePrivacyStrategy())

    @pytest.fixture
    def base_filter(self) -> PIIFilter:
        return PIIFilter(strategy=BaseStrategy())

    def test_base_filter_leaves_message_unchanged(
        self,
        base_filter: PIIFilter,
    ) -> None:
        """Base filter should not modify the message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Email: test@example.com",
            args=(),
            exc_info=None,
        )
        assert base_filter.filter(record) is True
        assert record.msg == "Email: test@example.com"

    def test_extreme_filter_scrubs_email(
        self,
        extreme_filter: PIIFilter,
    ) -> None:
        """Extreme filter should scrub emails from the message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Email: test@example.com",
            args=(),
            exc_info=None,
        )
        assert extreme_filter.filter(record) is True
        assert "test@example.com" not in record.msg
        assert "[REDACTED]" in record.msg

    def test_extreme_filter_scrubs_uuid(
        self,
        extreme_filter: PIIFilter,
    ) -> None:
        """Extreme filter should scrub UUIDs from the message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="UUID: 550e8400-e29b-41d4-a716-446655440000",
            args=(),
            exc_info=None,
        )
        assert extreme_filter.filter(record) is True
        assert "550e8400-e29b-41d4-a716-446655440000" not in record.msg

    def test_filter_always_returns_true(
        self,
        extreme_filter: PIIFilter,
    ) -> None:
        """Filter should never block a record."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="clean message",
            args=(),
            exc_info=None,
        )
        assert extreme_filter.filter(record) is True


# ---------------------------------------------------------------------------
# setup_compliance_logging tests
# ---------------------------------------------------------------------------


class TestSetupComplianceLogging:
    """setup_compliance_logging should configure the root logger."""

    def test_executes_without_error(self) -> None:
        """Should not raise when configuring with BaseStrategy."""
        setup_compliance_logging(strategy=BaseStrategy())

    def test_adds_pii_filter_to_root_logger(self) -> None:
        """Should add a PIIFilter to the root logger."""
        root = logging.getLogger()
        # Remove any pre-existing PIIFilter instances for clean assertion
        for f in list(root.filters):
            if isinstance(f, PIIFilter):
                root.removeFilter(f)

        setup_compliance_logging(strategy=BaseStrategy(), update_handlers=False)

        has_filter = any(isinstance(f, PIIFilter) for f in root.filters)
        assert has_filter, "Expected PIIFilter on root logger"

    def test_idempotent_when_called_twice(self) -> None:
        """Calling setup_compliance_logging twice should not duplicate filters."""
        root = logging.getLogger()
        for f in list(root.filters):
            if isinstance(f, PIIFilter):
                root.removeFilter(f)

        setup_compliance_logging(strategy=BaseStrategy(), update_handlers=False)
        filter_count_after_first = sum(1 for f in root.filters if isinstance(f, PIIFilter))

        setup_compliance_logging(strategy=BaseStrategy(), update_handlers=False)
        filter_count_after_second = sum(1 for f in root.filters if isinstance(f, PIIFilter))

        assert filter_count_after_second == filter_count_after_first == 1

    def test_does_not_add_filter_when_add_filter_false(self) -> None:
        """Setting add_filter=False should skip filter installation."""
        root = logging.getLogger()
        for f in list(root.filters):
            if isinstance(f, PIIFilter):
                root.removeFilter(f)

        setup_compliance_logging(strategy=BaseStrategy(), add_filter=False, update_handlers=False)

        has_filter = any(isinstance(f, PIIFilter) for f in root.filters)
        assert not has_filter, "No PIIFilter expected on root logger"

    def test_updates_handler_formatters(self) -> None:
        """When update_handlers=True, existing handler formatters should be replaced."""
        root = logging.getLogger()
        # Remove pre-existing handlers for clean test
        old_handlers = list(root.handlers)
        for h in old_handlers:
            root.removeHandler(h)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)

        setup_compliance_logging(strategy=ExtremePrivacyStrategy(), add_filter=False, update_handlers=True)

        assert isinstance(handler.formatter, ComplianceLogFormatter)

        # Cleanup
        root.removeHandler(handler)
        for h in old_handlers:
            root.addHandler(h)

    def test_skips_handler_with_existing_compliance_formatter(self) -> None:
        """Handlers already using ComplianceLogFormatter should not be replaced."""
        root = logging.getLogger()
        old_handlers = list(root.handlers)
        for h in old_handlers:
            root.removeHandler(h)

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        existing = ComplianceLogFormatter(fmt="%(message)s", strategy=BaseStrategy())
        handler.setFormatter(existing)
        root.addHandler(handler)

        setup_compliance_logging(strategy=ExtremePrivacyStrategy(), add_filter=False, update_handlers=True)

        # The formatter should STILL be the original instance (not replaced)
        assert handler.formatter is existing
        assert isinstance(handler.formatter, ComplianceLogFormatter)

        # Cleanup
        root.removeHandler(handler)
        for h in old_handlers:
            root.addHandler(h)

    def test_with_style_brace(self) -> None:
        """ComplianceLogFormatter should work with '{' style formatting."""
        formatter = ComplianceLogFormatter(
            fmt="{levelname} - {message}",
            style="{",
            strategy=ExtremePrivacyStrategy(),
        )
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Email: user@test.com",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "INFO" in result
        assert "[REDACTED]" in result
        assert "user@test.com" not in result

    def test_with_style_dollar(self) -> None:
        """ComplianceLogFormatter should work with '$' style formatting."""
        formatter = ComplianceLogFormatter(
            fmt="$levelname - $message",
            style="$",
            strategy=ExtremePrivacyStrategy(),
        )
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=42,
            msg="UUID: 550e8400-e29b-41d4-a716-446655440000",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "WARNING" in result
        assert "[REDACTED]" in result
        assert "550e8400-e29b-41d4-a716-446655440000" not in result


# ---------------------------------------------------------------------------
# Strategy resolution tests
# --------------------------------------------------------------------------


class TestStrategyResolution:
    """Tests for the internal _resolve_strategy function."""

    def test_fallback_on_factory_failure(self) -> None:
        """When factory resolution fails, should fall back to BaseStrategy."""
        _reset_strategy_cache()
        with patch(
            "src.core.compliance.factory.get_compliance_strategy_from_config",
            side_effect=ValueError("fail"),
        ):
            # ComplianceLogFormatter with no strategy triggers _resolve_strategy
            formatter = ComplianceLogFormatter()
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname=__file__,
                lineno=42,
                msg="clean message",
                args=(),
                exc_info=None,
            )
            result = formatter.format(record)
            # BaseStrategy = passthrough, no scrubbing
            assert result is not None
            assert result.endswith("clean message")

    def test_cache_hit_returns_same_strategy(self) -> None:
        """Calling _resolve_strategy twice should return cached strategy."""
        import src.core.compliance.logger as compliance_logger_mod

        _reset_strategy_cache()
        assert compliance_logger_mod._RESOLVED_STRATEGY is None

        # First call populates the cache
        ComplianceLogFormatter()
        cached = compliance_logger_mod._RESOLVED_STRATEGY
        assert cached is not None

        # Second call should not re-resolve (same object)
        ComplianceLogFormatter()
        assert compliance_logger_mod._RESOLVED_STRATEGY is cached

        _reset_strategy_cache()

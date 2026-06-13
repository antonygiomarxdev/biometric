"""Tests for the read-only database engine factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sqlalchemy import Engine

from src.db.readonly import get_readonly_engine


class TestGetReadOnlyEngine:
    """``get_readonly_engine`` should return a locked-down SQLAlchemy engine."""

    @patch("src.db.readonly.create_engine")
    def test_returns_sqlalchemy_engine(
        self, mock_create_engine: MagicMock
    ) -> None:
        """It returns a SQLAlchemy ``Engine`` instance."""
        mock_engine = MagicMock(spec=Engine)
        mock_create_engine.return_value = mock_engine

        result = get_readonly_engine()

        assert result is mock_engine

    @patch("src.db.readonly.create_engine")
    def test_engine_uses_read_only_execution_options(
        self, mock_create_engine: MagicMock
    ) -> None:
        """The engine is created with read-only transaction semantics."""
        get_readonly_engine()

        mock_create_engine.assert_called_once()
        _call_args, call_kwargs = mock_create_engine.call_args

        assert "execution_options" in call_kwargs
        options = call_kwargs["execution_options"]

        assert options.get("isolation_level") == "AUTOCOMMIT"
        assert options.get("postgresql_readonly") is True

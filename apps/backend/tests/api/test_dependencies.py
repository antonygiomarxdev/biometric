"""
Tests for :mod:`~src.api.dependencies` — DI providers and security.

Covers:
- :class:`AppResources` — lifecycle (init/dispose).
- :func:`get_db` — session factory and error path.
- :func:`get_current_user` — token validation and user lookup.
- :class:`RequireRole` — role-based access control.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# AppResources
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_config() -> Any:
    """Return a ``Config`` with SQLite-compatible pool settings."""
    from src.core.config import Config

    return Config(
        database_url="sqlite:///:memory:",
        db_pool_size=1,
        db_max_overflow=0,
        log_level="DEBUG",
    )


class TestAppResources:
    """Lifecycle tests for :class:`AppResources`."""

    def test_initial_state(self) -> None:
        """All resources start as ``None``."""
        from src.api.dependencies import AppResources

        r = AppResources()
        assert r.engine is None
        assert r.session_factory is None
        assert r.async_engine is None
        assert r.async_session_factory is None
        assert r.process_pool is None

    def test_init_db_creates_engine(self) -> None:
        """``init_db()`` creates SQLAlchemy engines and session factories."""
        from src.api.dependencies import AppResources

        mock_engine = MagicMock()
        mock_async_engine = MagicMock()
        with (
            patch("src.api.dependencies.create_engine", return_value=mock_engine),
            patch(
                "src.api.dependencies.create_async_engine",
                return_value=mock_async_engine,
            ),
        ):
            r = AppResources()
            r.init_db(database_url="sqlite:///:memory:")
            assert r.engine is mock_engine
            assert r.session_factory is not None
            assert r.async_engine is mock_async_engine
            assert r.async_session_factory is not None

    def test_init_db_uses_config_url_when_none(self, sqlite_config: Any) -> None:
        """When ``database_url`` is None, the config default is used."""
        from src.api.dependencies import AppResources

        mock_engine = MagicMock()
        mock_async_engine = MagicMock()
        with (
            patch("src.api.dependencies.create_engine", return_value=mock_engine),
            patch(
                "src.api.dependencies.create_async_engine",
                return_value=mock_async_engine,
            ),
            patch("src.api.dependencies.config", sqlite_config),
        ):
            r = AppResources()
            r.init_db(database_url=None)
            assert r.engine is mock_engine
            assert r.async_engine is mock_async_engine

    def test_init_process_pool(self) -> None:
        """``init_process_pool()`` creates a ``ProcessPoolExecutor``."""
        from src.api.dependencies import AppResources

        r = AppResources()
        r.init_process_pool(max_workers=1)
        assert r.process_pool is not None

    @pytest.mark.asyncio
    async def test_dispose_shuts_down_pool_and_engine(self) -> None:
        """``dispose()`` shuts down the process pool and disposes engines."""
        from src.api.dependencies import AppResources

        mock_engine = MagicMock()
        mock_async_engine = MagicMock()
        mock_async_engine.dispose = AsyncMock()
        mock_pool = MagicMock()
        with (
            patch("src.api.dependencies.create_engine", return_value=mock_engine),
            patch(
                "src.api.dependencies.create_async_engine",
                return_value=mock_async_engine,
            ),
        ):
            r = AppResources()
            r.init_process_pool(max_workers=1)
            r.process_pool = mock_pool
            r.init_db(database_url="sqlite:///:memory:")

            await r.dispose()
            mock_pool.shutdown.assert_called_once_with(wait=True)
            mock_engine.dispose.assert_called_once()
            mock_async_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispose_safe_when_not_initialized(self) -> None:
        """``dispose()`` is a no-op when resources were never initialised."""
        from src.api.dependencies import AppResources

        r = AppResources()
        await r.dispose()


# ---------------------------------------------------------------------------
# get_db
# ---------------------------------------------------------------------------


class TestGetDb:
    """Tests for the :func:`get_db` async generator."""

    @pytest.mark.asyncio
    async def test_yields_session_when_initialized(self) -> None:
        """An async generator yields a session and closes it on exit."""
        from src.api.dependencies import AppResources, get_db

        mock_engine = MagicMock()
        mock_async_engine = MagicMock()
        mock_session_factory = MagicMock()
        mock_session = MagicMock()

        with (
            patch("src.api.dependencies.create_engine", return_value=mock_engine),
            patch(
                "src.api.dependencies.create_async_engine",
                return_value=mock_async_engine,
            ),
            patch(
                "src.api.dependencies.sessionmaker",
                return_value=mock_session_factory,
            ),
        ):
            resources = AppResources()
            resources.init_db(database_url="sqlite:///:memory:")
            resources.session_factory = mock_session_factory
            mock_session_factory.return_value = mock_session

            with patch("src.api.dependencies.resources", resources):
                gen = get_db()
                session = await gen.__anext__()
                assert session is mock_session

                try:
                    await gen.__anext__()
                except StopAsyncIteration:
                    pass
                mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_not_initialized(self) -> None:
        """A ``RuntimeError`` is raised if the session factory is uninitialised."""
        from src.api.dependencies import AppResources, get_db

        with patch(
            "src.api.dependencies.resources",
            AppResources(),
        ):
            gen = get_db()
            with pytest.raises(RuntimeError, match="Database not initialised"):
                await gen.__anext__()


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    """Tests for :func:`get_current_user`."""

    @pytest.mark.asyncio
    async def test_raises_on_invalid_token(self) -> None:
        from src.api.dependencies import get_current_user

        mock_db = MagicMock()
        with (
            patch(
                "src.api.dependencies.decode_access_token",
                return_value=None,
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(token="invalid", session=mock_db)
            assert exc_info.value.status_code == 401
            assert "Could not validate credentials" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_raises_when_token_missing_sub(self) -> None:
        from src.api.dependencies import get_current_user

        mock_db = MagicMock()
        with (
            patch(
                "src.api.dependencies.decode_access_token",
                return_value={"role": "perito"},
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(token="no-sub-token", session=mock_db)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_raises_when_user_not_found(self) -> None:
        from src.api.dependencies import get_current_user

        mock_db = MagicMock()

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.scalar_one_or_none.return_value = None
            return result

        mock_db.execute = _execute

        with (
            patch(
                "src.api.dependencies.decode_access_token",
                return_value={"sub": "unknown_user"},
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(token="valid-token", session=mock_db)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_raises_when_user_inactive(self) -> None:
        from src.api.dependencies import get_current_user
        from src.db.models import User

        mock_db = MagicMock()
        inactive_user = MagicMock(spec=User)
        inactive_user.is_active = False

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.scalar_one_or_none.return_value = inactive_user
            return result

        mock_db.execute = _execute

        with (
            patch(
                "src.api.dependencies.decode_access_token",
                return_value={"sub": "inactive_user"},
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user(token="valid-token", session=mock_db)
            assert exc_info.value.status_code == 403
            assert "Inactive" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_returns_user_when_valid(self) -> None:
        from src.api.dependencies import get_current_user
        from src.db.models import User

        mock_db = MagicMock()
        active_user = MagicMock(spec=User)
        active_user.is_active = True
        active_user.username = "perito1"

        async def _execute(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.scalar_one_or_none.return_value = active_user
            return result

        mock_db.execute = _execute

        with (
            patch(
                "src.api.dependencies.decode_access_token",
                return_value={"sub": "perito1"},
            ),
        ):
            result = await get_current_user(token="valid-token", session=mock_db)
            assert result is active_user


# ---------------------------------------------------------------------------
# RequireRole
# ---------------------------------------------------------------------------


class TestRequireRole:
    """Tests for :class:`RequireRole`."""

    def test_passes_with_correct_role(self) -> None:
        """A user with an allowed role passes without raising."""
        from src.api.dependencies import RequireRole

        mock_user = MagicMock()
        mock_user.role = "Admin"

        check = RequireRole("Admin", "Supervisor")
        # Should not raise
        check(current_user=mock_user)

    def test_raises_with_wrong_role(self) -> None:
        """A user without any allowed role raises HTTP 403."""
        from src.api.dependencies import RequireRole

        mock_user = MagicMock()
        mock_user.role = "Auditor"

        check = RequireRole("Admin", "Perito")
        with pytest.raises(HTTPException) as exc_info:
            check(current_user=mock_user)
        assert exc_info.value.status_code == 403
        assert "not authorised" in exc_info.value.detail

    def test_raises_with_multiple_roles_required(self) -> None:
        """Error message lists all required roles."""
        from src.api.dependencies import RequireRole

        mock_user = MagicMock()
        mock_user.role = "Viewer"

        check = RequireRole("Admin", "Perito")
        with pytest.raises(HTTPException) as exc_info:
            check(current_user=mock_user)
        assert "Admin" in exc_info.value.detail
        assert "Perito" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


class TestLifespan:
    """Tests for the :func:`lifespan` async context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_initialises_and_disposes(self, sqlite_config: Any) -> None:
        """The lifespan context manager calls init and dispose."""
        from src.api.dependencies import AppResources, lifespan

        test_resources = AppResources()

        init_db_called = False
        init_pool_called = False
        dispose_called = False

        def _track_init_db(*args: object, **kwargs: object) -> None:
            nonlocal init_db_called
            init_db_called = True
            # Must also set async engine to keep init_db from failing
            test_resources.async_engine = MagicMock()
            test_resources.async_session_factory = MagicMock()

        def _track_init_pool(*args: object, **kwargs: object) -> None:
            nonlocal init_pool_called
            init_pool_called = True

        async def _track_dispose() -> None:
            nonlocal dispose_called
            dispose_called = True

        with (
            patch.object(test_resources, "init_db", _track_init_db),
            patch.object(test_resources, "init_process_pool", _track_init_pool),
            patch.object(test_resources, "dispose", _track_dispose),
            patch("src.api.dependencies.config", sqlite_config),
            patch("src.api.dependencies.resources", test_resources),
        ):
            async with lifespan(MagicMock()):
                pass

        assert init_db_called, "init_db should be called on startup"
        assert init_pool_called, "init_process_pool should be called on startup"
        assert dispose_called, "dispose should be called on shutdown"

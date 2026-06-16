"""
Integration tests for the auth router (``/api/v1/auth``).
Mocked async session with ``execute`` returning filterable Users.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_async_db, get_current_user
from src.db.models import User


def _make_app(mock_session: MagicMock | None = None) -> FastAPI:
    """Construct a minimal FastAPI with only the auth router."""
    from src.api.routers.auth import router

    application = FastAPI()
    if mock_session is not None:
        application.dependency_overrides[get_async_db] = lambda: mock_session
    application.include_router(router)
    return application


def _make_mock_user(
    username: str = "perito1",
    role: str = "Perito",
    is_active: bool = True,
    email: str = "perito1@forense.gob.ni",
    full_name: str = "Perito Uno",
    hashed_password: str | None = None,
) -> MagicMock:
    from src.services.auth_service import get_password_hash

    user = MagicMock(spec=User)
    user.id = "550e8400-e29b-41d4-a716-446655440000"
    user.username = username
    user.email = email
    user.full_name = full_name
    user.role = role
    user.is_active = is_active
    user.hashed_password = hashed_password or get_password_hash("password123")
    return user


def _make_mock_session(user: MagicMock | None = None) -> MagicMock:
    """Return a mock with async execute() returning the user."""
    mock_db = MagicMock()

    async def _execute(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.scalar_one_or_none.return_value = user
        return result

    mock_db.execute = _execute
    return mock_db


@pytest.mark.asyncio
class TestLogin:
    """``POST /api/v1/auth/login`` — authenticate and receive JWT."""

    async def test_returns_token_on_valid_credentials(self) -> None:
        from src.services.auth_service import get_password_hash

        user = _make_mock_user(hashed_password=get_password_hash("password123"))
        mock_db = _make_mock_session(user=user)
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "perito1", "password": "password123"},
            )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["role"] == "Perito"
        assert data["username"] == "perito1"

    async def test_returns_401_on_unknown_user(self) -> None:
        mock_db = _make_mock_session(user=None)
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "unknown", "password": "password123"},
            )

        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    async def test_returns_401_on_wrong_password(self) -> None:
        user = _make_mock_user()
        mock_db = _make_mock_session(user=user)
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "perito1", "password": "wrong_password"},
            )

        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    async def test_returns_403_on_inactive_user(self) -> None:
        user = _make_mock_user(is_active=False)
        mock_db = _make_mock_session(user=user)
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "perito1", "password": "password123"},
            )

        assert response.status_code == 403
        assert "inactive" in response.json()["detail"].lower()

    async def test_rejects_missing_username(self) -> None:
        mock_db = _make_mock_session()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login", data={"password": "password123"},
            )

        assert response.status_code == 422

    async def test_rejects_missing_password(self) -> None:
        mock_db = _make_mock_session()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login", data={"username": "perito1"},
            )

        assert response.status_code == 422


@pytest.mark.asyncio
class TestReadCurrentUser:
    """``GET /api/v1/auth/me`` — authenticated user profile."""

    async def test_returns_user_profile_with_valid_token(self) -> None:
        mock_user = _make_mock_user()
        app = _make_app()
        app.dependency_overrides[get_current_user] = lambda: mock_user
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/auth/me",
                headers={"Authorization": "Bearer valid-token"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "perito1"
        assert data["email"] == "perito1@forense.gob.ni"
        assert data["role"] == "Perito"
        assert data["is_active"] is True
        assert data["full_name"] == "Perito Uno"

    async def test_returns_401_without_token(self) -> None:
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/auth/me")

        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

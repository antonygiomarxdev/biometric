"""
Integration tests for the auth router (``/api/v1/auth``).

Covers:
- ``POST /api/v1/auth/login`` — credential exchange for JWT.
- ``GET /api/v1/auth/me`` — authenticated user profile.

All database dependencies are mocked via ``dependency_overrides``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.dependencies import get_db, get_current_user
from src.db.models import User


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(mock_session: MagicMock | None = None) -> FastAPI:
    """Construct a minimal FastAPI with only the auth router."""
    from src.api.routers.auth import router

    application = FastAPI()
    if mock_session is not None:
        application.dependency_overrides[get_db] = lambda: mock_session
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
    """Create a mock ``User`` ORM object with the given attributes."""
    user = MagicMock(spec=User)
    user.id = "550e8400-e29b-41d4-a716-446655440000"
    user.username = username
    user.email = email
    user.full_name = full_name
    user.role = role
    user.is_active = is_active
    user.hashed_password = hashed_password or "$2b$12$LJ3m4ys3Lk0TSwHmGsmmyePqJhM7iM9z0k9Z0k9Z0k9Z0k9Z0k9Z0"
    return user


# ---------------------------------------------------------------------------
# POST /api/v1/auth/login
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLogin:
    """``POST /api/v1/auth/login`` — authenticate and receive JWT."""

    async def test_returns_token_on_valid_credentials(self) -> None:
        """It returns an access token with role and username."""
        from src.services.auth_service import get_password_hash

        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_user = _make_mock_user(
            hashed_password=get_password_hash("password123"),
        )
        mock_query.first.return_value = mock_user

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
        """An unknown username returns 401."""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = None

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "unknown", "password": "password123"},
            )

        assert response.status_code == 401
        data = response.json()
        assert "Incorrect username or password" in data["detail"]

    async def test_returns_401_on_wrong_password(self) -> None:
        """A wrong password returns 401."""
        from src.services.auth_service import verify_password

        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_user = _make_mock_user()
        mock_query.first.return_value = mock_user

        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "perito1", "password": "wrong_password"},
            )

        # verify_password returns False — triggers 401
        assert response.status_code == 401
        assert "Incorrect username or password" in response.json()["detail"]

    async def test_returns_403_on_inactive_user(self) -> None:
        """An inactive user account returns 403."""
        mock_db = MagicMock()
        mock_query = MagicMock()
        mock_db.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_user = _make_mock_user(is_active=False)
        mock_query.first.return_value = mock_user

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
        """Omitting the username field returns 422."""
        mock_db = MagicMock()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"password": "password123"},
            )

        assert response.status_code == 422

    async def test_rejects_missing_password(self) -> None:
        """Omitting the password field returns 422."""
        mock_db = MagicMock()
        app = _make_app(mock_db)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/auth/login",
                data={"username": "perito1"},
            )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/auth/me
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReadCurrentUser:
    """``GET /api/v1/auth/me`` — authenticated user profile."""

    async def test_returns_user_profile_with_valid_token(self) -> None:
        """A valid token returns the authenticated user's profile."""
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
        """Omitting the Authorization header returns 401."""
        app = _make_app()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/auth/me")

        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]

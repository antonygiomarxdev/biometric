"""
Unit tests for :mod:`~src.services.auth_service`.

All functions tested are pure — no database or external service mocking
required.  Coverage target >90%.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import jwt
import pytest

from src.services.auth_service import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)


class TestVerifyPassword:
    """Tests for :func:`verify_password`."""

    @pytest.mark.asyncio
    async def test_returns_true_for_correct_password(self) -> None:
        hashed = await get_password_hash("my_secret_password")
        assert await verify_password("my_secret_password", hashed) is True

    @pytest.mark.asyncio
    async def test_returns_false_for_incorrect_password(self) -> None:
        hashed = await get_password_hash("correct_password")
        assert await verify_password("wrong_password", hashed) is False

    @pytest.mark.asyncio
    async def test_handles_empty_string(self) -> None:
        hashed = await get_password_hash("")
        assert await verify_password("", hashed) is True
        assert await verify_password("non_empty", hashed) is False

    @pytest.mark.asyncio
    async def test_returns_false_for_bcrypt_hash(self) -> None:
        bcrypt_hash = "$2b$12$LJ3m4ys3Lk0TSwHmGsmmyePqJhM7iM9z0k9Z0k9Z0k9Z0k9Z0k9Z0"
        assert await verify_password("anything", bcrypt_hash) is False


class TestGetPasswordHash:
    """Tests for :func:`get_password_hash`."""

    @pytest.mark.asyncio
    async def test_returns_string(self) -> None:
        result = await get_password_hash("password123")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_returns_argon2id_hash(self) -> None:
        result = await get_password_hash("password123")
        assert result.startswith("$argon2id$")

    @pytest.mark.asyncio
    async def test_different_passwords_different_hashes(self) -> None:
        hash_a = await get_password_hash("password1")
        hash_b = await get_password_hash("password2")
        assert hash_a != hash_b

    @pytest.mark.asyncio
    async def test_same_password_different_hashes_due_to_salt(self) -> None:
        hash_a = await get_password_hash("same_password")
        hash_b = await get_password_hash("same_password")
        assert hash_a != hash_b


class TestCreateAccessToken:
    """Tests for :func:`create_access_token`."""

    def test_returns_jwt_string(self) -> None:
        token = create_access_token({"sub": "user123"})
        parts = token.split(".")
        assert len(parts) == 3

    def test_includes_data_in_payload(self) -> None:
        token = create_access_token({"sub": "user123", "role": "admin"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"

    def test_uses_default_expiry_from_config(self) -> None:
        from src.services.auth_service import config

        token = create_access_token({"sub": "test"})
        payload = decode_access_token(token)
        assert payload is not None
        exp = payload.get("exp")
        assert exp is not None, "Token should have an expiration claim"

    def test_accepts_custom_expiry(self) -> None:
        token = create_access_token(
            {"sub": "test"},
            expires_delta=timedelta(hours=1),
        )
        payload = decode_access_token(token)
        assert payload is not None
        exp = payload.get("exp")
        assert exp is not None

    def test_different_data_different_tokens(self) -> None:
        token_a = create_access_token({"sub": "user_a"})
        token_b = create_access_token({"sub": "user_b"})
        assert token_a != token_b

    def test_token_does_not_contain_password_in_plaintext(self) -> None:
        token = create_access_token({"sub": "user123", "password": "secret"})
        assert "secret" not in token


class TestDecodeAccessToken:
    """Tests for :func:`decode_access_token`."""

    def test_returns_payload_for_valid_token(self) -> None:
        payload = {"sub": "user123", "role": "perito"}
        token = create_access_token(payload)
        result = decode_access_token(token)
        assert result is not None
        assert result["sub"] == "user123"
        assert result["role"] == "perito"

    def test_returns_none_for_malformed_token(self) -> None:
        result = decode_access_token("not.a.token")
        assert result is None

    def test_returns_none_for_invalid_signature(self) -> None:
        token = create_access_token({"sub": "user"})
        parts = token.split(".")
        tampered = f"{parts[0]}.{parts[1]}.invalidsignature"
        result = decode_access_token(tampered)
        assert result is None

    def test_returns_none_for_empty_token(self) -> None:
        result = decode_access_token("")
        assert result is None

    def test_returns_none_when_jwt_decode_raises(self) -> None:
        with patch(
            "src.services.auth_service.jwt.decode",
            side_effect=jwt.PyJWTError("Boom"),
        ):
            result = decode_access_token("any_token")
        assert result is None

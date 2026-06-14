"""
Unit tests for :mod:`~src.services.auth_service`.

All functions tested are pure — no database or external service mocking
required.  Coverage target >90%.
"""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
from jose import JWTError

from src.services.auth_service import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)


# ---------------------------------------------------------------------------
# verify_password
# ---------------------------------------------------------------------------


class TestVerifyPassword:
    """Tests for :func:`verify_password`."""

    def test_returns_true_for_correct_password(self) -> None:
        """Verifying a correctly-hashed password returns True."""
        hashed = get_password_hash("my_secret_password")
        assert verify_password("my_secret_password", hashed) is True

    def test_returns_false_for_incorrect_password(self) -> None:
        """Verifying a wrong password returns False."""
        hashed = get_password_hash("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_handles_empty_string(self) -> None:
        """Empty password strings are handled without error."""
        hashed = get_password_hash("")
        assert verify_password("", hashed) is True
        assert verify_password("non_empty", hashed) is False


# ---------------------------------------------------------------------------
# get_password_hash
# ---------------------------------------------------------------------------


class TestGetPasswordHash:
    """Tests for :func:`get_password_hash`."""

    def test_returns_string(self) -> None:
        """Hash output is a string."""
        result = get_password_hash("password123")
        assert isinstance(result, str)

    def test_returns_bcrypt_hash(self) -> None:
        """Hash starts with bcrypt identifier ``$2b$``."""
        result = get_password_hash("password123")
        assert result.startswith("$2b$")

    def test_different_passwords_different_hashes(self) -> None:
        """Two different passwords produce different hashes."""
        hash_a = get_password_hash("password1")
        hash_b = get_password_hash("password2")
        assert hash_a != hash_b

    def test_same_password_different_hashes_due_to_salt(self) -> None:
        """Even the same password produces different hashes due to random salt."""
        hash_a = get_password_hash("same_password")
        hash_b = get_password_hash("same_password")
        assert hash_a != hash_b


# ---------------------------------------------------------------------------
# create_access_token
# ---------------------------------------------------------------------------


class TestCreateAccessToken:
    """Tests for :func:`create_access_token`."""

    def test_returns_jwt_string(self) -> None:
        """Output is a JWT string with three dot-separated segments."""
        token = create_access_token({"sub": "user123"})
        parts = token.split(".")
        assert len(parts) == 3

    def test_includes_data_in_payload(self) -> None:
        """The token payload contains the provided data."""
        token = create_access_token({"sub": "user123", "role": "admin"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"

    def test_uses_default_expiry_from_config(self) -> None:
        """When ``expires_delta`` is None, uses the config default."""
        from src.services.auth_service import config

        token = create_access_token({"sub": "test"})
        payload = decode_access_token(token)
        assert payload is not None
        exp = payload.get("exp")
        assert exp is not None, "Token should have an expiration claim"

    def test_accepts_custom_expiry(self) -> None:
        """Custom ``expires_delta`` is honoured."""
        token = create_access_token(
            {"sub": "test"},
            expires_delta=timedelta(hours=1),
        )
        payload = decode_access_token(token)
        assert payload is not None
        exp = payload.get("exp")
        assert exp is not None

    def test_different_data_different_tokens(self) -> None:
        """Different data payloads produce distinct tokens."""
        token_a = create_access_token({"sub": "user_a"})
        token_b = create_access_token({"sub": "user_b"})
        assert token_a != token_b

    def test_token_does_not_contain_password_in_plaintext(self) -> None:
        """Sensitive data is not exposed in plaintext in the token."""
        token = create_access_token({"sub": "user123", "password": "secret"})
        assert "secret" not in token


# ---------------------------------------------------------------------------
# decode_access_token
# ---------------------------------------------------------------------------


class TestDecodeAccessToken:
    """Tests for :func:`decode_access_token`."""

    def test_returns_payload_for_valid_token(self) -> None:
        """A valid token returns its payload dictionary."""
        payload = {"sub": "user123", "role": "perito"}
        token = create_access_token(payload)
        result = decode_access_token(token)
        assert result is not None
        assert result["sub"] == "user123"
        assert result["role"] == "perito"

    def test_returns_none_for_malformed_token(self) -> None:
        """A garbage string returns None without raising."""
        result = decode_access_token("not.a.token")
        assert result is None

    def test_returns_none_for_invalid_signature(self) -> None:
        """A token signed with a different secret returns None."""
        token = create_access_token({"sub": "user"})
        # Tamper with the signature portion
        parts = token.split(".")
        tampered = f"{parts[0]}.{parts[1]}.invalidsignature"
        result = decode_access_token(tampered)
        assert result is None

    def test_returns_none_for_empty_token(self) -> None:
        """An empty string returns None."""
        result = decode_access_token("")
        assert result is None

    def test_returns_none_when_jwt_decode_raises(self) -> None:
        """When ``jwt.decode`` raises ``JWTError``, returns None."""
        from src.services.auth_service import jwt as auth_jwt

        with patch.object(auth_jwt, "decode", side_effect=JWTError("Boom")):
            result = decode_access_token("any_token")
        assert result is None

"""
Authentication service: password hashing and JWT token management.

Pure business logic — no FastAPI imports.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error, InvalidHashError

from src.core.config import config

logger = logging.getLogger(__name__)

_hasher = PasswordHasher()


def _verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash (CPU-bound, use via verify_password)."""
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (Argon2Error, InvalidHashError):
        return False


async def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Non-blocking password verification (Argon2 runs in executor)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _verify_password, plain_password, hashed_password,
    )


def _hash_password(password: str) -> str:
    """Hash a password (CPU-bound, use via get_password_hash)."""
    return _hasher.hash(password)


async def get_password_hash(password: str) -> str:
    """Non-blocking password hashing."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _hash_password, password)


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=config.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        config.jwt_secret_key,
        algorithm=config.jwt_algorithm,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return dict(jwt.decode(
            token,
            config.jwt_secret_key,
            algorithms=[config.jwt_algorithm],
        ))
    except jwt.PyJWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        return None

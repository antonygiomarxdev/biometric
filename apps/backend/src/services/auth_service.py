"""
Authentication service: password hashing, JWT token management, and
FastAPI security dependencies for RBAC.

Provides:
- ``verify_password`` / ``get_password_hash`` — bcrypt via passlib
- ``create_access_token`` / ``decode_access_token`` — JWT via python-jose
- ``get_current_user`` — FastAPI dependency that extracts and validates
  the Bearer token from the ``Authorization`` header.
- ``RequireRole`` — closure that returns a dependency enforcing role
  membership (e.g. ``RequireRole("Admin", "Perito")``).
"""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from src.core.config import config
from src.api.dependencies import get_db
from src.db.models import User

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password with bcrypt (salt rounds = 12 by default)."""
    return pwd_context.hash(password)


# ---------------------------------------------------------------------------
# JWT token management
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT access token.

    The token payload includes the ``sub`` (subject) claim plus any
    additional data.  Expiration is set to
    ``config.jwt_access_token_expire_minutes`` by default.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=config.jwt_access_token_expire_minutes)
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        config.jwt_secret_key,
        algorithm=config.jwt_algorithm,
    )
    return encoded_jwt


def decode_access_token(token: str) -> dict[str, Any] | None:
    """
    Decode and validate a JWT access token.

    Returns the payload dict on success, or ``None`` if the token is
    expired, malformed, or otherwise invalid.
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            config.jwt_secret_key,
            algorithms=[config.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# FastAPI dependencies — authentication
# ---------------------------------------------------------------------------


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency that extracts and validates the current user
    from the Bearer token.

    Raises ``HTTPException(401)`` if the token is missing, invalid,
    expired, or if the referenced user does not exist or is inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    username: str | None = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )

    return user


# ---------------------------------------------------------------------------
# FastAPI dependencies — role-based authorisation
# ---------------------------------------------------------------------------


class RequireRole:
    """
    Callable class that creates a FastAPI dependency enforcing that
    the current user has one of the specified roles.

    Usage::

        from src.services.auth_service import RequireRole

        @router.get("/cases", dependencies=[Depends(RequireRole("Admin", "Perito"))])
        async def list_cases(...):
            ...

    Or inline in a route::

        @router.get("/audit")
        async def view_audit(
            current_user: User = Depends(get_current_user),
            _: None = Depends(RequireRole("Admin")),
        ):
            ...
    """

    def __init__(self, *roles: str) -> None:
        self._roles = roles

    def __call__(self, current_user: User = Depends(get_current_user)) -> None:
        if current_user.role not in self._roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Role '{current_user.role}' is not authorised. "
                    f"Required one of: {', '.join(self._roles)}"
                ),
            )

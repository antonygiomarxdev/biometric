"""
Authentication router — login endpoint and token issuance.

Per D-02, D-03:
  - Mounted under ``/api/v1/auth``.
  - Uses ``OAuth2PasswordRequestForm`` for standard password flow.

Endpoints:
  - ``POST /login`` — Exchange credentials for a signed JWT access token.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, get_current_user
from src.services.auth_service import create_access_token, verify_password, get_password_hash
from src.core.config import config
from src.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/auth",
    tags=["auth"],
)


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> dict:
    """
    Authenticate a user and return a JWT access token.

    Accepts ``username`` and ``password`` form fields (per the
    OAuth2 password flow).  On success returns a signed token with
    the user's ``sub`` (username) and ``role`` claims.

    **Example request (form-encoded):**

        username=perito1&password=secreta

    **Example response:**

    .. code-block:: json

        {
            "access_token": "eyJhbGci...",
            "token_type": "bearer",
            "role": "Perito",
            "username": "perito1"
        }
    """
    # Look up the user by username
    user = (
        db.query(User)
        .filter(User.username == form_data.username)
        .first()
    )

    if user is None:
        logger.warning("Login failed: unknown user '%s'", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        logger.warning("Login blocked: inactive user '%s'", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    if not verify_password(form_data.password, user.hashed_password):
        logger.warning("Login failed: wrong password for '%s'", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Issue access token with username as subject and role claim
    access_token_expires = timedelta(
        minutes=config.jwt_access_token_expire_minutes
    )
    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
        },
        expires_delta=access_token_expires,
    )

    logger.info(
        "Login successful: user='%s' role='%s'",
        user.username,
        user.role,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "role": user.role,
        "username": user.username,
    }


@router.get("/me")
async def read_current_user(
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Return the profile of the currently authenticated user.

    Requires a valid Bearer token in the ``Authorization`` header.
    """
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "is_active": current_user.is_active,
    }

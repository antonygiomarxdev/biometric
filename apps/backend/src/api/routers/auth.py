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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_async_db, get_current_user
from src.api.prefix import API_PREFIX
from src.core.config import config
from src.db.models import User
from src.services.auth_service import create_access_token, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix=f"{API_PREFIX}/auth",
    tags=["auth"],
)


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_async_db),
) -> dict[str, Any]:
    """
    Authenticate a user and return a JWT access token.
    """
    result = await session.execute(
        select(User).where(User.username == form_data.username)
    )
    user = result.scalar_one_or_none()

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

    if not await verify_password(form_data.password, user.hashed_password):
        logger.warning("Login failed: wrong password for '%s'", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

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
) -> dict[str, Any]:
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

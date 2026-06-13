"""
FastAPI dependency injection (DI) providers and lifespan management.

Per D-04: Eliminate global singletons. Use FastAPI Depends() with
resources scoped to a lifespan manager.

Provides:
- ``get_db`` — async generator yielding a SQLAlchemy session.
- ``lifespan`` — async context manager that initialises the DB engine
  and a ``ProcessPoolExecutor`` for CPU-bound fingerprint matching,
  then disposes both on shutdown (per D-12).
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import config

logger = logging.getLogger(__name__)


class AppResources:
    """
    Container for application-scoped resources.

    Initialised by the lifespan manager and accessed via Depends().
    This replaces the previous pattern of module-level singletons.
    """

    def __init__(self) -> None:
        self.engine: Engine | None = None
        self.session_factory: sessionmaker[Session] | None = None
        self.process_pool: ProcessPoolExecutor | None = None

    def init_db(self, database_url: str | None = None) -> None:
        """Create the SQLAlchemy engine and session factory."""
        url = database_url or config.database_url
        self.engine = create_engine(
            url,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_pre_ping=True,
            echo=config.log_level == "DEBUG",
        )
        self.session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )
        logger.info(
            "Database engine initialised (pool_size=%s, max_overflow=%s)",
            config.db_pool_size,
            config.db_max_overflow,
        )

    def init_process_pool(self, max_workers: int | None = None) -> None:
        """
        Create the ``ProcessPoolExecutor`` for CPU-bound tasks.

        Used by fingerprint matching and image processing endpoints
        to avoid blocking the ASGI event loop (D-12).
        """
        self.process_pool = ProcessPoolExecutor(
            max_workers=max_workers,
        )
        logger.info(
            "ProcessPoolExecutor initialised (max_workers=%s)",
            max_workers,
        )

    def dispose(self) -> None:
        """Gracefully shut down all resources."""
        if self.process_pool is not None:
            self.process_pool.shutdown(wait=True)
            logger.info("ProcessPoolExecutor shut down")

        if self.engine is not None:
            self.engine.dispose()
            logger.info("Database engine disposed")


# Singleton — but owned and managed by the lifespan, not by the module.
# Modules import ``resources`` but never call ``init_*`` directly.
resources = AppResources()


async def get_db() -> AsyncGenerator[Session, None]:
    """
    FastAPI dependency that yields a SQLAlchemy session.

    Usage::

        from fastapi import Depends
        from sqlalchemy.orm import Session

        @router.get("/cases")
        async def list_cases(db: Session = Depends(get_db)):
            ...

    The session is automatically closed when the request finishes.
    """
    if resources.session_factory is None:
        raise RuntimeError(
            "Database not initialised. Ensure the lifespan "
            "context manager has been installed on the FastAPI app."
        )

    session = resources.session_factory()
    try:
        yield session
    finally:
        session.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Async lifespan context manager for the FastAPI application.

    Startup (before ``yield``):
      - Initialise the database engine and session factory.
      - Create a ``ProcessPoolExecutor`` for CPU-bound tasks.

    Shutdown (after ``yield``):
      - Shut down the process pool (wait for running tasks).
      - Dispose of the database engine.

    Install on the FastAPI app::

        app = FastAPI(lifespan=lifespan)

    Or compose multiple lifespan managers with
    ``contextlib.AsyncExitStack``.
    """
    logger.info("Starting up — initialising application resources")

    # ---- startup ----
    resources.init_db()
    resources.init_process_pool()

    logger.info("Application ready")

    try:
        yield
    finally:
        # ---- shutdown ----
        logger.info("Shutting down — disposing application resources")
        resources.dispose()
        logger.info("Shutdown complete")

"""
FastAPI security dependencies: authentication and role-based access control.

Separated from auth_service.py to maintain Clean Architecture (services
must not depend on the API layer).
"""

import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from src.db.models import User
from src.services.auth_service import decode_access_token

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
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


class RequireRole:
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

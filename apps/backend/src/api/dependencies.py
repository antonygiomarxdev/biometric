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

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from src.ai.loader import ModelLoader
from src.core.config import config
from src.db.models import User
from src.services.auth_service import decode_access_token

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from src.db.qdrant_embedding_repository import QdrantEmbeddingRepository
    from src.services.embedding_service import EmbeddingService

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
        self.async_engine: AsyncEngine | None = None
        self.async_session_factory: async_sessionmaker[AsyncSession] | None = None
        self.process_pool: ProcessPoolExecutor | None = None

    def init_db(self, database_url: str | None = None) -> None:
        """Create SQLAlchemy engines (sync + async) and session factories."""
        url = database_url or config.database_url
        # Sync engine (legacy)
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
        # Async engine (modern)
        async_url = config.async_database_url
        self.async_engine = create_async_engine(
            async_url,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_pre_ping=True,
            echo=config.log_level == "DEBUG",
        )
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            expire_on_commit=False,
        )
        logger.info(
            "Database engines initialised (pool_size=%s, max_overflow=%s)",
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

    async def dispose(self) -> None:
        """Gracefully shut down all resources.

        Order matters: stop accepting new requests first (the
        ProcessPoolExecutor drains in-flight subprocesses), then
        release network/IO resources (Qdrant, MinIO, ModelLoader
        pool), and only then dispose of the DB engines.  Reversing
        the order can leave async DB sessions in flight when the
        engine is torn down.
        """
        if self.process_pool is not None:
            self.process_pool.shutdown(wait=True)
            logger.info("ProcessPoolExecutor shut down")

        from src.storage.object_storage import storage
        storage.close()
        logger.info("MinIO client released")

        await close_qdrant_repo()
        logger.info("Qdrant client closed")

        loader = get_model_loader_cached()
        await loader.shutdown()
        logger.info("ModelLoader pool shut down")

        if self.engine is not None:
            self.engine.dispose()
            logger.info("Sync database engine disposed")

        if self.async_engine is not None:
            await self.async_engine.dispose()
            logger.info("Async database engine disposed")


# Singleton — but owned and managed by the lifespan, not by the module.
resources = AppResources()


async def get_db() -> AsyncGenerator[Session, None]:
    """Legacy sync session (deprecated — new code should use ``get_async_db``)."""
    if resources.session_factory is None:
        msg = (
            "Database not initialised. Ensure the lifespan "
            "context manager has been installed on the FastAPI app."
        )
        raise RuntimeError(msg)

    session = resources.session_factory()
    try:
        yield session
    finally:
        session.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async session dependency — use for all new code."""
    if resources.async_session_factory is None:
        msg = (
            "Async database not initialised. Ensure the lifespan "
            "context manager has been installed on the FastAPI app."
        )
        raise RuntimeError(msg)

    async with resources.async_session_factory() as session:
        yield session


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
        await resources.dispose()
        logger.info("Shutdown complete")

# ---------------------------------------------------------------------------
# Deep Embedding providers (Phase 29)
# ---------------------------------------------------------------------------


_embedding_model_loader: ModelLoader | None = None
_qdrant_repo: "QdrantEmbeddingRepository | None" = None


def get_model_loader_cached() -> ModelLoader:
    global _embedding_model_loader
    if _embedding_model_loader is None:
        _embedding_model_loader = ModelLoader()
    return _embedding_model_loader


def get_model_loader() -> ModelLoader:
    return get_model_loader_cached()


def get_embedding_repository() -> "QdrantEmbeddingRepository":
    """Cached singleton — the underlying Qdrant client owns an HTTP/2
    connection pool and should be reused across requests, not
    re-created per call.  The instance is closed in
    ``AppResources.dispose`` (called from the FastAPI lifespan).
    """
    global _qdrant_repo
    if _qdrant_repo is None:
        from qdrant_client import QdrantClient

        from src.db.qdrant_embedding_repository import QdrantEmbeddingRepository

        client = QdrantClient(host="localhost", port=6333)
        _qdrant_repo = QdrantEmbeddingRepository(client)
        _qdrant_repo.ensure_collection()
    return _qdrant_repo


async def close_qdrant_repo() -> None:
    global _qdrant_repo
    if _qdrant_repo is not None:
        _qdrant_repo.close()
        _qdrant_repo = None


def get_embedding_service(
    loader: ModelLoader = Depends(get_model_loader),
    repo: "QdrantEmbeddingRepository" = Depends(get_embedding_repository),
) -> EmbeddingService:
    from src.services.embedding_service import EmbeddingService

    return EmbeddingService(loader, repo)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_async_db),
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

    result = await session.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()
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

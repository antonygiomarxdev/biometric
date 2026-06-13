# NOTE: _VectorRecord is an internal auto-increment table, distinct from db.models.FingerprintVector (UUID PK).
"""Vector index with pgvector for similarity search."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from pgvector.sqlalchemy import Vector  # type: ignore[import-untyped]
from sqlalchemy import Column, Integer, text

from src.core.config import config
from src.core.metrics import timed
from src.storage.database import Base

if TYPE_CHECKING:
    from src.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class _VectorRecord(Base):
    """Internal auto-increment vector storage for pgvector similarity search."""

    __tablename__ = "fingerprint_vectors"

    id: Column[int] = Column(Integer, primary_key=True, autoincrement=True)
    embedding: Column[list[float]] = Column(Vector(256))  # 256-dimensional vector

    def __repr__(self):
        return f"<_VectorRecord(id={self.id})>"


class VectorIndex:
    """Vector index using pgvector on PostgreSQL."""

    def __init__(
        self, dimension: int | None = None, db_manager: DatabaseManager | None = None
    ):
        """
        Args:
            dimension: Vector dimension (default: config.vector_dimension)
            db_manager: Database manager (uses global if None)
        """
        from src.storage.database import db_manager as default_manager

        self.dimension = dimension or config.vector_dimension
        self.db_manager: DatabaseManager = db_manager or default_manager  # noqa: UP037

        # Try to initialize extension and index, but don't fail if there are issues
        try:
            self._ensure_extension()
            self._ensure_index()
        except Exception as e:
            # Don't block instance creation
            logger.debug(
                f"VectorIndex: Error en inicialización (se reintentará cuando se use): {e}"
            )

    def _ensure_extension(self):
        """Ensure the pgvector extension is enabled."""
        try:
            session = self.db_manager.get_session()
            try:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                session.commit()
                logger.info("Extensión pgvector verificada")
            except Exception as e:
                logger.warning(f"No se pudo crear extensión vector: {e}")
                logger.debug(
                    "Esto puede ser normal si la base de datos no está disponible o la extensión ya existe"
                )
                session.rollback()
                raise  # Re-lanzar para que __init__ pueda manejarlo
            finally:
                session.close()
        except Exception as e:
            # If there's a connection error, raise it to be handled in __init__
            logger.debug(f"Error al conectar con BD para verificar extensión: {e}")
            raise

    def _ensure_index(self):
        """Create the IVFFlat index if it doesn't exist."""
        session = self.db_manager.get_session()
        try:
            # First verify the table exists
            table_exists = session.execute(
                text(
                    """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'fingerprint_vectors'
                )
            """
                )
            ).scalar()

            if not table_exists:
                logger.debug(
                    "La tabla fingerprint_vectors aún no existe, se creará cuando sea necesaria"
                )
                return

            # Check if the index already exists
            result = session.execute(
                text(
                    """
                SELECT 1 FROM pg_indexes
                WHERE tablename = 'fingerprint_vectors'
                AND indexname = 'fingerprint_vectors_embedding_idx'
            """
                )
            ).fetchone()

            if not result:
                # Create IVFFlat index for fast searches
                # lists = sqrt(total_rows) is a good heuristic
                session.execute(
                    text(
                        """
                    CREATE INDEX IF NOT EXISTS fingerprint_vectors_embedding_idx
                    ON fingerprint_vectors
                    USING ivfflat (embedding vector_l2_ops)
                    WITH (lists = 100)
                """
                    )
                )
                session.commit()
                logger.info("Índice IVFFlat creado para pgvector")
        except Exception as e:
            logger.warning(f"No se pudo crear índice: {e}")
            logger.debug(
                "Esto puede ser normal si la base de datos no está disponible o el índice ya existe"
            )
            session.rollback()
        finally:
            session.close()

    @timed("index_add_vector")
    def add(self, vector: np.ndarray) -> int:
        """Add a vector to the index.

        Args:
            vector: Numpy vector of dimension (dimension,)

        Returns:
            ID of the added vector
        """
        # Ensure correct dimension
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector)

        vector = np.asarray(vector, dtype=np.float32)

        session = self.db_manager.get_session()
        try:
            vec_record = _VectorRecord(embedding=vector.tolist())
            session.add(vec_record)
            session.commit()
            session.refresh(vec_record)
            return int(vec_record.id)
        except Exception as e:
            session.rollback()
            logger.error(f"Error añadiendo vector: {e}")
            raise
        finally:
            session.close()

    @timed("index_search")
    def search(self, vector: np.ndarray, k: int = 1) -> tuple[list[int], list[float]]:
        """Search the K most similar vectors using L2 distance.

        Args:
            vector: Query vector
            k: Number of results to return

        Returns:
            Tuple of (IDs, distances)
        """
        # Ensure correct dimension
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector)

        vector = np.asarray(vector, dtype=np.float32)
        vector_list = vector.tolist()

        session = self.db_manager.get_session()
        try:
            # K-NN search with <-> operator (L2 distance)
            results = session.execute(
                text(
                    """
                    SELECT id, embedding <-> :vector as distance
                    FROM fingerprint_vectors
                    ORDER BY embedding <-> :vector
                    LIMIT :k
                """
                ),
                {"vector": str(vector_list), "k": k},
            ).fetchall()

            if not results:
                return [], []

            ids = [row[0] for row in results]
            distances = [float(row[1]) for row in results]

            return ids, distances

        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return [], []
        finally:
            session.close()

    def _pad_or_truncate(self, vector: np.ndarray) -> np.ndarray:
        """Adjust the vector to the correct dimension."""
        if len(vector) > self.dimension:
            return vector[: self.dimension]
        elif len(vector) < self.dimension:
            padded = np.zeros(self.dimension, dtype=np.float32)
            padded[: len(vector)] = vector
            return padded
        return vector

    def size(self) -> int:
        """Return the number of vectors in the index."""
        session = self.db_manager.get_session()
        try:
            count = session.query(_VectorRecord).count()
            return count
        finally:
            session.close()

    def reset(self):
        """Remove all vectors from the index."""
        session = self.db_manager.get_session()
        try:
            session.query(_VectorRecord).delete()
            session.commit()
            logger.info("Índice vectorial reseteado")
        except Exception as e:
            session.rollback()
            logger.error(f"Error reseteando índice: {e}")
            raise
        finally:
            session.close()

    def get_by_id(self, vector_id: int) -> np.ndarray | None:
        """Retrieve a vector by its ID."""
        session = self.db_manager.get_session()
        try:
            record = (
                session.query(_VectorRecord)
                .filter(_VectorRecord.id == vector_id)
                .first()
            )

            if record:
                return np.array(record.embedding, dtype=np.float32)
            return None
        finally:
            session.close()

    def get_batch_by_ids(self, vector_ids: list[int]) -> list[np.ndarray | None]:
        """Retrieve multiple vectors by their IDs in a single query."""
        if not vector_ids:
            return []

        session = self.db_manager.get_session()
        try:
            records = (
                session.query(_VectorRecord)
                .filter(_VectorRecord.id.in_(vector_ids))
                .all()
            )

            # Map results
            vectors_map: dict[int, np.ndarray] = {
                int(rec.id): np.array(rec.embedding, dtype=np.float32)
                for rec in records
            }

            # Return in the requested order
            return [vectors_map.get(vid) for vid in vector_ids]
        finally:
            session.close()


# Global vector index instance (initialization with error handling)
# Initialized with error handling so it doesn't block app startup
vector_index: VectorIndex | None
try:
    vector_index = VectorIndex()
    logger.debug("Índice vectorial inicializado correctamente")
except Exception as e:
    logger.warning(f"Advertencia al inicializar vector_index: {e}")
    logger.debug("Esto puede ser normal si la base de datos aún no está lista")
    # Reintentar en el startup event
    vector_index = None


def get_vector_index() -> VectorIndex:
    """Get the vector index instance, initializing it if necessary."""
    global vector_index
    if vector_index is None:
        vector_index = VectorIndex()
    return vector_index

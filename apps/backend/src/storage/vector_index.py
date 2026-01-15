"""Índice vectorial con pgvector para búsqueda de similitud."""

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


class FingerprintVector(Base):
    """Tabla para almacenar vectores de huellas con pgvector."""

    __tablename__ = "fingerprint_vectors"

    id: Column[int] = Column(Integer, primary_key=True, autoincrement=True)
    embedding: Column[list[float]] = Column(Vector(256))  # Vector de 256 dimensiones

    def __repr__(self):
        return f"<FingerprintVector(id={self.id})>"


class VectorIndex:
    """Índice vectorial usando pgvector en PostgreSQL."""

    def __init__(
        self, dimension: int | None = None, db_manager: DatabaseManager | None = None
    ):
        """
        Args:
            dimension: Dimensión de los vectores (default: config.vector_dimension)
            db_manager: Database manager (usa el global si es None)
        """
        from src.storage.database import db_manager as default_manager

        self.dimension = dimension or config.vector_dimension
        self.db_manager: DatabaseManager = db_manager or default_manager  # noqa: UP037

        # Intentar inicializar extensión e índice, pero no fallar si hay problemas
        try:
            self._ensure_extension()
            self._ensure_index()
        except Exception as e:
            # No bloquear la creación de la instancia
            logger.debug(
                f"VectorIndex: Error en inicialización (se reintentará cuando se use): {e}"
            )

    def _ensure_extension(self):
        """Asegura que la extensión pgvector está habilitada."""
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
            # Si hay un error de conexión, lanzar para que se maneje en __init__
            logger.debug(f"Error al conectar con BD para verificar extensión: {e}")
            raise

    def _ensure_index(self):
        """Crea el índice IVFFlat si no existe."""
        session = self.db_manager.get_session()
        try:
            # Primero verificar que la tabla existe
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

            # Verificar si ya existe el índice
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
                # Crear índice IVFFlat para búsquedas rápidas
                # lists = sqrt(total_rows) es una buena heurística
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
        """Añade un vector al índice.

        Args:
            vector: Vector numpy de dimensión (dimension,)

        Returns:
            ID del vector añadido
        """
        # Asegurar dimensión correcta
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector)

        vector = np.asarray(vector, dtype=np.float32)

        session = self.db_manager.get_session()
        try:
            vec_record = FingerprintVector(embedding=vector.tolist())
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
        """Busca los K vectores más similares usando distancia L2.

        Args:
            vector: Vector de consulta
            k: Número de resultados a retornar

        Returns:
            Tupla de (IDs, distancias)
        """
        # Asegurar dimensión correcta
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector)

        vector = np.asarray(vector, dtype=np.float32)
        vector_list = vector.tolist()

        session = self.db_manager.get_session()
        try:
            # Búsqueda K-NN con operador <-> (distancia L2)
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
        """Ajusta el vector a la dimensión correcta."""
        if len(vector) > self.dimension:
            return vector[: self.dimension]
        elif len(vector) < self.dimension:
            padded = np.zeros(self.dimension, dtype=np.float32)
            padded[: len(vector)] = vector
            return padded
        return vector

    def size(self) -> int:
        """Retorna el número de vectores en el índice."""
        session = self.db_manager.get_session()
        try:
            count = session.query(FingerprintVector).count()
            return count
        finally:
            session.close()

    def reset(self):
        """Elimina todos los vectores del índice."""
        session = self.db_manager.get_session()
        try:
            session.query(FingerprintVector).delete()
            session.commit()
            logger.info("Índice vectorial reseteado")
        except Exception as e:
            session.rollback()
            logger.error(f"Error reseteando índice: {e}")
            raise
        finally:
            session.close()

    def get_by_id(self, vector_id: int) -> np.ndarray | None:
        """Recupera un vector por su ID."""
        session = self.db_manager.get_session()
        try:
            record = (
                session.query(FingerprintVector)
                .filter(FingerprintVector.id == vector_id)
                .first()
            )

            if record:
                return np.array(record.embedding, dtype=np.float32)
            return None
        finally:
            session.close()

    def get_batch_by_ids(self, vector_ids: list[int]) -> list[np.ndarray | None]:
        """Recupera múltiples vectores por sus IDs en una sola consulta."""
        if not vector_ids:
            return []

        session = self.db_manager.get_session()
        try:
            records = (
                session.query(FingerprintVector)
                .filter(FingerprintVector.id.in_(vector_ids))
                .all()
            )

            # Mapear resultados
            vectors_map: dict[int, np.ndarray] = {
                int(rec.id): np.array(rec.embedding, dtype=np.float32)
                for rec in records
            }

            # Retornar en el orden solicitado
            return [vectors_map.get(vid) for vid in vector_ids]
        finally:
            session.close()


# Instancia global del índice vectorial (inicialización con manejo de errores)
# Se inicializa con manejo de errores para no bloquear el inicio de la app
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
    """Obtiene la instancia del índice vectorial, inicializándola si es necesario."""
    global vector_index
    if vector_index is None:
        vector_index = VectorIndex()
    return vector_index

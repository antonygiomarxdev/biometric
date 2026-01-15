"""Modelos de base de datos y configuración ORM."""

from datetime import datetime
from typing import Optional

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.dialects.postgresql import JSONB

from src.core.config import config

Base = declarative_base()


class FingerprintRecord(Base):
    """Registro de huella dactilar en la base de datos."""
    
    __tablename__ = "fingerprints"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(100), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    document = Column(String(100), nullable=False, index=True)
    vector_index = Column(Integer, nullable=False, unique=True, index=True)
    num_minutiae = Column(Integer, nullable=False)
    
    # Nuevos campos para almacenamiento y reproducibilidad
    image_path = Column(String(500), nullable=True)  # Ruta en MinIO
    minutiae_data = Column(JSONB, nullable=True)     # Datos crudos de minucias
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Fingerprint(id={self.id}, person_id={self.person_id}, name={self.name})>"


class DatabaseManager:
    """Gestiona la conexión y sesiones de base de datos."""
    
    def __init__(self, database_url: str = None):
        """
        Args:
            database_url: URL de conexión a PostgreSQL
        """
        self.database_url = database_url or config.database_url
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=config.db_pool_size,
            max_overflow=config.db_max_overflow,
            pool_pre_ping=True,
            echo=config.log_level == "DEBUG",
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Crea todas las tablas en la base de datos."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Elimina todas las tablas de la base de datos."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Obtiene una nueva sesión de base de datos."""
        return self.SessionLocal()
    
    def close(self):
        """Cierra el engine de base de datos."""
        self.engine.dispose()


# Instancia global del gestor de base de datos
db_manager = DatabaseManager()


def get_db_session() -> Session:
    """Dependency para obtener sesión de base de datos."""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()

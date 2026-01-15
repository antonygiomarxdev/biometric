"""Configuración de pytest y fixtures compartidas."""

import os
import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.core.config import Config
from src.storage.database import DatabaseManager, Base
from src.storage.vector_index import VectorIndex
from src.storage.repository import FingerprintRepository


@pytest.fixture(scope="session")
def test_config():
    """Configuración de prueba."""
    return Config(
        database_url="sqlite:///:memory:",
        db_pool_size=1,
        db_max_overflow=0,
        vector_dimension=256,
        vector_index_lists=100,
        image_resize_width=350,
        enhancement_enabled=True,
        batch_size=4,
        num_workers=1,
        match_threshold=0.8,
        top_k_matches=5,
        log_level="DEBUG",
        enable_metrics=True,
    )


@pytest.fixture
def db_manager(test_config):
    """Database manager para pruebas."""
    manager = DatabaseManager(database_url=test_config.database_url)
    manager.create_tables()
    yield manager
    manager.drop_tables()
    manager.close()


@pytest.fixture
def vector_index(db_manager, test_config):
    """Índice vectorial para pruebas."""
    index = VectorIndex(
        dimension=test_config.vector_dimension,
        db_manager=db_manager
    )
    
    yield index
    
    # Cleanup
    index.reset()


@pytest.fixture
def repository(db_manager, vector_index):
    """Repositorio para pruebas."""
    repo = FingerprintRepository()
    repo.db_manager = db_manager
    repo.vector_index = vector_index
    return repo


@pytest.fixture
def sample_image():
    """Genera una imagen sintética de huella para pruebas."""
    # Crear una imagen sintética simple con patrones
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # Agregar algunas líneas para simular crestas
    for i in range(10, 190, 10):
        img[i:i+3, 10:190] = 255
    
    # Agregar algo de ruido
    noise = np.random.randint(0, 50, (200, 200), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return img


@pytest.fixture
def fixtures_dir():
    """Directorio de fixtures."""
    return Path(__file__).parent / "fixtures"

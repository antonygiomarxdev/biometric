import os
from pathlib import Path
from pydantic_settings import BaseSettings

ROOT = Path(__file__).resolve().parents[1]

class TestConfig(BaseSettings):
    """Configuration and constants for tests and benchmark scripts."""
    
    # Qdrant test settings
    qdrant_test_collection: str = "test_orchestrator_e2e"
    qdrant_chunk_type: str = "delaunay"
    
    # Synthetic minutiae settings
    synthetic_grid_size: int = 5
    synthetic_minutia_spacing: float = 20.0
    
    # Test entities
    test_person_1: str = "alice"
    test_person_2: str = "bob"
    test_person_3: str = "carol"
    test_fp_suffix: str = "_fp1"
    test_probe_suffix: str = "_probe"

    # Datasets
    socofing_dir: Path = ROOT / "static" / "SOCOFing"
    socofing_real: Path = ROOT / "static" / "SOCOFing" / "Real"
    socofing_altered_easy: Path = ROOT / "static" / "SOCOFing" / "Altered-Easy"
    socofing_altered_medium: Path = ROOT / "static" / "SOCOFing" / "Altered-Medium"
    socofing_altered_hard: Path = ROOT / "static" / "SOCOFing" / "Altered-Hard"

test_config = TestConfig()

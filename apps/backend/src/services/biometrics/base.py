from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict
from dataclasses import dataclass

@dataclass
class BiometricResult:
    matched: bool
    score: float
    metadata: Dict[str, Any]
    person_id: Optional[str] = None

class BiometricProvider(ABC):
    """
    Interfaz base para todos los proveedores de biometría (Huella, Facial, Iris, etc.)
    """

    @abstractmethod
    def extract_features(self, input_data: Any) -> Any:
        """
        Extrae características (embeddings, minucias) de la entrada.
        """
        pass

    @abstractmethod
    def compare(self, probe: Any, gallery: Any) -> float:
        """
        Compara dos muestras y devuelve un score de similitud (0.0 - 1.0).
        """
        pass
    
    @abstractmethod
    def validate_quality(self, input_data: Any) -> Tuple[bool, str]:
        """
        Verifica si la muestra tiene suficiente calidad.
        """
        pass

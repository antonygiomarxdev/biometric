from typing import Any, Tuple
from ..base import BiometricProvider, BiometricResult
from src.services.fingerprint_service import fingerprint_service
from src.core.types import NormalizedFingerprint

class FingerprintProvider(BiometricProvider):
    def extract_features(self, input_data: Any) -> NormalizedFingerprint:
        # input_data debe ser una imagen (numpy array) o bytes
        if isinstance(input_data, bytes):
            return fingerprint_service.process_image_from_bytes(input_data)
        else:
            # Asumimos numpy array
            return fingerprint_service.process_image(input_data)

    def compare(self, probe: NormalizedFingerprint, gallery: NormalizedFingerprint) -> float:
        # Implementación de comparación 1:1 usando el algoritmo de minucias
        # Por ahora simulamos o usamos una función de distancia si existe
        # En la arquitectura actual, el matching real ocurre en la BD (vectorial)
        # o en un matcher específico.
        # Para cumplir la interfaz, podríamos exponer logic del matcher.
        pass

    def validate_quality(self, input_data: Any) -> Tuple[bool, str]:
        # Implementar validación de calidad
        # fingerprint_service ya hace algunas validaciones
        return True, "OK"

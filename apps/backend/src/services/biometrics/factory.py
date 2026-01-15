from typing import Dict, Type
from .base import BiometricProvider
from .providers.fingerprint import FingerprintProvider
from .providers.face import FaceProvider

class BiometricFactory:
    _providers: Dict[str, Type[BiometricProvider]] = {
        "fingerprint": FingerprintProvider,
        "face": FaceProvider
    }

    @classmethod
    def get_provider(cls, modality: str) -> BiometricProvider:
        provider_class = cls._providers.get(modality)
        if not provider_class:
            raise ValueError(f"Modality '{modality}' not supported")
        return provider_class()

    @classmethod
    def register_provider(cls, modality: str, provider: Type[BiometricProvider]):
        cls._providers[modality] = provider

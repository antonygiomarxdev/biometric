from src.fingerprint.domain.repositories.fingerprint_repository import (
    FingerprintRepository,
)


class FingerprintRepositoryImpl(FingerprintRepository):
    def compare(self, fingerprint_1, fingerprint_2):
        # Implementaci�n de la l�gica de comparaci�n usando minutiae o TensorFlow
        pass

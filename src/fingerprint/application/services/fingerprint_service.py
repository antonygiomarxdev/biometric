from src.fingerprint.domain.repositories.fingerprint_repository import (
    FingerprintRepository,
)


class FingerprintService:
    def __init__(self, repository: FingerprintRepository):
        self.repository = repository

    def compare_fingerprints(self, fingerprint_1, fingerprint_2):
        # L�gica para comparar huellas dactilares
        return self.repository.compare(fingerprint_1, fingerprint_2)

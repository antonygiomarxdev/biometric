from typing import List, Optional

from src.fingerprint.domain.entities.minutiae import Minutiae
from src.fingerprint.domain.value_objects.minutiae_vectorizer import MinutiaeVectorizer
from src.fingerprint.infrastructure.vector_index.faiss_index import FingerprintVectorIndex
from src.fingerprint.infrastructure.persistence.postgres.fingerprint_repository_postgres import (
    FingerprintRepositoryPostgres,
)


class IdentifyFingerprintUseCase:
    """Identify a fingerprint from minutiae using a FAISS index."""

    def __init__(
        self,
        vector_index: FingerprintVectorIndex,
        repository: FingerprintRepositoryPostgres,
    ) -> None:
        self.vector_index = vector_index
        self.repository = repository

    def execute(
        self,
        minutiae: List[Minutiae],
        k: int = 1,
        threshold: float = 0.8,
    ) -> Optional[dict]:
        vector = MinutiaeVectorizer.to_vector(minutiae)
        ids, distances = self.vector_index.search(vector, k)
        if not ids:
            return None
        if distances[0] > threshold:
            return None
        person = self.repository.get_person_by_vector_id(ids[0])
        if not person:
            return None
        person["score"] = float(distances[0])
        return person

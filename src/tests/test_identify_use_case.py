from src.fingerprint.application.use_cases.identify_fingerprint import (
    IdentifyFingerprintUseCase,
)
from src.fingerprint.domain.entities.minutiae import Minutiae


class DummyIndex:
    def search(self, vector, k=1):
        return [0], [0.5]


class DummyRepo:
    def get_person_by_vector_id(self, vector_id):
        return {"person_id": "p1", "name": "Alice", "document": "123"}


def test_identify_returns_person_dict():
    minutiae = [Minutiae("termination", (0, 0), 0.0)]
    use_case = IdentifyFingerprintUseCase(DummyIndex(), DummyRepo())
    result = use_case.execute(minutiae, threshold=1.0)
    assert result == {
        "person_id": "p1",
        "name": "Alice",
        "document": "123",
        "score": 0.5,
    }

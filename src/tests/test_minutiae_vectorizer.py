from src.fingerprint.domain.entities.minutiae import Minutiae
from src.fingerprint.domain.value_objects.minutiae_vectorizer import MinutiaeVectorizer


def test_to_vector_returns_numpy_array():
    minutiae = [
        Minutiae("termination", (10, 20), 0.1),
        Minutiae("bifurcation", (30, 40), 0.2),
    ]
    vec = MinutiaeVectorizer.to_vector(minutiae)
    assert vec.shape == (8,)
    assert vec.dtype.name == "float32"

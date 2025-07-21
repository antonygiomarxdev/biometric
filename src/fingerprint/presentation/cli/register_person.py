import argparse
import json
import cv2
import numpy as np

from src.fingerprint.application.use_cases.extract_minutiae import (
    ExtractMinutiaeUseCase,
)
from src.fingerprint.domain.entities.minutiae import Minutiae
from src.fingerprint.domain.value_objects.minutiae_vectorizer import MinutiaeVectorizer
from src.fingerprint.infrastructure.opencv import (
    FingerprintImageEnhancerImpl,
    FingerprintMinutiaeExtractorImpl,
)
from src.fingerprint.infrastructure.persistence.postgres.fingerprint_repository_postgres import (
    FingerprintRepositoryPostgres,
)
from src.fingerprint.infrastructure.vector_index.faiss_index import (
    FingerprintVectorIndex,
)


def load_minutiae_from_image(path: str) -> list[Minutiae]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {path} could not be read")
    enhancer = FingerprintImageEnhancerImpl()
    extractor = FingerprintMinutiaeExtractorImpl()
    use_case = ExtractMinutiaeUseCase(enhancer, extractor)
    return use_case.execute(img)


def load_minutiae_from_json(path: str) -> list[Minutiae]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Minutiae(**m) for m in data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a fingerprint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to fingerprint image")
    group.add_argument("--minutiae", help="Path to minutiae JSON file")
    parser.add_argument("--person-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--document", required=True)
    parser.add_argument(
        "--dsn",
        default="dbname=fingerprint user=postgres password=postgres host=localhost",
        help="PostgreSQL DSN",
    )
    parser.add_argument(
        "--index-path",
        default="/data/index.faiss",
        help="Path to FAISS index file",
    )
    args = parser.parse_args()

    if args.image:
        minutiae = load_minutiae_from_image(args.image)
    else:
        minutiae = load_minutiae_from_json(args.minutiae)

    vector = MinutiaeVectorizer.to_vector(minutiae)
    index = FingerprintVectorIndex(args.index_path, dim=vector.shape[0])
    repo = FingerprintRepositoryPostgres(args.dsn)

    vector_id = index.add(vector)
    repo.register_person(vector_id, args.person_id, args.name, args.document)
    print(f"Registered vector id {vector_id}")


if __name__ == "__main__":
    main()

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form

from src.fingerprint.application.use_cases.identify_fingerprint import (
    IdentifyFingerprintUseCase,
)
from src.fingerprint.domain.entities.minutiae import Minutiae
from src.fingerprint.domain.value_objects.minutiae_vectorizer import MinutiaeVectorizer
from src.fingerprint.infrastructure.persistence.postgres.fingerprint_repository_postgres import (
    FingerprintRepositoryPostgres,
)
from src.fingerprint.infrastructure.vector_index.faiss_index import FingerprintVectorIndex

from src.fingerprint.application.use_cases.extract_minutiae import (
    ExtractMinutiaeUseCase,
)
from src.fingerprint.infrastructure.opencv import (
    FingerprintImageEnhancerImpl,
    FingerprintMinutiaeExtractorImpl,
)

app = FastAPI()

# Global services (simple demo configuration)
vector_index = FingerprintVectorIndex("/data/index.faiss", dim=256)
repository = FingerprintRepositoryPostgres(
    "dbname=fingerprint user=postgres password=postgres host=postgres"
)


@app.post("/extract_minutiae/")
async def extract_minutiae(file: UploadFile = File(...)):
    # Leer la imagen desde el archivo cargado
    image_data = await file.read()
    image = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "No se pudo leer la imagen."}

    # Crear servicios
    enhancer_impl = FingerprintImageEnhancerImpl()
    extractor_impl = FingerprintMinutiaeExtractorImpl()
    extract_minutiae_use_case = ExtractMinutiaeUseCase(
        enhancer_impl, extractor_impl
    )

    # Ejecutar el caso de uso
    minutiae = extract_minutiae_use_case.execute(img)

    return {
        "terminations": sum(1 for m in minutiae if m.type == "termination"),
        "bifurcations": sum(1 for m in minutiae if m.type == "bifurcation"),
    }


@app.post("/identify/")
async def identify(
    file: UploadFile = File(None),
    minutiae: list[dict] | None = None,
):
    """Identify a fingerprint from an image or minutiae."""
    if file is not None:
        image_data = await file.read()
        image = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "No se pudo leer la imagen."}
        enhancer = FingerprintImageEnhancerImpl()
        extractor = FingerprintMinutiaeExtractorImpl()
        use_case_ext = ExtractMinutiaeUseCase(enhancer, extractor)
        minutiae_objs = use_case_ext.execute(img)
    elif minutiae is not None:
        minutiae_objs = [Minutiae(**m) for m in minutiae]
    else:
        return {"error": "No data provided."}

    use_case = IdentifyFingerprintUseCase(vector_index, repository)
    result = use_case.execute(minutiae_objs)
    return {"match": result}


@app.post("/register/")
async def register(
    person_id: str = Form(...),
    name: str = Form(...),
    document: str = Form(...),
    file: UploadFile = File(None),
    minutiae: list[dict] | None = None,
):
    """Register a new fingerprint."""
    if file is not None:
        image_data = await file.read()
        image = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"error": "No se pudo leer la imagen."}
        enhancer = FingerprintImageEnhancerImpl()
        extractor = FingerprintMinutiaeExtractorImpl()
        use_case_ext = ExtractMinutiaeUseCase(enhancer, extractor)
        minutiae_objs = use_case_ext.execute(img)
    elif minutiae is not None:
        minutiae_objs = [Minutiae(**m) for m in minutiae]
    else:
        return {"error": "No data provided."}

    vector = MinutiaeVectorizer.to_vector(minutiae_objs)
    vector_id = vector_index.add(vector)
    repository.register_person(vector_id, person_id, name, document)
    return {"vector_id": vector_id}

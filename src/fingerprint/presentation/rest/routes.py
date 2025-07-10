import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File

from src.fingerprint.application.use_cases.extract_minutiae import (
    ExtractMinutiaeUseCase,
)
from src.fingerprint.infrastructure.opencv import (
    FingerprintImageEnhancerImpl,
    FingerprintMinutiaeExtractorImpl,
)

app = FastAPI()


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

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File

from src.fingerprint.application.use_cases.extract_minutiae import (
    ExtractMinutiaeUseCase,
)
from src.fingerprint.infrastructure.opencv.fingerprint_minutiae_extractor_impl import (
    MinutiaeExtractorImpl,
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

    # Crear el servicio de extracción de minutiae
    extractor_impl = MinutiaeExtractorImpl()
    extract_minutiae_use_case = ExtractMinutiaeUseCase(extractor_impl)

    # Ejecutar el caso de uso
    features_term, features_bif = extract_minutiae_use_case.execute(img)

    return {
        "terminations": len(features_term),
        "bifurcations": len(features_bif),
    }

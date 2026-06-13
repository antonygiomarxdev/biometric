"""API REST para el sistema de huellas dactilares."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core.config import config
from src.core.metrics import metrics
from src.core.types import MinutiaType
from src.services.comparison_service import comparison_service
from src.services.fingerprint_service import fingerprint_service
from src.storage.database import db_manager
from src.storage.object_storage import storage
from src.storage.repository import repository

# Configurar logging estructurado
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Loggers específicos para diferentes módulos
api_logger = logging.getLogger("api.rest")
processing_logger = logging.getLogger("processing")
storage_logger = logging.getLogger("storage")

# Thread pool para ejecutar operaciones CPU-intensivas sin bloquear el event loop
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fingerprint_processor")

# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema Biométrico de Huellas Dactilares",
    description="API para procesamiento, registro e identificación de huellas",
    version="1.0.0",
)

# Configurar CORS
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Incluir routers por recurso (per D-02)
from src.api.routers import (
    auth_router,
    auditoria_router,
    cases_router,
    decisiones_router,
    dictamenes_router,
    evidencias_router,
)
app.include_router(auth_router)
app.include_router(auditoria_router)
app.include_router(cases_router)
app.include_router(decisiones_router)
app.include_router(dictamenes_router)
app.include_router(evidencias_router)


# Modelos Pydantic para requests/responses
class HealthResponse(BaseModel):
    status: str
    database_records: int
    vector_index_size: int
    config: dict[str, Any]


class MinutiaPoint(BaseModel):
    x: int
    y: int
    type: int  # 0: termination, 1: bifurcation
    angle: float


class ExtractResponse(BaseModel):
    minutiae_count: int
    terminations: int
    bifurcations: int
    minutiae: list[MinutiaPoint]
    processing_time_ms: Optional[float] = None


class RegisterRequest(BaseModel):
    person_id: str
    name: str
    document: str


class RegisterResponse(BaseModel):
    success: bool
    record_id: int
    person_id: str
    minutiae_count: int
    message: str


class IdentifyResponse(BaseModel):
    matched: bool
    person_id: Optional[str] = None
    name: Optional[str] = None
    document: Optional[str] = None
    score: float
    distance: Optional[float] = None  # None cuando no hay match, para evitar inf
    processing_time_ms: Optional[float] = None


class MetricsResponse(BaseModel):
    operations: dict[str, dict[str, float]]


class DiagnosticResponse(BaseModel):
    image_decoded: bool
    image_shape: Optional[list[int]]
    image_dtype: Optional[str]
    image_stats: Optional[dict[str, float]]
    enhancement_completed: bool
    enhanced_stats: Optional[dict[str, float]]
    extraction_completed: bool
    candidates_before_filter: int
    candidates_after_filter: int
    skeleton_pixels: Optional[int]
    skeleton_ratio: Optional[float]
    binary_white_ratio: Optional[float]
    error: Optional[str] = None


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Inicialización de la aplicación."""
    logger.info("Iniciando aplicación...")
    logger.info(f"Configuración: ENV={config.env}, LOG_LEVEL={config.log_level}")
    logger.info(
        f"Database URL: {config.database_url.split('@')[1] if '@' in config.database_url else config.database_url}"
    )

    # Crear tablas si no existen (con manejo de errores)
    try:
        db_manager.create_tables()
        logger.info("Base de datos inicializada correctamente")

        # Ejecutar migraciones adicionales si es necesario
        # Esto asegura que las columnas nuevas estén presentes
        try:
            from sqlalchemy import text

            session = db_manager.get_session()
            try:
                # Verificar si image_path existe, si no, agregarlo
                result = session.execute(
                    text(
                        """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'fingerprints' AND column_name = 'image_path'
                """
                    )
                )
                if result.fetchone() is None:
                    logger.info(
                        "Agregando columna image_path a la tabla fingerprints..."
                    )
                    session.execute(
                        text(
                            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS image_path VARCHAR(500)"
                        )
                    )
                    session.execute(
                        text(
                            "ALTER TABLE fingerprints ADD COLUMN IF NOT EXISTS minutiae_data JSONB"
                        )
                    )
                    session.commit()
                    logger.info(
                        "Columnas image_path y minutiae_data agregadas exitosamente"
                    )
                else:
                    logger.debug("Columnas image_path y minutiae_data ya existen")
            finally:
                session.close()
        except Exception as mig_error:
            logger.warning(f"Error ejecutando migraciones: {mig_error}")
            # No bloqueamos el inicio si falla la migración
    except Exception as e:
        logger.warning(f"No se pudo inicializar la base de datos: {e}")
        logger.warning(
            "La aplicación puede iniciarse, pero algunas funciones pueden fallar hasta que la BD esté disponible"
        )

    # Intentar inicializar vector_index si no se pudo al importar
    try:
        from src.storage.vector_index import get_vector_index, vector_index

        if vector_index is None:
            get_vector_index()
            logger.info("Índice vectorial inicializado en startup")
        else:
            logger.info("Índice vectorial ya estaba inicializado")
    except Exception as e:
        logger.error(f"Error crítico: No se pudo inicializar el índice vectorial: {e}")
        logger.error("Las funciones de búsqueda vectorial no estarán disponibles")


@app.on_event("shutdown")
async def shutdown_event():
    """Limpieza al cerrar la aplicación."""
    try:
        logger.info("Cerrando aplicación...")
        executor.shutdown(wait=True)
        db_manager.close()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Errores normales durante el shutdown, no los logueamos
        pass
    except Exception as e:
        logger.warning(f"Error durante shutdown (no crítico): {e}")


# Endpoints
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Verifica el estado del sistema."""
    from src.storage.repository import repository
    from src.storage.vector_index import get_vector_index, vector_index

    # Intentar inicializar vector_index si no está disponible
    try:
        if vector_index is None:
            idx = get_vector_index()
        else:
            idx = vector_index

        db_records = repository.count()
        vector_size = idx.size() if idx else 0
    except Exception as e:
        logger.error(f"Error en health check: {e}")
        db_records = 0
        vector_size = 0

    return HealthResponse(
        status="healthy",
        database_records=db_records,
        vector_index_size=vector_size,
        config={
            "vector_backend": "pgvector",
            "vector_dimension": config.vector_dimension,
            "match_threshold": config.match_threshold,
            "top_k_matches": config.top_k_matches,
        },
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract_minutiae(file: UploadFile = File(...)):
    """Extrae minutiae de una imagen de huella.

    Args:
        file: Archivo de imagen (BMP, PNG, JPEG)

    Returns:
        Conteo de minutiae extraídas
    """
    try:
        api_logger.info(
            f"Iniciando extracción de minutiae - archivo: {file.filename}, tipo: {file.content_type}"
        )

        # Leer imagen
        image_bytes = await file.read()
        api_logger.debug(
            f"Imagen leída del request - tamaño: {len(image_bytes)} bytes, archivo: {file.filename}"
        )

        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")

        # Procesar en thread pool para no bloquear el event loop
        # IMPORTANTE: Decodificar la imagen primero, luego procesar como en el script que funciona
        api_logger.info("Iniciando procesamiento de imagen en thread pool...")
        processing_logger.info(
            f"Iniciando pipeline de procesamiento - tamaño imagen: {len(image_bytes)} bytes"
        )

        # Decodificar imagen primero (igual que cv2.imread en el script)
        import cv2
        import numpy as np

        api_logger.debug(
            f"Decodificando imagen desde bytes - tamaño: {len(image_bytes)} bytes"
        )
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            api_logger.error(
                "cv2.imdecode retornó None - el archivo podría no ser una imagen válida"
            )
            raise HTTPException(
                status_code=400,
                detail="No se pudo decodificar la imagen. Verifica que el archivo sea una imagen válida (BMP, PNG, JPEG) y no esté corrupta.",
            )

        api_logger.debug(
            f"Imagen decodificada exitosamente - shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}"
        )

        # Validar que la imagen tenga un tamaño razonable
        if image.size == 0:
            raise HTTPException(
                status_code=400, detail="La imagen decodificada está vacía"
            )

        if image.shape[0] < 50 or image.shape[1] < 50:
            api_logger.warning(
                f"Imagen muy pequeña: {image.shape} - podría no tener suficiente detalle"
            )

        # Procesar usando el mismo método que el script que funciona
        loop = asyncio.get_event_loop()
        fingerprint = await loop.run_in_executor(
            executor,
            fingerprint_service.process_image,
            image,
            "unknown",  # fingerprint_id
            True,  # resize=True, igual que en el script
        )

        terminations = sum(
            1 for m in fingerprint.minutiae if m.type == MinutiaType.TERMINATION
        )
        bifurcations = sum(
            1 for m in fingerprint.minutiae if m.type == MinutiaType.BIFURCATION
        )
        processing_logger.info(
            f"Procesamiento completado exitosamente - "
            f"minutiae: {len(fingerprint.minutiae)}, "
            f"terminaciones: {terminations}, "
            f"bifurcaciones: {bifurcations}"
        )
        api_logger.info(
            f"Procesamiento completado. Minutiae encontradas: {len(fingerprint.minutiae)}"
        )

        # Si no se encontraron minutiae, dar información más detallada
        if len(fingerprint.minutiae) == 0:
            api_logger.warning(
                "No se encontraron minutiae. Posibles causas: "
                "imagen de baja calidad, imagen no es una huella válida, "
                "o problemas en el procesamiento. Revisar logs de processing para más detalles."
            )

        # Obtener métricas de procesamiento
        # El servicio usa "process_pipeline" como nombre de métrica
        stats = metrics.get_stats("process_pipeline")
        processing_time = stats.get("mean") if stats else None

        if processing_time:
            api_logger.debug(f"Tiempo de procesamiento: {processing_time:.2f}ms")

        # Convertir minucias a formato respuesta
        minutiae_list = [
            MinutiaPoint(
                x=m.x,
                y=m.y,
                type=0 if m.type == MinutiaType.TERMINATION else 1,
                angle=m.angle,
            )
            for m in fingerprint.minutiae
        ]

        api_logger.debug(f"Preparando respuesta con {len(minutiae_list)} minutiae")
        return ExtractResponse(
            minutiae_count=len(fingerprint.minutiae),
            terminations=terminations,
            bifurcations=bifurcations,
            minutiae=minutiae_list,
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except ValueError as e:
        api_logger.error(
            f"Error de validación extrayendo minutiae - mensaje: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=400, detail=f"Error procesando la imagen: {str(e)}"
        ) from e
    except Exception as e:
        api_logger.error(
            f"Error extrayendo minutiae - tipo: {type(e).__name__}, mensaje: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error procesando la imagen: {str(e)}"
        ) from e


@app.post("/register", response_model=RegisterResponse)
async def register_fingerprint(
    person_id: str = Form(...),
    name: str = Form(...),
    document: str = Form(...),
    file: UploadFile = File(...),
):
    """Registra una nueva huella en el sistema.

    Args:
        person_id: ID único de la persona
        name: Nombre completo
        document: Número de documento
        file: Archivo de imagen de la huella

    Returns:
        Confirmación de registro
    """
    try:
        # Procesar imagen
        api_logger.info(
            f"Iniciando registro de huella - person_id: {person_id}, name: {name}, document: {document}"
        )
        image_bytes = await file.read()
        api_logger.debug(
            f"Imagen recibida para registro - tamaño: {len(image_bytes)} bytes, person_id: {person_id}"
        )

        # Decodificar imagen primero (igual que cv2.imread en el script que funciona)
        import cv2
        import numpy as np

        api_logger.debug(
            f"Decodificando imagen desde bytes - tamaño: {len(image_bytes)} bytes"
        )
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            api_logger.error(
                "cv2.imdecode retornó None - el archivo podría no ser una imagen válida"
            )
            raise HTTPException(
                status_code=400,
                detail="No se pudo decodificar la imagen. Verifica que el archivo sea una imagen válida (BMP, PNG, JPEG) y no esté corrupta.",
            )

        api_logger.debug(
            f"Imagen decodificada exitosamente - shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}"
        )

        # Validar que la imagen tenga un tamaño razonable
        if image.size == 0:
            raise HTTPException(
                status_code=400, detail="La imagen decodificada está vacía"
            )

        if image.shape[0] < 50 or image.shape[1] < 50:
            api_logger.warning(
                f"Imagen muy pequeña: {image.shape} - podría no tener suficiente detalle"
            )

        # Ejecutar en thread pool para no bloquear - usar el mismo método que el script
        processing_logger.info(
            f"Procesando imagen para registro - person_id: {person_id}"
        )
        loop = asyncio.get_event_loop()
        fingerprint = await loop.run_in_executor(
            executor,
            fingerprint_service.process_image,
            image,
            f"person_{person_id}",  # fingerprint_id
            True,  # resize=True, igual que en el script que funciona
        )

        if not fingerprint.minutiae:
            api_logger.warning(
                f"No se pudieron extraer minutiae durante el registro - person_id: {person_id}"
            )
            raise HTTPException(
                status_code=400,
                detail="No se pudieron extraer minutiae de la imagen. Por favor, verifica que la imagen sea válida y esté en buen estado.",
            )

        # Registrar en el sistema
        storage_logger.info(
            f"Guardando registro en base de datos - person_id: {person_id}"
        )
        record_id = comparison_service.register_fingerprint(
            fingerprint=fingerprint,
            person_id=person_id,
            name=name,
            document=document,
            image_bytes=image_bytes,  # Pasamos la imagen original
        )

        api_logger.info(
            f"Registro completado exitosamente - "
            f"record_id: {record_id}, person_id: {person_id}, minutiae_count: {len(fingerprint.minutiae)}"
        )
        return RegisterResponse(
            success=True,
            record_id=record_id,
            person_id=person_id,
            minutiae_count=len(fingerprint.minutiae),
            message=f"Huella registrada exitosamente para {name}",
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(
            f"Error registrando huella - "
            f"person_id: {person_id}, tipo: {type(e).__name__}, mensaje: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error registrando la huella: {str(e)}"
        ) from e


@app.post("/identify", response_model=IdentifyResponse)
async def identify_fingerprint(file: UploadFile = File(...)):
    """Identifica una huella buscando coincidencias en el sistema.

    Args:
        file: Archivo de imagen de la huella

    Returns:
        Resultado de la identificación
    """
    try:
        api_logger.info("Iniciando identificación de huella")
        # Procesar imagen - usar el mismo flujo que el script que funciona
        image_bytes = await file.read()
        api_logger.debug(
            f"Imagen recibida para identificación - tamaño: {len(image_bytes)} bytes"
        )

        # Decodificar imagen primero (igual que cv2.imread en el script)
        import cv2
        import numpy as np

        api_logger.debug(
            f"Decodificando imagen desde bytes - tamaño: {len(image_bytes)} bytes"
        )
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            api_logger.error(
                "cv2.imdecode retornó None - el archivo podría no ser una imagen válida"
            )
            raise HTTPException(
                status_code=400,
                detail="No se pudo decodificar la imagen. Verifica que el archivo sea una imagen válida (BMP, PNG, JPEG) y no esté corrupta.",
            )

        api_logger.debug(
            f"Imagen decodificada exitosamente - shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}"
        )

        # Validar que la imagen tenga un tamaño razonable
        if image.size == 0:
            raise HTTPException(
                status_code=400, detail="La imagen decodificada está vacía"
            )

        if image.shape[0] < 50 or image.shape[1] < 50:
            api_logger.warning(
                f"Imagen muy pequeña: {image.shape} - podría no tener suficiente detalle"
            )

        # Ejecutar en thread pool para no bloquear
        processing_logger.info("Procesando imagen para identificación")
        loop = asyncio.get_event_loop()
        fingerprint = await loop.run_in_executor(
            executor,
            fingerprint_service.process_image,
            image,
            "identify",  # fingerprint_id
            True,  # resize=True, igual que en el script
        )

        if not fingerprint.minutiae:
            api_logger.warning(
                "No se pudieron extraer minutiae durante la identificación"
            )
            raise HTTPException(
                status_code=400,
                detail="No se pudieron extraer minutiae de la imagen. Por favor, verifica que la imagen sea válida.",
            )

        # Identificar
        storage_logger.info("Buscando coincidencias en base de datos")
        result = comparison_service.identify(fingerprint)

        api_logger.info(
            f"Identificación completada - "
            f"matched: {result.matched}, person_id: {result.person_id}, "
            f"score: {result.score}, distance: {result.l2_distance}"
        )

        # Obtener métricas
        stats = metrics.get_stats("repository_identify")
        processing_time = stats.get("mean") if stats else None

        # Convertir inf o valores muy grandes a None para compatibilidad JSON
        distance = result.l2_distance
        if distance is not None and (distance == float("inf") or distance >= 1e10):
            distance = None

        return IdentifyResponse(
            matched=result.matched,
            person_id=result.person_id,
            name=result.metadata.get("name") if result.matched else None,
            document=result.metadata.get("document") if result.matched else None,
            score=result.score,
            distance=distance,  # None cuando no hay match
            processing_time_ms=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(
            f"Error identificando huella - tipo: {type(e).__name__}, mensaje: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Error identificando la huella: {str(e)}"
        ) from e


@app.get("/fingerprints/{person_id}/image")
async def get_fingerprint_image(person_id: str):
    """Recupera la imagen original de una huella.

    Args:
        person_id: ID de la persona

    Returns:
        Imagen (image/bmp)
    """
    try:
        record = repository.get_by_person_id(person_id)
        if not record or not record.image_path:
            raise HTTPException(status_code=404, detail="Imagen no encontrada")

        image_bytes = storage.get_file(record.image_path)
        if not image_bytes:
            raise HTTPException(
                status_code=404, detail="Archivo no encontrado en storage"
            )

        return Response(content=image_bytes, media_type="image/bmp")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recuperando imagen: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/fingerprints/{person_id}/details")
async def get_fingerprint_details(person_id: str):
    """Recupera detalles y minucias guardadas.

    Args:
        person_id: ID de la persona

    Returns:
        JSON con metadatos y minucias
    """
    try:
        record = repository.get_by_person_id(person_id)
        if not record:
            raise HTTPException(status_code=404, detail="Registro no encontrado")

        return {
            "person_id": record.person_id,
            "name": record.name,
            "document": record.document,
            "minutiae_count": record.num_minutiae,
            "minutiae": record.minutiae_data,
            "has_image": bool(record.image_path),
        }
    except Exception as e:
        logger.error(f"Error recuperando detalles: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Obtiene métricas de performance del sistema."""
    operations = {}

    for operation in metrics.metrics.keys():
        stats = metrics.get_stats(operation)
        if stats:
            operations[operation] = stats

    return MetricsResponse(operations=operations)


@app.post("/extract/diagnostic", response_model=DiagnosticResponse)
async def extract_diagnostic(file: UploadFile = File(...)):
    """Endpoint de diagnóstico para entender por qué no se extraen minutiae."""
    import cv2
    import numpy as np

    diagnostic = {
        "image_decoded": False,
        "image_shape": None,
        "image_dtype": None,
        "image_stats": None,
        "enhancement_completed": False,
        "enhanced_stats": None,
        "extraction_completed": False,
        "candidates_before_filter": 0,
        "candidates_after_filter": 0,
        "skeleton_pixels": None,
        "skeleton_ratio": None,
        "binary_white_ratio": None,
        "error": None,
    }

    try:
        # 1. Decodificar imagen
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            diagnostic["error"] = "No se pudo decodificar la imagen"
            return DiagnosticResponse(**diagnostic)

        diagnostic["image_decoded"] = True
        diagnostic["image_shape"] = list(image.shape)
        diagnostic["image_dtype"] = str(image.dtype)
        diagnostic["image_stats"] = {
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean()),
            "std": float(image.std()),
        }

        # 2. Enhancement
        from src.processing.enhancer import create_enhancer

        enhancer = create_enhancer()
        enhanced = enhancer.enhance(image, resize=True)

        diagnostic["enhancement_completed"] = True
        diagnostic["enhanced_stats"] = {
            "min": float(enhanced.min()),
            "max": float(enhanced.max()),
            "mean": float(enhanced.mean()),
            "std": float(enhanced.std()),
        }

        # 3. Binarización
        binary_white = np.sum(enhanced > 127)
        total_pixels = enhanced.size
        diagnostic["binary_white_ratio"] = float(binary_white / total_pixels)

        # 4. Skeletonización
        from skimage.morphology import skeletonize

        binary = enhanced > 127
        skel = skeletonize(binary).astype(np.uint8)
        skel_pixels = np.sum(skel > 0)
        diagnostic["skeleton_pixels"] = int(skel_pixels)
        diagnostic["skeleton_ratio"] = float(skel_pixels / total_pixels)

        # 5. Extracción (usar método interno para diagnóstico)
        from src.processing.extractor import SkeletonMinutiaeExtractor

        extractor = SkeletonMinutiaeExtractor()
        candidates = extractor._detect_crossing_number(skel)
        diagnostic["candidates_before_filter"] = len(candidates)

        # 6. Filtrado
        mask = extractor._create_mask(skel)
        filtered = extractor._filter_candidates(candidates, mask, skel.shape)
        diagnostic["candidates_after_filter"] = len(filtered)
        diagnostic["extraction_completed"] = True

    except Exception as e:
        diagnostic["error"] = str(e)
        api_logger.error(f"Error en diagnóstico: {e}", exc_info=True)

    return DiagnosticResponse(**diagnostic)


@app.post("/metrics/reset")
async def reset_metrics():
    """Resetea las métricas acumuladas."""
    metrics.reset()
    return {"message": "Métricas reseteadas"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

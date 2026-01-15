"""
Servicio principal de procesamiento de huellas.
Clean Code: Orquestador con inyección de dependencias y paralelismo.
"""

import concurrent.futures
import logging
from typing import List, Optional, Union

import cv2
import numpy as np

from src.core.gpu_utils import GPUConfig
from src.core.interfaces import IEnhancer, IFeatureExtractor, INormalizer
from src.core.metrics import measure_time, timed
from src.core.types import MinutiaCandidate, NormalizedFingerprint
from src.processing.enhancer import create_enhancer
from src.processing.extractor import SkeletonMinutiaeExtractor
from src.processing.normalization import MinutiaNormalizer

logger = logging.getLogger(__name__)


class FingerprintService:
    """Orquesta el pipeline completo de procesamiento de huellas."""

    def __init__(
        self,
        enhancer: Optional[IEnhancer] = None,
        extractor: Optional[IFeatureExtractor] = None,
        normalizer: Optional[INormalizer] = None,
    ):
        self.enhancer = enhancer or create_enhancer()
        self.extractor = extractor or SkeletonMinutiaeExtractor()
        self.normalizer = normalizer or MinutiaNormalizer()

        # Configurar executor según hardware
        self.use_parallel = not GPUConfig.is_enabled()
        self._max_workers = None if self.use_parallel else 1

    def process_image(
        self, image: np.ndarray, fingerprint_id: str = "unknown", resize: bool = True
    ) -> NormalizedFingerprint:
        """
        Procesa una imagen de huella completa: enhance + extract + normalize.
        """
        if image is None:
            raise ValueError("La imagen no puede ser None")

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        import logging

        processing_logger = logging.getLogger("processing")

        processing_logger.info(
            f"Procesando imagen - shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, resize: {resize}"
        )

        with measure_time("process_pipeline"):
            # 1. Enhancement (GPU/CPU)
            processing_logger.debug("Iniciando enhancement de imagen...")
            enhanced = self.enhancer.enhance(image, resize=resize)
            processing_logger.debug(
                f"Enhancement completado - shape: {enhanced.shape}, "
                f"dtype: {enhanced.dtype}, min: {enhanced.min()}, max: {enhanced.max()}"
            )

            # 2. Extraction (El extractor maneja la skeletonización si es necesario)
            processing_logger.debug("Iniciando extracción de minutiae...")
            candidates = self.extractor.extract(enhanced)
            processing_logger.info(f"Minutiae extraídas: {len(candidates)}")

            if len(candidates) == 0:
                processing_logger.warning(
                    "No se encontraron minutiae. Posibles causas: "
                    "imagen de baja calidad, imagen no es una huella válida, "
                    "o problemas en el procesamiento."
                )

            # 3. Normalization
            processing_logger.debug("Normalizando minutiae...")
            normalized_fp = self.normalizer.normalize(candidates, image.shape)
            processing_logger.info(
                f"Normalización completada - minutiae finales: {len(normalized_fp.minutiae)}"
            )

            # Asignar ID real (Normalizer no lo sabe)
            # normalized_fp es inmutable (dataclass frozen), necesitamos replace
            from dataclasses import replace

            normalized_fp = replace(normalized_fp, id=fingerprint_id)

        return normalized_fp

    def process_batch(
        self,
        images: List[np.ndarray],
        fingerprint_ids: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[Optional[NormalizedFingerprint]]:
        """
        Procesa múltiples imágenes en batch.
        Usa ProcessPoolExecutor si estamos en CPU para saturar cores.
        """
        if fingerprint_ids is None:
            fingerprint_ids = ["unknown"] * len(images)

        results = [None] * len(images)

        # Si hay GPU, procesamos secuencialmente (o en batch real si el enhancer lo soportara)
        # para evitar contexto CUDA en múltiples procesos.
        if not self.use_parallel:
            for i, (img, fid) in enumerate(zip(images, fingerprint_ids)):
                try:
                    results[i] = self.process_image(img, fingerprint_id=fid)
                except Exception as e:
                    logger.error(f"Error procesando imagen {fid}: {e}")
            return results

        # En CPU usamos multiprocesamiento
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            # Enviamos tareas
            future_to_idx = {
                executor.submit(self._process_wrapper, img, fid): i
                for i, (img, fid) in enumerate(zip(images, fingerprint_ids))
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error en worker {idx}: {e}")
                    results[idx] = None

        return results

    @staticmethod
    def _process_wrapper(img, fid):
        # Método estático para pickling en multiprocessing
        # Necesitamos instanciar un servicio fresco en el worker
        # Ojo: esto crea overhead de inicialización por cada imagen si no se hace con cuidado.
        # Mejor sería usar `initializer` en el Pool para crear el servicio una vez por worker.
        # Por simplicidad aquí instanciamos. CpuEnhancer es ligero.
        service = FingerprintService()
        return service.process_image(img, fingerprint_id=fid)

    def process_image_from_path(self, image_path: str) -> NormalizedFingerprint:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        return self.process_image(image, fingerprint_id=image_path)

    def process_image_from_bytes(
        self, image_bytes: bytes, fingerprint_id: str = "unknown"
    ) -> NormalizedFingerprint:
        """Procesa una imagen desde bytes."""
        logger.debug(f"Decodificando imagen desde bytes - tamaño: {len(image_bytes)} bytes")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            logger.error("cv2.imdecode retornó None - el archivo podría no ser una imagen válida")
            raise ValueError(
                "No se pudo decodificar la imagen. "
                "Verifica que el archivo sea una imagen válida (BMP, PNG, JPEG) y no esté corrupta."
            )
        
        logger.debug(
            f"Imagen decodificada exitosamente - shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}"
        )
        
        # Validar que la imagen tenga un tamaño razonable
        if image.size == 0:
            raise ValueError("La imagen decodificada está vacía")
        
        if image.shape[0] < 50 or image.shape[1] < 50:
            logger.warning(f"Imagen muy pequeña: {image.shape} - podría no tener suficiente detalle")
        
        # Asegurar que resize=True se pase explícitamente (aunque es el default)
        # Esto es importante porque algunas imágenes pequeñas necesitan ser redimensionadas
        return self.process_image(image, fingerprint_id=fingerprint_id, resize=True)


# Instancia global
fingerprint_service = FingerprintService()

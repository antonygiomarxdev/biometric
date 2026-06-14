"""
Main fingerprint processing service.
Clean Code: Orchestrator with dependency injection and parallelism.
"""

import concurrent.futures
import logging
from typing import List, Optional, Union, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.ai.model_manager import ModelManager


from src.core.interfaces import IEnhancer, IFeatureExtractor, INormalizer
from src.core.metrics import measure_time, timed
from src.core.types import MinutiaCandidate, NormalizedFingerprint
from src.processing.enhancer import create_enhancer
from src.processing.extractor import SkeletonMinutiaeExtractor
from src.processing.normalization import MinutiaNormalizer

logger = logging.getLogger(__name__)


class FingerprintService:
    """Orchestrates the complete fingerprint processing pipeline."""

    def __init__(
        self,
        enhancer: Optional[IEnhancer] = None,
        extractor: Optional[IFeatureExtractor] = None,
        normalizer: Optional[INormalizer] = None,
    ):
        self.enhancer = enhancer or create_enhancer()
        self.extractor = extractor or SkeletonMinutiaeExtractor()
        self.normalizer = normalizer or MinutiaNormalizer()

        # Configure executor based on hardware
        self.use_parallel = True  # Always parallel on CPU
        self._max_workers = None
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def process_image(
        self, image: np.ndarray, fingerprint_id: str = "unknown", resize: bool = True
    ) -> NormalizedFingerprint:
        """
        Process a complete fingerprint image: enhance + extract + normalize.
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

            # 2. Extraction (The extractor handles skeletonization if necessary)
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

            # Assign real ID (Normalizer doesn't know it)
            # normalized_fp is immutable (frozen dataclass), we need replace
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
        Process multiple images in batch.
        Uses ProcessPoolExecutor on CPU to saturate cores.
        """
        if fingerprint_ids is None:
            fingerprint_ids = ["unknown"] * len(images)

        results = [None] * len(images)

        # If GPU is available, process sequentially (or real batch if enhancer supports it)
        # to avoid CUDA context in multiple processes.
        if not self.use_parallel:
            for i, (img, fid) in enumerate(zip(images, fingerprint_ids)):
                try:
                    results[i] = self.process_image(img, fingerprint_id=fid)
                except Exception as e:
                    logger.error(f"Error procesando imagen {fid}: {e}")
            return results

        # On CPU we use multiprocessing
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
        # Static method for pickling in multiprocessing
        # We need to instantiate a fresh service in the worker
        # Note: this creates initialization overhead per image if not done carefully.
        # Better would be to use `initializer` in the Pool to create the service once per worker.
        # For simplicity we instantiate here. CpuEnhancer is lightweight.
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
        """Process an image from bytes."""
        logger.debug(f"Decodificando imagen desde bytes - tamaño: {len(image_bytes)} bytes")
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            logger.error("cv2.imdecode retornó None - el archivo podría no ser una imagen válida")
            raise ValueError(
                "No se pudo decodificar la imagen. "
                "Verifica que el archivo sea una imagen válida (BMP, PNG, JPEG) y no esté corrupta."
            )
        
        # Validate that the image has a reasonable size BEFORE accessing
        # .min() / .max() / .mean(), which would raise on a zero-size array.
        if image.size == 0:
            raise ValueError("La imagen decodificada está vacía")

        logger.debug(
            f"Imagen decodificada exitosamente - shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, mean: {image.mean():.2f}"
        )
        
        if image.shape[0] < 50 or image.shape[1] < 50:
            logger.warning(f"Imagen muy pequeña: {image.shape} - podría no tener suficiente detalle")
        
        # Ensure resize=True is passed explicitly (even though it is the default)
        # This is important because some small images need to be resized
        return self.process_image(image, fingerprint_id=fingerprint_id, resize=True)


# Global instance
fingerprint_service = FingerprintService()


def create_ai_fingerprint_service(
    model_manager: "ModelManager",
    use_segmentation: bool = True,
    use_enhancement: bool = True,
    use_dl_extractor: bool = True,
) -> FingerprintService:
    """Create a FingerprintService configured with AI components.
    
    Falls back to CPU components for any disabled AI stage.
    """
    from src.processing.enhancer import create_enhancer
    from src.processing.extractor import AiFeatureExtractor
    from src.ai import AiConfig
    
    config = AiConfig()
    enhancer = None
    if use_segmentation and use_enhancement:
        enhancer = create_enhancer("full_ai", model_manager=model_manager)
    elif use_segmentation:
        enhancer = create_enhancer("segmentation", model_manager=model_manager)
    elif use_enhancement:
        enhancer = create_enhancer("enhancement", model_manager=model_manager)

    extractor = AiFeatureExtractor(model_manager) if use_dl_extractor else None

    return FingerprintService(
        enhancer=enhancer,  # None = CPU enhancer (default)
        extractor=extractor,  # None = skeleton extractor (default)
    )


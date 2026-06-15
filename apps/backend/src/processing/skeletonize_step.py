from __future__ import annotations

import logging

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from src.core.interfaces import IPipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class SkeletonizationStep(IPipelineStep):
    """
    Convierte la imagen mejorada en un esqueleto estricto (1 pixel de grosor).
    
    Aplica binarización de Otsu y un filtro morfológico para eliminar 
    "pelusas" (islas pequeñas desconectadas) antes de esqueletizar. 
    El resultado se guarda en `ctx.skeleton` como un array uint8 (0 y 1).
    """
    def __init__(self, min_island_size: int = 20) -> None:
        self.min_island_size = min_island_size

    def process(self, ctx: PipelineContext) -> None:
        source = ctx.enhanced_image if ctx.enhanced_image is not None else ctx.raw_image
        
        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        # 1. Binarización robusta (Otsu)
        _, binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_bool = binary > 0

        # 2. Limpieza de "pelusas" (elimina manchas blancas desconectadas)
        if self.min_island_size > 0:
            binary_bool = remove_small_objects(binary_bool, min_size=self.min_island_size)

        white_pixels = int(binary_bool.sum())
        if white_pixels < 10:
            logger.warning("SkeletonizationStep: image too dark or empty")
            ctx.skeleton = np.zeros_like(binary_bool, dtype=np.uint8)
            return

        # 3. Esqueletización biológica
        skel_bool = skeletonize(binary_bool)
        
        # Guardar como uint8 estricto (0/1)
        ctx.skeleton = skel_bool.astype(np.uint8)
        
        logger.debug(
            "SkeletonizationStep: skeleton generated with %d pixels", 
            int(ctx.skeleton.sum())
        )

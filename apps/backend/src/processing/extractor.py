"""
Extracción de características (minucias).
Clean Code: Implementación de IFeatureExtractor con tipos estrictos.

Available extractors
--------------------
* :class:`SkeletonMinutiaeExtractor` — traditional Crossing Number on skeleton
* :class:`GradientRidgeExtractor` — Harris-corner-based validation cross-check
* :class:`AiFeatureExtractor` — deep-learning extraction via ONNX Runtime
"""

import logging
import math

import cv2
import numpy as np
from skimage.morphology import convex_hull_image, erosion, square

from src.ai.config import AiConfig
from src.ai.extraction import ExtractionProcessor
from src.ai.model_manager import ModelManager
from src.core.metrics import timed
from src.core.types import AlgorithmOrigin, MinutiaCandidate, MinutiaType

logger = logging.getLogger("processing.extractor")


class SkeletonMinutiaeExtractor:
    """
    Extractor basado en esqueleto (Crossing Number).
    """

    def __init__(self, border_margin: int = 10, erosion_size: int = 5):
        self.border_margin = border_margin
        self.erosion_size = erosion_size

    @timed("extract_minutiae_skeleton")
    def extract(self, image: np.ndarray) -> list[MinutiaCandidate]:
        """
        Extrae minucias.
        Acepta imagen binaria (del Enhancer) y aplica skeletonización internamente.

        IMPORTANTE: La imagen puede venir como uint8 (0-255) del enhancer,
        pero el skeleton DEBE ser binario (0/1) para que el Crossing Number funcione.
        """
        from skimage.morphology import skeletonize

        logger.debug(
            f"Extrayendo minutiae - imagen shape: {image.shape}, dtype: {image.dtype}, "
            f"min: {image.min()}, max: {image.max()}, unique values: {len(np.unique(image))}"
        )

        # 1. Binarizar robustamente - usar umbral adaptativo si la imagen tiene problemas
        # Calcular umbral usando Otsu si es posible
        try:
            import cv2

            if len(np.unique(image)) > 2:
                # Imagen no binaria, usar Otsu
                threshold_value, binary = cv2.threshold(
                    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                binary = binary > 0
                logger.debug(f"Binarización con Otsu - umbral: {threshold_value}")
            else:
                # Ya es binaria o casi binaria
                threshold_value = 127
                binary = image > threshold_value
                logger.debug(f"Binarización directa - umbral: {threshold_value}")
        except Exception as e:
            logger.warning(f"Error en binarización adaptativa, usando umbral fijo: {e}")
            threshold_value = 127
            binary = image > threshold_value

        white_pixels = np.sum(binary)
        total_pixels = binary.size
        logger.debug(
            f"Binarización - píxeles blancos: {white_pixels}/{total_pixels} ({100*white_pixels/total_pixels:.1f}%)"
        )

        if white_pixels == 0:
            logger.warning("Imagen completamente negra después de binarización")
            return []

        if white_pixels == total_pixels:
            logger.warning("Imagen completamente blanca después de binarización")
            return []

        if white_pixels < total_pixels * 0.05:
            logger.warning(
                f"Muy pocos píxeles blancos ({100*white_pixels/total_pixels:.1f}%) - la imagen podría estar invertida"
            )
            # Intentar invertir
            binary = ~binary
            white_pixels = np.sum(binary)
            logger.debug(
                f"Después de invertir - píxeles blancos: {white_pixels}/{total_pixels} ({100*white_pixels/total_pixels:.1f}%)"
            )

        # 2. Esqueletizar (CPU) - Paso crítico para Crossing Number
        logger.debug("Iniciando skeletonización...")
        try:
            skel_bool = skeletonize(binary)
            # CRÍTICO: El skeleton DEBE ser binario (0/1) para que el Crossing Number funcione correctamente
            # skeletonize retorna boolean, pero lo convertimos a uint8 asegurando que sea 0 o 1
            skel = (skel_bool.astype(np.uint8) > 0).astype(np.uint8)  # Forzar a 0/1
            skel_pixels = np.sum(skel > 0)
            logger.debug(
                f"Skeletonización completada - píxeles del esqueleto: {skel_pixels} ({100*skel_pixels/total_pixels:.2f}%), "
                f"dtype: {skel.dtype}, min: {skel.min()}, max: {skel.max()}"
            )

            # Validar que el skeleton sea realmente binario
            unique_vals = np.unique(skel)
            if len(unique_vals) > 2 or (
                len(unique_vals) == 2 and not all(v in [0, 1] for v in unique_vals)
            ):
                logger.warning(
                    f"Skeleton no es binario! Valores únicos: {unique_vals}. "
                    f"Normalizando a 0/1..."
                )
                skel = (skel > 0).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error durante skeletonización: {e}", exc_info=True)
            return []

        if skel_pixels == 0:
            logger.warning(
                f"No se generó esqueleto - la imagen podría no tener estructura de crestas. "
                f"Binary stats: white={white_pixels}/{total_pixels} ({100*white_pixels/total_pixels:.1f}%), "
                f"shape={binary.shape}"
            )
            return []

        # 3. Detectar
        logger.debug("Detectando minutiae usando Crossing Number...")
        logger.debug(
            f"Esqueleto shape: {skel.shape}, dtype: {skel.dtype}, min: {skel.min()}, max: {skel.max()}"
        )
        # Asegurar que el skeleton sea binario antes de pasar al CN
        # CRÍTICO: El skeleton DEBE ser 0/1, no 0-255
        skel_binary = (skel > 0).astype(np.uint8)

        # Validar una vez más antes de pasar al CN
        skel_unique = np.unique(skel_binary)
        if len(skel_unique) > 2 or not all(v in [0, 1] for v in skel_unique):
            logger.error(
                f"skel_binary no es binario antes de _detect_crossing_number! "
                f"Valores únicos: {skel_unique}, min: {skel_binary.min()}, max: {skel_binary.max()}"
            )
            skel_binary = (skel_binary > 0).astype(np.uint8)

        logger.debug(
            f"Antes de _detect_crossing_number - skel_binary unique: {np.unique(skel_binary)}, "
            f"min: {skel_binary.min()}, max: {skel_binary.max()}, dtype: {skel_binary.dtype}"
        )

        candidates = self._detect_crossing_number(skel_binary)
        logger.info(f"Minutiae detectadas antes de filtrado: {len(candidates)}")

        if len(candidates) == 0:
            logger.warning(
                f"Crossing Number no detectó candidatas. "
                f"Esqueleto: {skel_pixels} píxeles ({100*skel_pixels/total_pixels:.2f}%), "
                f"shape: {skel.shape}"
            )

        if len(candidates) == 0:
            logger.warning("No se detectaron minutiae con Crossing Number")
            return []

        # 4. Filtrado geométrico
        # Usamos el esqueleto para la máscara convexa
        logger.debug("Aplicando filtrado geométrico...")
        mask = self._create_mask(skel)
        candidates_before = len(candidates)
        candidates = self._filter_candidates(candidates, mask, skel.shape)
        logger.info(
            f"Minutiae después de filtrado: {len(candidates)} (eliminadas: {candidates_before - len(candidates)})"
        )

        if len(candidates) == 0:
            logger.warning(
                f"Todas las candidatas fueron filtradas. "
                f"border_margin: {self.border_margin}, erosion_size: {self.erosion_size}"
            )

        return candidates

    def _detect_crossing_number(self, skel: np.ndarray) -> list[MinutiaCandidate]:
        """
        Detecta minutias usando Crossing Number (CN) vectorizado.
        CN = 0.5 * sum(|Pi - P_{i+1}|)

        IMPORTANTE: skel DEBE ser binario (0/1) para que el CN funcione correctamente.
        """
        rows, cols = skel.shape
        candidates = []

        # Validar que el skeleton sea binario (0/1)
        unique_vals = np.unique(skel)
        logger.debug(
            f"_detect_crossing_number - skeleton unique values: {unique_vals}, "
            f"min: {skel.min()}, max: {skel.max()}, dtype: {skel.dtype}"
        )

        if len(unique_vals) > 2 or (
            len(unique_vals) == 2 and not all(v in [0, 1] for v in unique_vals)
        ):
            logger.error(
                f"Skeleton no es binario en _detect_crossing_number! Valores únicos: {unique_vals}. "
                f"Normalizando a 0/1..."
            )
            skel = (skel > 0).astype(np.uint8)
            logger.debug(
                f"Después de normalizar - unique: {np.unique(skel)}, min: {skel.min()}, max: {skel.max()}"
            )

        # Asegurar que sea realmente binario (0/1) antes de calcular CN
        skel = (skel > 0).astype(np.uint8)

        # Padding para manejar bordes sin ifs
        padded = np.pad(skel, 1, mode="constant")

        # Validar que el padded también sea binario
        padded_unique = np.unique(padded)
        logger.debug(
            f"Padded skeleton unique values: {padded_unique}, min: {padded.min()}, max: {padded.max()}"
        )

        if len(padded_unique) > 2 or not all(v in [0, 1] for v in padded_unique):
            logger.error(f"Padded skeleton no es binario! Valores: {padded_unique}")
            padded = (padded > 0).astype(np.uint8)

        # Obtener vecinos desplazados (vectorizado)
        # P2 P3 P4
        # P9 P1 P5
        # P8 P7 P6
        # P1 es padded[1:-1, 1:-1]

        p2 = padded[0:-2, 0:-2]
        p3 = padded[0:-2, 1:-1]
        p4 = padded[0:-2, 2:]
        p5 = padded[1:-1, 2:]
        p6 = padded[2:, 2:]
        p7 = padded[2:, 1:-1]
        p8 = padded[2:, 0:-2]
        p9 = padded[1:-1, 0:-2]

        # Ciclo de vecinos: P2, P3, P4, P5, P6, P7, P8, P9, P2
        neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]

        # Calcular CN
        # CRÍTICO: Los neighbors deben ser binarios (0/1) para que el CN funcione
        # Verificar que todos los neighbors sean binarios
        for i, n in enumerate(neighbors[:8]):  # Solo los primeros 8, el 9 es duplicado
            n_unique = np.unique(n)
            if len(n_unique) > 2 or not all(v in [0, 1] for v in n_unique):
                logger.error(f"Neighbor {i+2} no es binario! Valores: {n_unique}")
                neighbors[i] = (n > 0).astype(np.uint8)

        cn_sum = np.zeros(skel.shape, dtype=np.int32)
        for i in range(8):
            # Casting a int16 para evitar underflow en uint8 (0 - 1 = 255)
            n_curr = neighbors[i].astype(np.int16)
            n_next = neighbors[i + 1].astype(np.int16)
            diff = np.abs(n_curr - n_next)
            
            # Validar que las diferencias sean 0 o 1
            diff_max = diff.max()
            if diff_max > 1:
                logger.error(
                    f"Diff entre neighbors {i} y {i+1} tiene valores > 1! Max: {diff_max}. "
                    f"Esto indica que los neighbors no son binarios."
                )
            cn_sum += diff

        cn = cn_sum / 2

        # Validar que CN esté en el rango correcto (0-4 para skeleton binario)
        cn_max = cn.max()
        if cn_max > 4:
            logger.error(
                f"CN tiene valores fuera del rango esperado (0-4)! Max: {cn_max}. "
                f"Esto indica un problema con el skeleton o el cálculo del CN."
            )

        # Filtrar solo en el esqueleto
        # cn tiene floats, skel es 0/1 (o 0/255)
        # Asegurar que skel sea 0/1 para multiplicacion
        skel_bool = skel > 0
        skel_pixels = np.sum(skel_bool)

        logger.debug(
            f"Crossing Number calculado - shape: {cn.shape}, min: {cn.min()}, max: {cn.max()}, "
            f"mean: {cn[skel_bool].mean():.2f} (solo en skeleton), "
            f"skeleton pixels: {skel_pixels}"
        )

        # Estadísticas de CN en el skeleton
        cn_in_skeleton = cn[skel_bool]
        if len(cn_in_skeleton) > 0:
            unique_cn, counts = np.unique(cn_in_skeleton, return_counts=True)
            logger.debug(
                f"Distribución de CN en skeleton: {dict(zip(unique_cn.astype(int), counts, strict=True))}"
            )

        # Terminaciones (CN=1)
        terminations = (cn == 1) & skel_bool
        # Bifurcaciones (CN=3)
        bifurcations = (cn == 3) & skel_bool

        # Extraer coordenadas
        term_y, term_x = np.where(terminations)
        bif_y, bif_x = np.where(bifurcations)

        logger.debug(
            f"Crossing Number - Terminaciones encontradas: {len(term_x)}, Bifurcaciones: {len(bif_x)}"
        )

        if len(term_x) == 0 and len(bif_x) == 0 and skel_pixels > 0:
            # Si hay skeleton pero no se detectan minutiae, puede ser un problema con el CN
            logger.warning(
                f"No se detectaron minutiae pero hay {skel_pixels} píxeles en el skeleton. "
                f"CN stats: min={cn[skel_bool].min()}, max={cn[skel_bool].max()}, "
                f"mean={cn[skel_bool].mean():.2f}"
            )

        # Procesar Terminaciones
        for y, x in zip(term_y, term_x, strict=True):
            # Calcular ángulo (aún local, pero rápido porque son pocos puntos)
            # Extraer vecindario 3x3 del padded
            # Coordenadas en padded son y+1, x+1
            blk = padded[y : y + 3, x : x + 3]
            angle = self._compute_angle(blk, MinutiaType.TERMINATION)
            candidates.append(
                MinutiaCandidate(
                    x=int(x),
                    y=int(y),
                    angle=angle,
                    type=MinutiaType.TERMINATION,
                    confidence=1.0,
                    origin=AlgorithmOrigin.SKELETON,
                )
            )

        # Procesar Bifurcaciones
        for y, x in zip(bif_y, bif_x, strict=True):
            blk = padded[y : y + 3, x : x + 3]
            angle = self._compute_angle(blk, MinutiaType.BIFURCATION)
            candidates.append(
                MinutiaCandidate(
                    x=int(x),
                    y=int(y),
                    angle=angle,
                    type=MinutiaType.BIFURCATION,
                    confidence=1.0,
                    origin=AlgorithmOrigin.SKELETON,
                )
            )

        return candidates

    def _compute_angle(self, blk: np.ndarray, m_type: MinutiaType) -> float:
        # Simplificación: Calcular ángulo basado en vecinos
        # ... (lógica existente adaptada)
        # Si es terminación, ángulo hacia el vecino
        # Si es bifurcación, promedio de las 3 ramas
        center_y, center_x = 1, 1
        angles = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                if blk[i, j]:
                    # atan2(y, x) -> y es i, x es j (relativo)
                    # Ojo coordenadas imagen: y crece hacia abajo
                    dy = i - center_y
                    dx = j - center_x
                    angle = math.degrees(math.atan2(dy, dx))
                    angles.append(angle)

        if not angles:
            return 0.0

        if m_type == MinutiaType.TERMINATION:
            return angles[0]
        elif m_type == MinutiaType.BIFURCATION and len(angles) >= 3:
            # Promedio vectorial correcto
            sin_sum = sum(math.sin(math.radians(a)) for a in angles)
            cos_sum = sum(math.cos(math.radians(a)) for a in angles)
            mean_angle = math.degrees(math.atan2(sin_sum, cos_sum))
            return mean_angle

        return angles[0]

    def _create_mask(self, skel: np.ndarray) -> np.ndarray:
        """Crea una máscara para filtrar candidatas fuera del área válida."""
        mask = skel > 0
        try:
            mask = convex_hull_image(mask)
            # Usar erosión más pequeña si hay muy pocas candidatas
            # erosion_size puede ser demasiado agresivo
            erosion_kernel_size = min(
                self.erosion_size, 3
            )  # Máximo 3 para no ser tan agresivo
            if erosion_kernel_size > 0:
                mask = erosion(mask, square(erosion_kernel_size))
        except Exception as e:
            logger.warning(f"Error creando máscara convexa, usando máscara simple: {e}")
            # Si falla, usar máscara simple basada en el esqueleto
            mask = skel > 0

        return mask

    def _filter_candidates(
        self,
        candidates: list[MinutiaCandidate],
        mask: np.ndarray,
        shape: tuple[int, int],
    ) -> list[MinutiaCandidate]:
        rows, cols = shape
        filtered = []
        border_rejected = 0
        mask_rejected = 0

        for m in candidates:
            # 1. Bordes
            if m.x < self.border_margin or m.x > cols - self.border_margin:
                border_rejected += 1
                continue
            if m.y < self.border_margin or m.y > rows - self.border_margin:
                border_rejected += 1
                continue

            # 2. Máscara (Convex Hull)
            if m.y >= rows or m.x >= cols:
                mask_rejected += 1
                continue
            if not mask[m.y, m.x]:
                mask_rejected += 1
                continue

            filtered.append(m)

        if border_rejected > 0:
            logger.debug(f"Filtrado - rechazadas por bordes: {border_rejected}")
        if mask_rejected > 0:
            logger.debug(f"Filtrado - rechazadas por máscara: {mask_rejected}")

        # Si todas fueron filtradas, intentar con márgenes más pequeños y sin máscara
        if len(filtered) == 0 and len(candidates) > 0:
            logger.warning(
                f"Todas las candidatas fueron filtradas ({len(candidates)} candidatas originales). "
                f"Intentando con filtrado menos agresivo..."
            )

            # Primero intentar solo con márgenes reducidos
            original_margin = self.border_margin
            self.border_margin = max(1, self.border_margin // 2)
            filtered = []
            for m in candidates:
                if m.x < self.border_margin or m.x > cols - self.border_margin:
                    continue
                if m.y < self.border_margin or m.y > rows - self.border_margin:
                    continue
                if m.y >= rows or m.x >= cols:
                    continue
                # Intentar sin máscara primero
                filtered.append(m)

            # Si aún no hay resultados, ignorar la máscara completamente
            if len(filtered) == 0:
                logger.warning(
                    "Aún sin resultados con márgenes reducidos. Ignorando máscara convexa..."
                )
                self.border_margin = 1  # Mínimo margen
                filtered = []
                for m in candidates:
                    if m.x < 1 or m.x >= cols - 1:
                        continue
                    if m.y < 1 or m.y >= rows - 1:
                        continue
                    if m.y >= rows or m.x >= cols:
                        continue
                    filtered.append(m)

            self.border_margin = original_margin
            logger.info(
                f"Con filtrado menos agresivo se encontraron {len(filtered)} candidatas"
            )

        return filtered


class GradientRidgeExtractor:
    """
    Extractor basado en Harris Corner Detection para encontrar puntos de interés (bifurcaciones/terminaciones)
    directamente en la imagen mejorada, sin esqueletización.
    Útil como validación cruzada.
    """

    def extract(self, image: np.ndarray) -> list[MinutiaCandidate]:
        import cv2

        # Harris detecta esquinas (cambios fuertes en dos direcciones)
        # Esto correlaciona bien con bifurcaciones y terminaciones
        dst = cv2.cornerHarris(image, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)

        # Umbralización relativa
        thresh = 0.01 * dst.max()
        ret, dst_bin = cv2.threshold(dst, thresh, 255, 0)
        dst_bin = np.uint8(dst_bin)

        # Encontrar centroides de las esquinas detectadas
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_bin)

        candidates = []
        # El label 0 es el fondo
        for i in range(1, ret):
            x, y = centroids[i]

            # Harris no distingue tipo, asumimos Bifurcación (mayor complejidad local)
            # Confianza menor que Skeleton
            candidates.append(
                MinutiaCandidate(
                    x=int(x),
                    y=int(y),
                    angle=0.0,  # Harris no da orientación directa
                    type=MinutiaType.BIFURCATION,
                    confidence=0.7,
                    origin=AlgorithmOrigin.GABOR,  # Usamos GABOR como proxy de "no-skeleton"
                )
            )

        return candidates


class AiFeatureExtractor:
    """Deep-learning minutiae extractor using ONNX Runtime.

    Replaces the traditional skeletonisation + Crossing Number approach
    with a neural network that detects minutiae directly from enhanced
    fingerprint images.  The :class:`SkeletonMinutiaeExtractor` remains
    available as a fallback.

    Architecture
    ------------
    Input → :meth:`ExtractionProcessor.preprocess` → canvas padding →
    :meth:`ModelManager.run_extraction` → ONNX inference →
    :meth:`ExtractionProcessor.postprocess` → NMS + coordinate remap →
    ``list[MinutiaCandidate]``
    """

    def __init__(
        self,
        model_manager: ModelManager,
        processor: ExtractionProcessor | None = None,
    ) -> None:
        """Initialise the AI extractor.

        Args:
            model_manager: Initialised :class:`ModelManager` with a loaded
                extraction model.
            processor: Optional custom :class:`ExtractionProcessor`. A
                default instance is created when ``None``.
        """
        self.model_manager = model_manager
        self.processor = processor or ExtractionProcessor(AiConfig())

    @timed("extract_minutiae_dl")
    def extract(self, image: np.ndarray) -> list[MinutiaCandidate]:
        """Extract minutiae using the DL model inference pipeline.

        Handles grayscale conversion, blank-image guard, and model
        failure gracefully (returns empty list instead of raising).

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) uint8 image.

        Returns:
            List of :class:`MinutiaCandidate` with
            :attr:`AlgorithmOrigin.DEEP_LEARNING`, or an empty list on
            failure.
        """
        if image is None or image.size == 0:
            logger.warning("Empty image passed to AiFeatureExtractor")
            return []

        # Convert BGR to grayscale if needed
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image[:, :, 0]

        original_shape = image.shape[:2]
        try:
            input_tensor = self.processor.preprocess(image)
            raw_output = self.model_manager.run_extraction(input_tensor)
            candidates = self.processor.postprocess(raw_output, original_shape)
            return candidates
        except Exception:
            logger.exception("DL extraction failed")
            return []


# Alias para compatibilidad
MinutiaeExtractor = SkeletonMinutiaeExtractor

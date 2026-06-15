"""
Main fingerprint processing service.
Clean Code: declarative orchestrator using a Uniform Pipeline Pattern.
The entire process is just a sequence of `IPipelineStep`s operating
on a shared `PipelineContext`.
"""

import asyncio
import concurrent.futures
import logging
from dataclasses import replace
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.ai.model_manager import ModelManager
    from src.domain.forensic_rules import IForensicValidationStrategy


from src.core.interfaces import (
    AsyncPipelineStep,
    IEnhancer,
    IFeatureExtractor,
    INormalizer,
    IPipelineStep,
    PipelineContext,
)
from src.core.metrics import measure_time
from src.core.types import NormalizedFingerprint
from src.processing.enhancer import create_enhancer
from src.processing.enhancers.gpu import GpuEnhancer  # imported for type hinting
from src.processing.extractor import SkeletonMinutiaeExtractor
from src.processing.normalization import MinutiaNormalizer
from src.processing.post_hooks import (
    BorderMaskCleaner,
    BrokenRidgeHealer,
    EnsembleFusionFilter,
    LowConfidenceFilter,
    OrientationRefiner,
    QualityFilter,
    SpurRemover,
)
from src.processing.pre_hooks import (
    OrientationFieldAnalyzer,
    SingularityDetector,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapters to wrap legacy components into IPipelineStep
# ---------------------------------------------------------------------------


class EnhancerStep(IPipelineStep):
    """Wraps an enhancer as a pipeline step.

    Supports both the legacy ``IEnhancer`` (with ``enhance(img, resize)``)
    and the ``IPipelineStep``-based ``GpuEnhancer`` (with ``process(ctx)``).
    """
    def __init__(self, enhancer: IEnhancer | IPipelineStep, resize: bool = True):
        self.enhancer = enhancer
        self.resize = resize

    def process(self, ctx: PipelineContext) -> None:
        source = ctx.preprocessed_image if ctx.preprocessed_image is not None else ctx.raw_image
        if hasattr(self.enhancer, "enhance"):
            # Legacy IEnhancer path (CpuEnhancer, AiEnhancer, etc.)
            from src.core.interfaces import IEnhancer
            enh: IEnhancer = self.enhancer  # type: ignore[assignment]
            ctx.enhanced_image = enh.enhance(source, resize=self.resize)
        else:
            # IPipelineStep path (GpuEnhancer)
            step: IPipelineStep = self.enhancer  # type: ignore[assignment]
            sub_ctx = PipelineContext(raw_image=source)
            step.process(sub_ctx)
            ctx.enhanced_image = sub_ctx.enhanced_image
        ctx.preprocessed_image = ctx.enhanced_image


class MaskResizerStep(IPipelineStep):
    """Resizes the quality mask to match the enhanced image shape."""
    def process(self, ctx: PipelineContext) -> None:
        if ctx.quality_mask is not None and ctx.enhanced_image is not None:
            if ctx.quality_mask.shape != ctx.enhanced_image.shape:
                h, w = ctx.enhanced_image.shape
                ctx.quality_mask = cv2.resize(
                    ctx.quality_mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)


class ExtractorStep(IPipelineStep):
    """Wraps an IFeatureExtractor as a pipeline step."""
    def __init__(self, extractors: list[IFeatureExtractor]):
        self.extractors = extractors

    def process(self, ctx: PipelineContext) -> None:
        if ctx.enhanced_image is None:
            raise ValueError("No enhanced image available for extraction")
        ctx.candidate_groups = [ext.extract(ctx.enhanced_image) for ext in self.extractors]


class NormalizerStep(IPipelineStep):
    """Wraps an INormalizer as a pipeline step."""
    def __init__(self, normalizer: INormalizer):
        self.normalizer = normalizer

    def process(self, ctx: PipelineContext) -> None:
        if not ctx.candidates:
            logger.warning("No se encontraron minutiae finales")
        # Use the enhanced image shape for the normalizer so that
        # absolute coordinates match the image dimensions.
        img_shape = ctx.enhanced_image.shape if ctx.enhanced_image is not None else ctx.raw_image.shape
        normalized_fp = self.normalizer.normalize(
            ctx.candidates, img_shape
        )
        ctx.normalized_fingerprint = replace(normalized_fp, id=ctx.fingerprint_id)


# ---------------------------------------------------------------------------
# Default production pipeline (declarative).
# ---------------------------------------------------------------------------

def build_production_pipeline(
    enhancer: IEnhancer,
    extractors: list[IFeatureExtractor],
    normalizer: INormalizer,
    resize: bool = True,
) -> list[IPipelineStep]:
    """Constructs the canonical processing chain."""
    return [
        # 1. Enhancement (upscales, gabors)
        EnhancerStep(enhancer, resize=resize),
        
        # 2. Pre-hooks (run on the enhanced image)
        OrientationFieldAnalyzer(),
        SingularityDetector(roi_radius=140),
        
        # 3. Extraction
        ExtractorStep(extractors),
        
        # 4. Fusion
        EnsembleFusionFilter(radius=8.0, min_votes=2),
        
        # 5. Post-hooks (cleanup)
        QualityFilter(),
        SpurRemover(max_distance=10.0),
        BrokenRidgeHealer(max_distance=8.0),
        BorderMaskCleaner(border_px=0, roi_mode="core"),
        OrientationRefiner(window=16, coherence_threshold=0.65),
        LowConfidenceFilter(threshold=0.75),
        
        # 6. Normalisation
        NormalizerStep(normalizer),
    ]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class FingerprintService:
    """Orchestrates the complete fingerprint processing pipeline.

    The orchestration is fully declarative: a single :class:`PipelineContext`
    is created per image and threaded through the steps.
    """

    def __init__(
        self,
        enhancer: Optional[IEnhancer] = None,
        extractors: Optional[List[IFeatureExtractor]] = None,
        normalizer: Optional[INormalizer] = None,
        # Legacy hook injection (converted internally to pipeline steps)
        pre_hooks: Optional[List[IPipelineStep]] = None,
        fuse_hooks: Optional[List[IPipelineStep]] = None,
        post_hooks: Optional[List[IPipelineStep]] = None,
        # Legacy aliases
        extractor: Optional[IFeatureExtractor] = None,
        pre_processors: Optional[List[IPipelineStep]] = None,
        post_processors: Optional[List[IPipelineStep]] = None,
        # RAG Phase 10: forensic business rules
        validation_strategy: Optional["IForensicValidationStrategy"] = None,
    ):
        self._enhancer: IEnhancer = enhancer or create_enhancer()
        if extractors is not None:
            self._extractors: list[IFeatureExtractor] = list(extractors)
        elif extractor is not None:
            self._extractors = [extractor]
        else:
            self._extractors = [SkeletonMinutiaeExtractor()]

        self.normalizer = normalizer or MinutiaNormalizer()
        self.validation_strategy = validation_strategy

        # Build dynamic step list from provided hooks or defaults
        self.steps: list[IPipelineStep] = []
        self.steps.append(EnhancerStep(self.enhancer, resize=True))

        chosen_pre = pre_hooks if pre_hooks is not None else pre_processors
        if chosen_pre is not None:
            self.steps.extend(chosen_pre)
            self.steps.append(MaskResizerStep())
        else:
            self.steps.extend([OrientationFieldAnalyzer(), SingularityDetector(roi_radius=140)])

        self.steps.append(ExtractorStep(self.extractors))
        self.steps.append(EnsembleFusionFilter(radius=8.0, min_votes=2))

        chosen_post = post_hooks if post_hooks is not None else post_processors
        if chosen_post is not None:
            self.steps.extend(chosen_post)
        else:
            self.steps.extend([
                QualityFilter(),
                SpurRemover(max_distance=10.0),
                BrokenRidgeHealer(max_distance=8.0),
                BorderMaskCleaner(border_px=0, roi_mode="core"),
                OrientationRefiner(window=16, coherence_threshold=0.65),
                LowConfidenceFilter(threshold=0.75),
            ])

        self.steps.append(NormalizerStep(self.normalizer))

        self.use_parallel = True
        self._max_workers: Optional[int] = None
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @property
    def enhancer(self) -> IEnhancer:
        return self._enhancer

    @property
    def extractors(self) -> list[IFeatureExtractor]:
        return self._extractors

    @property
    def pre_processors(self) -> List[IPipelineStep]:
        return [s for s in self.steps if isinstance(s, IPipelineStep)]

    @property
    def post_processors(self) -> List[IPipelineStep]:
        return [s for s in self.steps if isinstance(s, IPipelineStep)]

    @property
    def extractor(self) -> IFeatureExtractor:
        return self.extractors[0]

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_image(
        self, image: np.ndarray, fingerprint_id: str = "unknown", resize: bool = True
    ) -> NormalizedFingerprint:
        if image is None:
            raise ValueError("La imagen no puede ser None")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ctx = PipelineContext(raw_image=image, fingerprint_id=fingerprint_id)

        # Update the resize flag on the EnhancerStep if requested
        for step in self.steps:
            if isinstance(step, EnhancerStep):
                step.resize = resize

        with measure_time("process_pipeline"):
            for step in self.steps:
                step.process(ctx)

        if self.validation_strategy is not None:
            self._apply_validation(ctx)

        if ctx.normalized_fingerprint is None:
            raise RuntimeError("Pipeline finished but no normalized_fingerprint was produced")
            
        return ctx.normalized_fingerprint

    def _apply_validation(self, ctx: PipelineContext) -> None:
        """Apply the forensic business rule to the extracted candidates.

        Strategy is injected (Clean Architecture). The service does
        not know whether this is enrollment or search — it only knows
        that SOME rule was wired in.
        """
        if self.validation_strategy is None:
            return
        # Validation is enforced at the domain level. If candidates
        # are too few, this raises InsufficientFeaturesError and
        # halts the pipeline before database/vector overhead.
        self.validation_strategy.validate(ctx.candidates)

    async def process_image_async(
        self, image: np.ndarray, fingerprint_id: str = "unknown", resize: bool = True
    ) -> NormalizedFingerprint:
        """Async version of :meth:`process_image`.

        Processes a fingerprint image without blocking the event loop.
        CPU-bound steps run in a thread-pool executor; steps that
        implement :class:`AsyncPipelineStep` run their native async path.
        """
        if image is None:
            raise ValueError("La imagen no puede ser None")
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ctx = PipelineContext(raw_image=image, fingerprint_id=fingerprint_id)

        for step in self.steps:
            if isinstance(step, EnhancerStep):
                step.resize = resize

        with measure_time("process_pipeline"):
            for step in self.steps:
                if isinstance(step, AsyncPipelineStep):
                    await step.process_async(ctx)
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, step.process, ctx)

        if ctx.normalized_fingerprint is None:
            raise RuntimeError("Pipeline finished but no normalized_fingerprint was produced")
        return ctx.normalized_fingerprint

    # ------------------------------------------------------------------
    # Batch / IO helpers
    # ------------------------------------------------------------------

    def process_batch(
        self,
        images: List[np.ndarray],
        fingerprint_ids: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[Optional[NormalizedFingerprint]]:
        if fingerprint_ids is None:
            fingerprint_ids = ["unknown"] * len(images)
        results: List[Optional[NormalizedFingerprint]] = [None] * len(images)

        if not self.use_parallel:
            for i, (img, fid) in enumerate(zip(images, fingerprint_ids)):
                try:
                    results[i] = self.process_image(img, fingerprint_id=fid)
                except Exception as e:
                    logger.error("Error procesando imagen %s: %s", fid, e)
            return results

        with concurrent.futures.ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_wrapper, img, fid): i
                for i, (img, fid) in enumerate(zip(images, fingerprint_ids))
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Error en worker %d: %s", idx, e)
                    results[idx] = None
        return results

    @staticmethod
    def _process_wrapper(img: np.ndarray, fid: str) -> NormalizedFingerprint:
        return FingerprintService().process_image(img, fingerprint_id=fid)

    def process_image_from_path(self, image_path: str) -> NormalizedFingerprint:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        return self.process_image(image, fingerprint_id=image_path)

    def process_image_from_bytes(
        self, image_bytes: bytes, fingerprint_id: str = "unknown"
    ) -> NormalizedFingerprint:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(
                "No se pudo decodificar la imagen. "
                "Verifica que el archivo sea una imagen válida (BMP, PNG, JPEG) y no esté corrupta."
            )
        if image.size == 0:
            raise ValueError("La imagen decodificada está vacía")
        if image.shape[0] < 50 or image.shape[1] < 50:
            logger.warning("Imagen muy pequeña: %s - podría no tener suficiente detalle", image.shape)
        return self.process_image(image, fingerprint_id=fingerprint_id, resize=True)


# ---------------------------------------------------------------------------
# Module-level singleton + AI factory (unchanged behaviour).
# ---------------------------------------------------------------------------


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

    AiConfig()
    enhancer: Optional[IEnhancer] = None
    if use_segmentation and use_enhancement:
        enhancer = create_enhancer("full_ai", model_manager=model_manager)
    elif use_segmentation:
        enhancer = create_enhancer("segmentation", model_manager=model_manager)
    elif use_enhancement:
        enhancer = create_enhancer("enhancement", model_manager=model_manager)

    extractors: List[IFeatureExtractor] = []
    if use_dl_extractor:
        extractors.append(AiFeatureExtractor(model_manager))

    return FingerprintService(
        enhancer=enhancer,
        extractors=extractors if extractors else None,
    )

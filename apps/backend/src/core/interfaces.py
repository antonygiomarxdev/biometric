"""
Interfaces base para el pipeline biométrico.
Clean Architecture: Dependency Inversion via Structural Subtyping (Protocols).

Utiliza un patrón de Pipeline Uniforme (Nivel 4 de extensibilidad) donde
todos los componentes de procesamiento de imagen son `IPipelineStep`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol, List, runtime_checkable
import numpy as np

from src.core.types import MinutiaCandidate, NormalizedFingerprint, MatchResult, RidgeGraph, TripletVector


# ---------------------------------------------------------------------------
# Pipeline context: single source of truth for shared state.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PipelineContext:
    """Per-image state that flows through the pipeline.

    The :class:`FingerprintService` creates one of these per image and
    threads it through every step. Hooks read what they need and write
    what they produce. Using a typed context object keeps the
    orchestrator generic and open to extension.
    """

    # Input (always present)
    raw_image: np.ndarray
    fingerprint_id: str = "unknown"

    # Stage outputs
    preprocessed_image: np.ndarray | None = None
    quality_mask: np.ndarray | None = None  # bool, True = valid
    roi_mask: np.ndarray | None = None  # bool, True = inside the core ROI disc
    orientation_field: np.ndarray | None = None  # radians, per-block
    coherence_field: np.ndarray | None = None  # [0..1], per-block

    # Enhancement output
    enhanced_image: np.ndarray | None = None  # after Gabor / binarisation
    skeleton: np.ndarray | None = None        # strict 0/1 binary skeleton

    # Detector outputs
    candidate_groups: list[list[MinutiaCandidate]] = field(default_factory=list)
    candidates: list[MinutiaCandidate] = field(default_factory=list)

    # Core singularity (forensic anchor). Populated by SingularityDetector.
    core: tuple[int, int] | None = None

    # Final Output
    normalized_fingerprint: NormalizedFingerprint | None = None

    # Phase 11: Ridge Graph (biological topology)
    ridge_graph: RidgeGraph | None = None

    # Provenance / debugging
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Uniform Pipeline Step
# ---------------------------------------------------------------------------


@runtime_checkable
class IPipelineStep(Protocol):
    """Uniform interface for all fingerprint processing components.

    Enhancers, Extractors, Post-Processors, and Normalizers all implement
    this protocol. They read from and mutate the `PipelineContext`.
    """
    def process(self, ctx: PipelineContext) -> None: ...


class AsyncPipelineStep:
    """Base class for steps that want non-blocking execution.

    Override :meth:`process_async` if the step has native async I/O
    (e.g. GPU kernel launches).  The default wraps :meth:`process`
    in a thread-pool executor, keeping the event loop free.
    """

    def process(self, ctx: PipelineContext) -> None:
        """Default sync no-op. Subclasses must override one of process/process_async."""

    async def process_async(self, ctx: PipelineContext) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.process, ctx)


# (Legacy aliases removed — use the Canonical protocols below)


# ---------------------------------------------------------------------------
# Canonical protocols for the processing layer (enhance / extract / normalise)
# ---------------------------------------------------------------------------


class IEnhancer(Protocol):
    """Image enhancement interface (used by CpuEnhancer, GpuEnhancer, etc.)."""
    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray: ...

@runtime_checkable
class IFeatureExtractor(Protocol):
    """Feature extraction interface (used by SkeletonMinutiaeExtractor, etc.)."""
    def extract(self, image: np.ndarray) -> List[MinutiaCandidate]: ...


class INormalizer(Protocol):
    """Normalisation interface (used by MinutiaNormalizer, etc.)."""
    def normalize(self, minutiae: List[MinutiaCandidate], img_shape: tuple[int, int]) -> NormalizedFingerprint: ...


# ---------------------------------------------------------------------------
# Matcher (separate subsystem)
# ---------------------------------------------------------------------------


class IMatcher(Protocol):
    """Protocolo para el motor de búsqueda biométrica."""

    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        """
        Busca coincidencias para un vector único.
        Args:
            probe: Vector de características a buscar.
            top_k: Número máximo de resultados a retornar.
        Returns:
            El mejor resultado encontrado.
        """
        ...

    async def match_batch(self, probes: np.ndarray, top_k: int = 5) -> List[MatchResult]:
        """
        Busca coincidencias para múltiples vectores en lote.
        Args:
            probes: Matriz de vectores.
            top_k: Resultados por vector.
        Returns:
            Lista de resultados correspondientes.
        """
        ...


class IVectorizer(Protocol):
    """Protocol for fingerprint vectorization (RAG chunking).

    Returns a list of `TripletVector` chunks, not a single global vector.
    Each chunk represents a local invariant structure.
    """
    def vectorize(self, ctx: PipelineContext) -> List[TripletVector]: ...

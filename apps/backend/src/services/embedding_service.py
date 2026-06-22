from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray

from src.api.prefix import API_PREFIX
from src.db.models import Person

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.ai.loader import ModelLoader

from src.db.qdrant_embedding_repository import (
    QdrantEmbeddingRepository,
    QdrantHit,
    QdrantPayload,
)
from src.services.sliding_window import (
    AggregatedHit,
    aggregate_hits_by_person,
    sliding_window_crops,
)

logger = logging.getLogger(__name__)

GRADCAM_LAYER: str = "stages.3.blocks.2.conv_dw"
PROBE_IMAGE_SIZE: int = 224


# Search modes.  ``single`` is the default — one embedding, one
# query, ~15 ms.  ``ensemble`` runs a sliding window over the probe
# and aggregates hits by ``person_id`` (max-pool).  ~135 ms.  Use
# ``ensemble`` for latentes and partial prints; use ``single`` for
# clean full prints (faster, identical quality for that case).
SEARCH_MODE_SINGLE: str = "single"
SEARCH_MODE_ENSEMBLE: str = "ensemble"
VALID_SEARCH_MODES: frozenset[str] = frozenset(
    {SEARCH_MODE_SINGLE, SEARCH_MODE_ENSEMBLE},
)


class GradCAMHook:
    """Forward + backward hook pair, attached/detached inside a lock.

    Single-use per call: ``create → forward → backward → compute → remove``.
    The lock around the whole sequence is enforced by
    ``EmbeddingService._compute_gradcam_locked`` so concurrent calls
    never share hook state.
    """

    def __init__(self, layer: nn.Module) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # PyTorch types the backward-hook signature with a TypeVar
        # (_grad_t = Tensor | tuple[Tensor, ...] | None) that
        # pyright cannot reconcile with our narrower types.  The
        # runtime check in ``_bwd`` handles the actual values.
        self.fwd = layer.register_forward_hook(self._fwd)
        self.bwd = layer.register_full_backward_hook(self._bwd)  # type: ignore[arg-type]

    def _fwd(
        self,
        _mod: nn.Module,
        _inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        self.activations = out.detach()

    def _bwd(
        self,
        _mod: nn.Module,
        _grad_in: object,
        grad_out: object,
    ) -> None:
        """Backward hook for ``register_full_backward_hook``.

        PyTorch types the second/third parameters as ``_grad_t``
        (Tensor | tuple[Tensor, ...] | None) and narrows them
        differently across the public stubs vs the runtime ABI.
        We accept the loose signature and narrow at runtime.
        """
        first: object = None
        if isinstance(grad_out, tuple) and grad_out:
            first = grad_out[0]
        elif isinstance(grad_out, torch.Tensor):
            first = grad_out
        if isinstance(first, torch.Tensor):
            self.gradients = first.detach()

    def remove(self) -> None:
        self.fwd.remove()
        self.bwd.remove()

    def compute(self) -> NDArray[np.float32]:
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hook not initialised: run forward+backward first")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
        lo = float(cam_np.min())
        hi = float(cam_np.max())
        return (cam_np - lo) / (hi - lo + 1e-8)


class SearchCandidate:
    """One rank-ordered match candidate returned by ``search``."""

    def __init__(
        self,
        person_id: str,
        score: float,
        full_name: str | None,
        external_id: str | None,
        image_url: str | None,
        capture_id: str,
        finger_name: str | None,
    ) -> None:
        self.person_id = person_id
        self.score = score
        self.full_name = full_name
        self.external_id = external_id
        self.image_url = image_url
        self.capture_id = capture_id
        self.finger_name = finger_name

    def to_dict(self) -> dict[str, str | float | None]:
        return {
            "person_id": self.person_id,
            "score": self.score,
            "full_name": self.full_name,
            "external_id": self.external_id,
            "image_url": self.image_url,
            "capture_id": self.capture_id,
            "finger_name": self.finger_name,
        }


class EmbeddingService:
    """AFR-Net embedding + GradCAM explainability.

    All public methods are ``async``.  CPU-bound work (cv2 decode,
    tensor move) runs in ``ModelLoader.pool`` (a dedicated
    ThreadPoolExecutor) so the event loop stays responsive.  The
    actual forward/backward through the PyTorch model is serialised
    by ``ModelLoader.inference_lock`` because PyTorch is not
    thread-safe when forward hooks are attached, and CUDA streams
    must not race.
    """

    def __init__(self, loader: ModelLoader,
                 qdrant: "QdrantEmbeddingRepository") -> None:
        self._loader = loader
        self._qdrant = qdrant

    @property
    def _device(self) -> str:
        return self._loader.device

    async def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._loader.pool, self._preprocess_sync, image_bytes,
        )

    def _preprocess_sync(self, image_bytes: bytes) -> torch.Tensor:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError("Failed to decode image bytes")
        # ``cv2.imdecode`` returns ``MatLike``; coerce to a strict
        # ``NDArray[uint8]`` so the rest of the pipeline stays typed.
        img: NDArray[np.uint8] = np.asarray(decoded, dtype=np.uint8)
        # Auto-center the fingerprint by trimming empty borders, then
        # padding back to a square.  SOCOFing (and the AFR-Net training
        # set) places the ridge pattern in the centre of a fixed
        # canvas; a user-uploaded crop may sit in a corner and the
        # affine resize destroys ridge spacing.
        centered = _center_on_content(img)
        resized = cv2.resize(centered, (PROBE_IMAGE_SIZE, PROBE_IMAGE_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor.to(self._device)

    @torch.no_grad()  # type: ignore[arg-type]
    def _embed_sync(self, x: torch.Tensor) -> NDArray[np.float32]:
        output = self._loader.embedding_model(x)
        return output["embedding"].squeeze().detach().cpu().numpy().astype(np.float32)

    def _compute_gradcam_sync(self, x: torch.Tensor) -> NDArray[np.float32]:
        with torch.enable_grad():
            model = self._loader.embedding_model
            target_layer = dict(model.cnn.named_modules())[GRADCAM_LAYER]
            hook = GradCAMHook(target_layer)
            try:
                model.zero_grad()
                x_in = x.detach().clone().requires_grad_(True)
                cnn_feat = model.cnn(x_in)
                loss = cnn_feat.sum()
                loss.backward()
                return hook.compute()
            finally:
                hook.remove()

    async def _embed_with_gradcam(self, x: torch.Tensor
                                   ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        async with self._loader.inference_lock:
            loop = asyncio.get_running_loop()
            emb = await loop.run_in_executor(
                self._loader.pool, self._embed_sync, x,
            )
            cam = await loop.run_in_executor(
                self._loader.pool, self._compute_gradcam_sync, x,
            )
            return emb, cam

    async def _embed_only(self, x: torch.Tensor) -> NDArray[np.float32]:
        async with self._loader.inference_lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._loader.pool, self._embed_sync, x,
            )

    @staticmethod
    def _heatmap_to_base64(cam: NDArray[np.float32]) -> str:
        cam_resized = cv2.resize(cam, (PROBE_IMAGE_SIZE, PROBE_IMAGE_SIZE))
        heatmap = (cam_resized * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
        _, buf = cv2.imencode(".png", colormap)
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    async def embed(self, image_bytes: bytes, enhance: bool = False,
                    with_gradcam: bool = False
                    ) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
        """Compute the 512-D embedding for a probe image.

        ``with_gradcam=True`` also computes the GradCAM heatmap (used
        by the search endpoint for explainability).  GradCAM is **not**
        computed on enrollment — it is a backward pass and would double
        the latency on the write path for no benefit.
        """
        x = await self._preprocess(image_bytes)

        if enhance:
            unet = self._loader.unet_model
            if unet is not None:
                async with self._loader.inference_lock:
                    loop = asyncio.get_running_loop()
                    x = await loop.run_in_executor(
                        self._loader.pool, lambda inp: unet(inp), x,
                    )

        if with_gradcam:
            return await self._embed_with_gradcam(x)
        emb = await self._embed_only(x)
        return emb, None

    async def enroll(self, image_bytes: bytes, capture_id: str,
                     person_id: str, finger_name: str | None = None
                     ) -> str:
        emb, _ = await self.embed(image_bytes, with_gradcam=False)
        fingerprint_id = f"{person_id}__{capture_id}"
        payload: QdrantPayload = {
            "person_id": person_id,
            "capture_id": capture_id,
        }
        if finger_name:
            payload["finger_name"] = finger_name
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._loader.pool,
            self._qdrant.upsert,
            fingerprint_id, emb, payload,
        )
        return fingerprint_id

    async def search(self, image_bytes: bytes, top_k: int = 10,
                     enhance: bool = False,
                     mode: str = SEARCH_MODE_SINGLE,
                     session: AsyncSession | None = None,
                     ) -> dict[str, object]:
        if mode not in VALID_SEARCH_MODES:
            raise ValueError(
                f"Invalid search mode: {mode!r}. "
                f"Valid modes: {sorted(VALID_SEARCH_MODES)}",
            )

        t0 = time.monotonic()
        emb, cam = await self.embed(image_bytes, enhance=enhance,
                                     with_gradcam=True)
        embed_ms = int((time.monotonic() - t0) * 1000)

        loop = asyncio.get_running_loop()

        if mode == SEARCH_MODE_SINGLE:
            hits: list[QdrantHit] = await loop.run_in_executor(
                self._loader.pool, self._qdrant.search, emb, top_k,
            )
            search_ms = int((time.monotonic() - t0) * 1000) - embed_ms
            aggregated: list[AggregatedHit] = [
                AggregatedHit(
                    person_id=str(h["payload"].get("person_id", "")),
                    score=float(h["score"]),
                    capture_id=str(h["payload"].get("capture_id", "")),
                    finger_name=(
                        str(h["payload"].get("finger_name"))
                        if h["payload"].get("finger_name") is not None
                        else None
                    ),
                    fingerprint_id=str(h.get("fingerprint_id", "")),
                    contributing_crops=1,
                )
                for h in hits
            ]
        else:
            # Ensemble mode: sliding window + max-pool by person_id.
            # We batch all crops into a single forward pass for
            # ~5-10x speedup over sequential embed.
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise ValueError("Failed to decode image for ensemble crops")
            decoded = np.asarray(decoded, dtype=np.uint8)
            # Preprocess the WHOLE probe first (centre + pad to a
            # square).  Without this, sliding_window_crops on a
            # small probe (e.g. a 24x25 corner crop) produces a
            # single padded crop that contains the same content as
            # the original — no diversity, no benefit.  Centring
            # first ensures each crop covers a distinct region of
            # the fingerprint.
            centred_full = _center_on_content(decoded)
            crops = sliding_window_crops(centred_full)
            logger.debug(
                "Ensemble mode: %d crops from %s image",
                len(crops), centred_full.shape,
            )
            # Preprocess each crop and stack into a batch tensor.
            batch_tensors: list[torch.Tensor] = []
            for crop in crops:
                resized = cv2.resize(
                    crop, (PROBE_IMAGE_SIZE, PROBE_IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                t = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)
                t = (t / 255.0 - 0.5) / 0.5
                batch_tensors.append(t)
            batch_input: torch.Tensor = torch.cat(batch_tensors, dim=0).to(
                self._device,
            )
            # One forward pass for the whole batch.  PyTorch dispatches
            # in parallel on the GPU; on CPU it's roughly the same
            # wall-clock as a single inference (memory bandwidth bound).
            async with self._loader.inference_lock:
                loop_async = asyncio.get_running_loop()
                raw_emb = await loop_async.run_in_executor(
                    self._loader.pool, self._embed_sync, batch_input,
                )
            # ``raw_emb`` may be ``(N, 512)`` for ``N > 1`` or ``(512,)``
            # for a single-element batch (squeeze drops the leading
            # axis when it has size 1).  Force a 2-D shape so the
            # per-crop indexing below is always correct.
            if raw_emb.ndim == 1:
                raw_emb = raw_emb.reshape(1, -1)
            embeddings: list[NDArray[np.float32]] = [
                raw_emb[i].astype(np.float32) for i in range(len(crops))
            ]
            # Query Qdrant for each crop in parallel.  Qdrant is
            # network-bound so the 9 queries overlap nicely.
            async def query_one(
                vec: NDArray[np.float32],
            ) -> list[QdrantHit]:
                return await loop_async.run_in_executor(
                    self._loader.pool, self._qdrant.search, vec, top_k,
                )
            hits_per_crop: list[list[QdrantHit]] = await asyncio.gather(
                *(query_one(vec) for vec in embeddings),
            )
            search_ms = int((time.monotonic() - t0) * 1000) - embed_ms
            # The aggregator takes a structurally-compatible plain
            # dict type; cast through ``list[list[dict[str, object]]]``
            # so the TypedDict type-narrowing is happy.
            dict_hits: list[list[dict[str, object]]] = [
                [dict(h) for h in hits] for hits in hits_per_crop
            ]
            aggregated_raw: list[AggregatedHit] = aggregate_hits_by_person(
                dict_hits,
            )
            aggregated: list[AggregatedHit] = aggregated_raw[:top_k]

        gradcam_b64 = self._heatmap_to_base64(cam) if cam is not None else ""

        candidates: list[SearchCandidate] = []
        for agg in aggregated:
            pid = agg["person_id"]
            capture_id = agg["capture_id"]
            finger_name = agg["finger_name"]
            full_name: str | None = None
            external_id: str | None = None
            person_uuid: uuid.UUID | None = None
            if pid:
                try:
                    person_uuid = uuid.UUID(str(pid))
                except ValueError:
                    person_uuid = None
            if session is not None and person_uuid is not None:
                person: Person | None = await session.get(Person, person_uuid)
                if person is not None:
                    full_name = person.full_name
                    external_id = person.external_id

            image_url = (
                f"{API_PREFIX}/captures/{capture_id}/image" if capture_id else None
            )
            candidates.append(SearchCandidate(
                person_id=pid,
                score=float(agg["score"]),
                full_name=full_name,
                external_id=external_id,
                image_url=image_url,
                capture_id=capture_id,
                finger_name=finger_name,
            ))

        total_ms = int((time.monotonic() - t0) * 1000)
        logger.debug(
            "Search: mode=%s embed=%dms search=%dms total=%dms candidates=%d",
            mode, embed_ms, search_ms, total_ms, len(candidates),
        )

        return {
            "query_time_ms": total_ms,
            "probe_gradcam_b64": gradcam_b64,
            "enhance_applied": enhance,
            "search_mode": mode,
            "total_candidates": len(candidates),
            "candidates": [c.to_dict() for c in candidates],
        }


def _center_on_content(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Trim empty borders, then re-pad to a square centred canvas.

    The AFR-Net was trained on SOCOFing where the fingerprint sits in
    the middle of the image.  When a user uploads a tightly cropped
    image (a latent, a sub-region, an ROI), the ridge pattern ends up
    in a corner and the resize stretches the empty pixels.  We detect
    the bounding box of non-zero content, crop to it, then re-pad the
    shorter side with zeros so the fingerprint lands back in the
    centre before the model sees it.
    """
    # Threshold: anything below 5 (very dark) is "ink".  SOCOFing is
    # black ridges on near-white background, so we use the inverse.
    mask = (img > 5).astype(np.uint8)
    if not mask.any():
        # Pure black image: keep the pad-and-resize path.
        h, w = img.shape
        max_side = max(h, w)
        top = (max_side - h) // 2
        left = (max_side - w) // 2
        padded = cv2.copyMakeBorder(
            img, top, max_side - h - top,
            left, max_side - w - left,
            cv2.BORDER_CONSTANT, value=0,
        )
        return np.asarray(padded, dtype=np.uint8)
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    cropped = img[y0:y1, x0:x1]
    ch, cw = cropped.shape
    side = max(ch, cw)
    top = (side - ch) // 2
    left = (side - cw) // 2
    padded = cv2.copyMakeBorder(
        cropped, top, side - ch - top,
        left, side - cw - left,
        cv2.BORDER_CONSTANT, value=0,
    )
    return np.asarray(padded, dtype=np.uint8)

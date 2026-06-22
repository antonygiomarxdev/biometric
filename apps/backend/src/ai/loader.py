from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from src.ai.models.afrnet import AFRNetFingerprint
from src.ai.models.unet import UNet
from src.core.config import config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Lazy loader for the embedding + U-Net models.

    Concurrency model: the inference path is **event-driven** (asyncio),
    not thread-based.  PyTorch's CPU and CUDA forward/backward passes
    are released into a *dedicated* ``ThreadPoolExecutor`` so the FastAPI
    event loop is never blocked by image preprocessing or tensor moves.

    Serialisation of the actual forward call is enforced by an
    ``asyncio.Lock`` — PyTorch is not thread-safe when forward hooks
    (e.g. GradCAM) are attached, and CUDA streams must be serialised
    per-process to avoid cuda kernel launch races.  This is *good*
    throughput because:

    * Image decoding (cv2), MinIO I/O, and the database session all
      run concurrently while the lock is held by a single inference.
    * Multiple requests share the same GPU/CPU model — there is no
      model duplication.
    * Throughput is bounded by inference latency, not by request count.
    """

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._embedding_model: AFRNetFingerprint | None = None
        self._unet_model: UNet | None = None
        self._inference_lock = asyncio.Lock()
        cpu_count = os.cpu_count() or 4
        self._pool = ThreadPoolExecutor(
            max_workers=min(4, cpu_count),
            thread_name_prefix="model-io",
        )

    async def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    @property
    def inference_lock(self) -> asyncio.Lock:
        return self._inference_lock

    @property
    def pool(self) -> ThreadPoolExecutor:
        return self._pool

    @property
    def embedding_model(self) -> AFRNetFingerprint:
        if self._embedding_model is None:
            self._load_embedding_model()
        return self._embedding_model  # type: ignore[return-value]

    @property
    def unet_model(self) -> UNet | None:
        if self._unet_model is None:
            self._unet_model = self._load_unet()
        return self._unet_model

    def _load_embedding_model(self) -> None:
        path = Path(config.embedding_model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Embedding model not found at {path}. "
                f"Copy best_model.pt from spike: cp .planning/spikes/06-afrnet-baseline/best_model.pt {path}"
            )
        logger.info("Loading embedding model from %s on %s", path, self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = AFRNetFingerprint(
            num_classes=config.embedding_num_classes,
            embedding_dim=config.embedding_dim,
            s=30.0, m=0.5,
        ).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._embedding_model = model
        n = sum(p.numel() for p in model.parameters())
        logger.info("Embedding model loaded: %d params on %s", n, self.device)

    def _load_unet(self) -> UNet | None:
        path = Path(config.unet_model_path)
        if not path.exists():
            logger.warning("U-Net model not found at %s, skipping", path)
            return None
        logger.info("Loading U-Net from %s on %s", path, self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model = UNet(in_ch=1, out_ch=1, base=32, depth=4).to(self.device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        n = sum(p.numel() for p in model.parameters())
        logger.info("U-Net loaded: %d params on %s", n, self.device)
        return model

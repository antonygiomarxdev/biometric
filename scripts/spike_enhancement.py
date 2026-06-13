#!/usr/bin/env python3
"""
Spike: U-Net with perceptual loss for fingerprint enhancement on SOCOFing.

Evaluates U-Net (segmentation-models-pytorch) vs simple CNN autoencoder
for the enhancement architecture decision (D-03). Produces ONNX models,
comparative metrics, and a documented recommendation.

Usage:
    python scripts/spike_enhancement.py                    # Full run (50 epochs)
    python scripts/spike_enhancement.py --epochs 5 --quick # Quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup — Spanish messages for user-facing output, English for code
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("spike_enhancement")

# ---------------------------------------------------------------------------
# Backend path for SkeletonMinutiaeExtractor
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_SRC = PROJECT_ROOT / "apps" / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(BACKEND_SRC.parent))  # apps/backend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOCOFING_DEFAULT = str(PROJECT_ROOT / "data" / "SOCOFing")
MODELS_DIR = PROJECT_ROOT / "data" / "models"
INPUT_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpikeMetrics:
    """Per-model evaluation metrics for the spike."""

    psnr: float = 0.0
    ssim: float = 0.0
    minutiae_recovery: float = 0.0
    inference_ms: float = 0.0
    model_size_mb: float = 0.0
    gpu_memory_mb: float = 0.0


@dataclass
class TrainingHistory:
    """Training history for one architecture."""

    name: str = ""
    train_losses: list[float] = field(default_factory=list)
    val_psnrs: list[float] = field(default_factory=list)
    val_ssims: list[float] = field(default_factory=list)
    val_recoveries: list[float] = field(default_factory=list)
    best_metrics: SpikeMetrics = field(default_factory=SpikeMetrics)


# ---------------------------------------------------------------------------
# SOCOFing dataset
# ---------------------------------------------------------------------------


def _parse_socofing_filename(
    filename: str,
) -> Optional[dict[str, str]]:
    """Extract subject metadata from a SOCOFing filename.

    Real format:   {id}__{gender}_{hand}_{finger}.BMP
    Altered format: {id}__{gender}_{hand}_{finger}_{suffix}.BMP

    Returns a dict with keys: subject_id, gender, hand, finger, suffix (None for Real).
    """
    stem = filename.replace(".BMP", "")
    parts = stem.split("__")
    if len(parts) != 2:
        return None
    subject_id = parts[0]
    rest = parts[1]
    # Altered files have a suffix after the finger name
    # The finger is always the last meaningful component before optional suffix
    tokens = rest.split("_")
    if len(tokens) < 3:
        return None
    # gender is first token, hand is second
    gender = tokens[0]
    hand = tokens[1]
    # finger is everything between hand and optional suffix
    # suffixes are known: CR, Obl, Zcut
    known_suffixes = {"CR", "Obl", "Zcut"}
    if tokens[-1] in known_suffixes:
        suffix = tokens[-1]
        finger = "_".join(tokens[2:-1])
    else:
        suffix = None
        finger = "_".join(tokens[2:])

    return {
        "subject_id": subject_id,
        "gender": gender,
        "hand": hand,
        "finger": finger,
        "suffix": suffix,
    }


def _build_socofing_pairs(
    dataset_path: str, altered_level: str = "Altered-Easy"
) -> list[tuple[str, str]]:
    """Build (real_path, altered_path) pairs from SOCOFing.

    Pairs are created by matching the base identifier (subject_id, gender, hand, finger)
    between Real/ and Altered/{altered_level}/ directories.

    Args:
        dataset_path: Path to the SOCOFing dataset root.
        altered_level: One of 'Altered-Easy', 'Altered-Medium', 'Altered-Hard'.

    Returns:
        List of (real_image_path, altered_image_path) tuples.
    """
    real_dir = Path(dataset_path) / "Real"
    altered_dir = Path(dataset_path) / "Altered" / altered_level

    if not real_dir.is_dir():
        raise FileNotFoundError(
            f"No se encuentra el directorio Real/ en {dataset_path}"
        )
    if not altered_dir.is_dir():
        raise FileNotFoundError(
            f"No se encuentra el directorio {altered_level}/ en {dataset_path}"
        )

    # Index real files by base identifier
    real_index: dict[str, str] = {}
    for fpath in real_dir.glob("*.BMP"):
        meta = _parse_socofing_filename(fpath.name)
        if meta is not None and meta["suffix"] is None:
            key = f"{meta['subject_id']}__{meta['gender']}_{meta['hand']}_{meta['finger']}"
            real_index[key] = str(fpath)

    # Match altered files to real files
    pairs: list[tuple[str, str]] = []
    for fpath in altered_dir.glob("*.BMP"):
        meta = _parse_socofing_filename(fpath.name)
        if meta is not None and meta["suffix"] is not None:
            # Construct key without the suffix
            key = f"{meta['subject_id']}__{meta['gender']}_{meta['hand']}_{meta['finger']}"
            if key in real_index:
                pairs.append((real_index[key], str(fpath)))

    logger.info(
        "Se encontraron %d pares Real-%s en %s",
        len(pairs),
        altered_level,
        dataset_path,
    )
    return pairs


class SOCOFingDataset(Dataset):
    """PyTorch Dataset for paired SOCOFing fingerprint images.

    Each sample returns (real_image, altered_image) as tensors.
    Real images are the clean ground truth; altered images are the degraded input.
    """

    def __init__(
        self,
        dataset_path: str,
        altered_level: str = "Altered-Easy",
        split: str = "train",
        train_ratio: float = 0.8,
        seed: int = 42,
        quick: bool = False,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_path: Path to SOCOFing dataset root.
            altered_level: Altered difficulty level.
            split: 'train' or 'val'.
            train_ratio: Fraction of data for training.
            seed: Random seed for reproducibility.
            quick: Use a small subset (100 pairs) for smoke testing.
        """
        super().__init__()
        self.input_size = INPUT_SIZE

        pairs = _build_socofing_pairs(dataset_path, altered_level)

        if quick:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(pairs), size=min(100, len(pairs)), replace=False)
            pairs = [pairs[i] for i in indices]
            logger.info(
                "Modo rápido: usando %d pares para pruebas",
                len(pairs),
            )

        # Deterministic split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(pairs))
        split_idx = int(len(pairs) * train_ratio)

        if split == "train":
            self.pairs = [pairs[i] for i in indices[:split_idx]]
        else:
            self.pairs = [pairs[i] for i in indices[split_idx:]]

        logger.info(
            "Dataset '%s': %d muestras cargadas",
            split,
            len(self.pairs),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_path, altered_path = self.pairs[idx]

        # Load as grayscale
        real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
        altered_img = cv2.imread(altered_path, cv2.IMREAD_GRAYSCALE)

        if real_img is None:
            raise RuntimeError(f"No se pudo leer: {real_path}")
        if altered_img is None:
            raise RuntimeError(f"No se pudo leer: {altered_path}")

        # Resize to fixed size
        real_img = cv2.resize(real_img, (self.input_size, self.input_size))
        altered_img = cv2.resize(altered_img, (self.input_size, self.input_size))

        # Normalize to [0, 1] and add channel dimension
        real_tensor = torch.from_numpy(real_img.astype(np.float32) / 255.0).unsqueeze(0)
        altered_tensor = torch.from_numpy(
            altered_img.astype(np.float32) / 255.0
        ).unsqueeze(0)

        return real_tensor, altered_tensor


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CNNAutoencoder(nn.Module):
    """Lightweight 6-layer CNN autoencoder as comparison baseline.

    Architecture:
        Encoder: Conv3x3(1→32) → Conv3x3(32→64) → Conv3x3(64→128) (stride 2 each)
        Decoder: ConvTranspose(128→64) → ConvTranspose(64→32) → Conv3x3(32→1)
        Each Conv has BatchNorm + ReLU; final layer has TanH to output in [-1,1].
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()

        # Encoder
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.enc3 = self._block(64, 128)

        # Decoder
        self.dec3 = self._up_block(128, 64)
        self.dec2 = self._up_block(64, 32)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def _block(
        in_c: int, out_c: int, stride: int = 2
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up_block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_c, out_c, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)      # 1 -> 32,  256x256
        e2 = self.enc2(e1)     # 32 -> 64, 128x128
        e3 = self.enc3(e2)     # 64 -> 128, 64x64

        # Decoder
        d3 = self.dec3(e3)     # 128 -> 64, 128x128
        d2 = self.dec2(d3)     # 64 -> 32,  256x256
        d1 = self.dec1(d2)     # 32 -> 1,   512x512
        return d1


def create_unet(in_channels: int = 1, classes: int = 1) -> nn.Module:
    """Create a U-Net with MobileNetV2 encoder using SMP.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        classes: Number of output classes (1 for regression).

    Returns:
        U-Net model in evaluation mode.
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=classes,
    )
    return model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class PerceptualLoss(nn.Module):
    """Combined perceptual loss: L1 + SSIM.

    Uses kornia's SSIM for structural similarity and L1 for pixel accuracy.
    Per RESEARCH.md "Don't Hand-Roll": uses kornia instead of custom loss.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        """Initialize.

        Args:
            alpha: Weight for L1 loss.
            beta: Weight for SSIM loss (1 - SSIM).
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        import kornia.losses as K

        l1_loss = F.l1_loss(pred, target)
        # SSIM loss = 1 - SSIM (higher SSIM = lower loss)
        ssim_loss = 1.0 - K.ssim(
            pred, target, window_size=11, reduction="mean"
        )
        return self.alpha * l1_loss + self.beta * ssim_loss


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_psnr(
    pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target)
    if mse < 1e-10:
        return 100.0
    psnr = 20.0 * torch.log10(max_val / torch.sqrt(mse))
    return float(psnr.item())


def compute_ssim_score(
    pred: torch.Tensor, target: torch.Tensor
) -> float:
    """Compute SSIM using kornia."""
    import kornia.losses as K

    with torch.no_grad():
        ssim_val = K.ssim(pred, target, window_size=11, reduction="mean")
    return float(ssim_val.item())


def compute_minutiae_recovery_rate(
    enhanced: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Compute minutiae recovery rate using the SkeletonMinutiaeExtractor.

    Extracts minutiae from both the enhanced and ground-truth images,
    then computes recovery as the ratio of matched minutiae density.

    Args:
        enhanced: Enhanced output image (uint8, 0-255).
        ground_truth: Real (clean) fingerprint image (uint8, 0-255).

    Returns:
        Recovery rate in [0, 1] where 1.0 means perfect recovery.
    """
    try:
        from src.processing.extractor import SkeletonMinutiaeExtractor
    except ImportError as exc:
        logger.warning(
            "No se pudo importar SkeletonMinutiaeExtractor: %s", exc
        )
        return 0.0

    extractor = SkeletonMinutiaeExtractor()

    try:
        gt_minutiae = extractor.extract(ground_truth)
        enhanced_minutiae = extractor.extract(enhanced)
    except Exception as exc:
        logger.warning("Error en extracción de minucias: %s", exc)
        return 0.0

    if len(gt_minutiae) == 0:
        # No ground truth minutiae — cannot compute recovery
        return 0.0 if len(enhanced_minutiae) == 0 else 0.5

    # Recovery rate: fraction of ground truth minutiae recovered
    # Use the smaller count as conservative estimate
    recovery = min(len(enhanced_minutiae), len(gt_minutiae)) / max(
        len(gt_minutiae), 1
    )
    return float(min(recovery, 1.0))


def _tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized [0,1] tensor to uint8 numpy array."""
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1) * 255
    return arr.astype(np.uint8)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    desc: str = "Training",
) -> float:
    """Train the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        desc: Progress bar description.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for real_imgs, altered_imgs in tqdm(dataloader, desc=desc, leave=False):
        real_imgs = real_imgs.to(device)
        altered_imgs = altered_imgs.to(device)

        optimizer.zero_grad()
        output = model(altered_imgs)
        loss = criterion(output, real_imgs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Validate the model.

    Args:
        model: The PyTorch model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device.

    Returns:
        Tuple of (avg_loss, avg_psnr, avg_ssim, avg_minutiae_recovery).
    """
    model.eval()
    total_loss = 0.0
    psnr_list: list[float] = []
    ssim_list: list[float] = []
    recovery_list: list[float] = []
    num_batches = len(dataloader)

    for real_imgs, altered_imgs in tqdm(
        dataloader, desc="Validation", leave=False
    ):
        real_imgs = real_imgs.to(device)
        altered_imgs = altered_imgs.to(device)

        output = model(altered_imgs)
        loss = criterion(output, real_imgs)
        total_loss += loss.item()

        # Metrics on each sample
        for i in range(real_imgs.size(0)):
            psnr_list.append(compute_psnr(output[i], real_imgs[i]))
            ssim_list.append(compute_ssim_score(output[i], real_imgs[i]))

            # Minutiae recovery on CPU numpy
            enhanced_np = _tensor_to_uint8(output[i].squeeze(0))
            gt_np = _tensor_to_uint8(real_imgs[i].squeeze(0))
            recovery_list.append(
                compute_minutiae_recovery_rate(enhanced_np, gt_np)
            )

    avg_loss = total_loss / max(num_batches, 1)
    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    avg_recovery = float(np.mean(recovery_list)) if recovery_list else 0.0

    return avg_loss, avg_psnr, avg_ssim, avg_recovery


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 1, INPUT_SIZE, INPUT_SIZE),
    device: torch.device = DEVICE,
) -> Optional[Path]:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained PyTorch model (evaluation mode).
        output_path: Path to save the ONNX file.
        input_shape: Dummy input shape.
        device: Device.

    Returns:
        Path to the exported ONNX file, or None on failure.
    """
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=18,
        )
        logger.info("Modelo exportado a ONNX: %s", output_path)
        return output_path
    except Exception as exc:
        logger.error("Error exportando a ONNX: %s", exc)
        return None


def validate_onnx(onnx_path: Path) -> bool:
    """Validate an ONNX model loads correctly with onnxruntime.

    Args:
        onnx_path: Path to the ONNX file.

    Returns:
        True if the model loads and runs inference successfully.
    """
    try:
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        session = ort.InferenceSession(str(onnx_path), providers=providers)
        input_name = session.get_inputs()[0].name
        dummy = np.random.randn(1, 1, INPUT_SIZE, INPUT_SIZE).astype(
            np.float32
        )
        outputs = session.run(None, {input_name: dummy})
        logger.info(
            "ONNX validation OK: %s -> output shape %s",
            onnx_path.name,
            outputs[0].shape,
        )
        return True
    except Exception as exc:
        logger.error("ONNX validation failed for %s: %s", onnx_path, exc)
        return False


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------


def train_architecture(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TrainingHistory:
    """Train a model and return its history.

    Args:
        name: Architecture name for logging.
        model: The PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_epochs: Number of epochs to train.
        device: Device to train on.
        lr: Learning rate.
        weight_decay: AdamW weight decay.

    Returns:
        TrainingHistory with losses and best metrics.
    """
    logger.info("=" * 60)
    logger.info("Entrenando: %s", name)
    logger.info("=" * 60)

    model = model.to(device)
    criterion = PerceptualLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = TrainingHistory(name=name)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            desc=f"{name} Epoch {epoch}/{num_epochs}",
        )

        val_loss, val_psnr, val_ssim, val_recovery = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history.train_losses.append(train_loss)
        history.val_psnrs.append(val_psnr)
        history.val_ssims.append(val_ssim)
        history.val_recoveries.append(val_recovery)

        epoch_time = time.time() - epoch_start

        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "PSNR: %.2f | SSIM: %.4f | Recovery: %.2f%% | "
            "Time: %.1fs | LR: %.2e",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            val_psnr,
            val_ssim,
            val_recovery * 100.0,
            epoch_time,
            scheduler.get_last_lr()[0],
        )

        # Track best metrics
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_metrics.psnr = val_psnr
            history.best_metrics.ssim = val_ssim
            history.best_metrics.minutiae_recovery = val_recovery

    # Measure inference speed and model size
    history.best_metrics = _measure_model_perf(
        history.best_metrics, model, device
    )

    return history


def _measure_model_perf(
    metrics: SpikeMetrics, model: nn.Module, device: torch.device
) -> SpikeMetrics:
    """Measure inference speed, model size, and GPU memory."""
    model.eval()
    dummy = torch.randn(1, 1, INPUT_SIZE, INPUT_SIZE, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # Timed inference (100 runs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metrics.inference_ms = (elapsed / 100) * 1000  # ms per image

    # Model size (estimated from parameters)
    param_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    metrics.model_size_mb = param_size / (1024 * 1024)

    # GPU memory
    if device.type == "cuda":
        metrics.gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (
            1024 * 1024
        )
        torch.cuda.reset_peak_memory_stats()

    return metrics


def print_comparison_table(histories: list[TrainingHistory]) -> None:
    """Print a formatted comparison table of all trained architectures."""
    logger.info("")
    logger.info("=" * 90)
    logger.info("COMPARATIVA DE ARQUITECTURAS - RESULTADOS FINALES")
    logger.info("=" * 90)

    header = f"{'Arquitectura':<20} {'PSNR':<10} {'SSIM':<10} {'Recuperación':<15} {'Inferencia':<15} {'Modelo':<12} {'GPU':<10}"
    sep = "-" * 90

    logger.info(header)
    logger.info(sep)

    for h in histories:
        m = h.best_metrics
        row = (
            f"{h.name:<20} "
            f"{m.psnr:<8.2f}  "
            f"{m.ssim:<8.4f}  "
            f"{m.minutiae_recovery * 100:<8.2f}%     "
            f"{m.inference_ms:<8.1f}ms    "
            f"{m.model_size_mb:<8.1f}MB  "
            f"{m.gpu_memory_mb:<8.0f}MB"
        )
        logger.info(row)

    logger.info(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Spike: U-Net evaluation for fingerprint enhancement",
    )
    parser.add_argument(
        "--dataset-path",
        default=SOCOFING_DEFAULT,
        help="Path to SOCOFing dataset root (default: data/SOCOFing)",
    )
    parser.add_argument(
        "--altered-level",
        default="Altered-Easy",
        choices=["Altered-Easy", "Altered-Medium", "Altered-Hard"],
        help="Altered difficulty level for training pairs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs per architecture",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 100 samples, 5 epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODELS_DIR),
        help="Output directory for ONNX models",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Main entry point for the enhancement spike."""
    args = parse_args(argv)

    logger.info("=" * 70)
    logger.info("SPIKE: Evaluación de U-Net para mejora de huellas")
    logger.info("=" * 70)
    logger.info("Dispositivo: %s", DEVICE)
    logger.info("Dataset: %s", args.dataset_path)
    logger.info("Nivel alterado: %s", args.altered_level)
    logger.info("Épocas: %d", args.epochs)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Modo rápido: %s", args.quick)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- Cargando dataset SOCOFing ---")

    train_dataset = SOCOFingDataset(
        dataset_path=args.dataset_path,
        altered_level=args.altered_level,
        split="train",
        quick=args.quick,
    )
    val_dataset = SOCOFingDataset(
        dataset_path=args.dataset_path,
        altered_level=args.altered_level,
        split="val",
        quick=args.quick,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    actual_epochs = min(args.epochs, 5) if args.quick else args.epochs

    # -----------------------------------------------------------------------
    # U-Net
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- Creando modelo U-Net (SMP MobileNetV2) ---")
    unet_model = create_unet(in_channels=1, classes=1)

    unet_history = train_architecture(
        name="U-Net (MobileNetV2)",
        model=unet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=actual_epochs,
        device=DEVICE,
        lr=args.lr,
    )

    # Export U-Net to ONNX
    unet_onnx = Path(args.output_dir) / "spike_unet.onnx"
    if export_to_onnx(unet_model, unet_onnx):
        validate_onnx(unet_onnx)

    # -----------------------------------------------------------------------
    # CNN Autoencoder
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("--- Creando modelo CNN Autoencoder ---")
    cnn_model = CNNAutoencoder(in_channels=1)

    cnn_history = train_architecture(
        name="CNN-Autoencoder",
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=actual_epochs,
        device=DEVICE,
        lr=args.lr,
    )

    # Export CNN to ONNX
    cnn_onnx = Path(args.output_dir) / "spike_cnn.onnx"
    if export_to_onnx(cnn_model, cnn_onnx):
        validate_onnx(cnn_onnx)

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    print_comparison_table([unet_history, cnn_history])

    # -----------------------------------------------------------------------
    # Save results JSON for findings document
    # -----------------------------------------------------------------------
    results = {
        "device": str(DEVICE),
        "dataset": args.dataset_path,
        "altered_level": args.altered_level,
        "epochs": actual_epochs,
        "quick": args.quick,
        "architectures": [],
    }

    for h in [unet_history, cnn_history]:
        m = h.best_metrics
        results["architectures"].append(
            {
                "name": h.name,
                "best_psnr": m.psnr,
                "best_ssim": m.ssim,
                "minutiae_recovery": m.minutiae_recovery,
                "inference_ms": m.inference_ms,
                "model_size_mb": m.model_size_mb,
                "gpu_memory_mb": m.gpu_memory_mb,
                "train_losses": h.train_losses,
                "val_psnrs": h.val_psnrs,
                "val_ssims": h.val_ssims,
                "val_recoveries": h.val_recoveries,
            }
        )

    results_path = Path(args.output_dir) / "spike_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Resultados guardados en: %s", results_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SPIKE COMPLETADO")
    logger.info("=" * 70)

    # Determine recommendation
    unet_psnr = unet_history.best_metrics.psnr
    cnn_psnr = cnn_history.best_metrics.psnr
    if unet_psnr > cnn_psnr:
        logger.info(
            "RECOMENDACIÓN: U-Net (PSNR %.2f vs CNN %.2f)",
            unet_psnr,
            cnn_psnr,
        )
    else:
        logger.info(
            "RECOMENDACIÓN: CNN Autoencoder (PSNR %.2f vs U-Net %.2f)",
            cnn_psnr,
            unet_psnr,
        )


if __name__ == "__main__":
    main()

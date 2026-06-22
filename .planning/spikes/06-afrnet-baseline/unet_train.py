"""U-Net for latent fingerprint enhancement.

Trains a small encoder-decoder network that maps degraded fingerprints
(Altered-Hard: z-cut, obliteration, central rotation) to their clean
counterparts (Real).

Inference: pass the enhanced image through the existing best_model.pt
embedding network for improved verification performance on latent-like
data.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    image_size = 224

    # Training
    epochs = 30
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5

    # U-Net
    base_channels = 32
    depth = 4

    # Loss weights
    lambda_l1 = 1.0
    lambda_perceptual = 0.1

    # Paths
    checkpoint_path = SPIKE06 / "unet_best.pt"
    seed = 42
    num_workers = 4


cfg = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset: pair Altered-Hard with Real
# ---------------------------------------------------------------------------

def build_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Build (altered_path, real_path) pairs from Altered-Hard."""
    real_dir = root / "Real"
    altered_dir = root / "Altered" / "Altered-Hard"

    real_index: dict[str, Path] = {}
    for p in sorted(real_dir.rglob("*.BMP")):
        real_index[p.stem] = p  # e.g. "100__M_Left_index_finger"

    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for p in sorted(altered_dir.rglob("*.BMP")):
        stem = p.stem
        for suffix in ["_CR", "_Obl", "_Zcut"]:
            if stem.endswith(suffix):
                base = stem[: -len(suffix)]
                if base in real_index:
                    pairs.append((p, real_index[base]))
                else:
                    missing += 1
                break
    print(f"  Pairs: {len(pairs)} ({missing} altered without Real match)")
    return pairs


class LatentPairDataset(Dataset):
    """Altered-Hard -> Real pairs for U-Net training."""

    def __init__(self, pairs: list[tuple[Path, Path]],
                 image_size: int = 224,
                 augment: bool = False) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        altered_path, real_path = self.pairs[idx]
        alt_img = cv2.imread(str(altered_path), cv2.IMREAD_GRAYSCALE)
        real_img = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE)
        if alt_img is None or real_img is None:
            raise FileNotFoundError(f"Could not read pair: {altered_path} or {real_path}")

        alt_img = self._preprocess(alt_img)
        real_img = self._preprocess(real_img)

        if self.augment:
            alt_img, real_img = self._augment(alt_img, real_img)

        alt_t = torch.from_numpy(alt_img).float().unsqueeze(0) / 255.0
        real_t = torch.from_numpy(real_img).float().unsqueeze(0) / 255.0
        # Normalize to [-1, 1] to match embedding model input
        alt_t = (alt_t - 0.5) / 0.5
        real_t = (real_t - 0.5) / 0.5
        return alt_t, real_t

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        max_side = max(h, w)
        top = (max_side - h) // 2
        left = (max_side - w) // 2
        padded = cv2.copyMakeBorder(
            img, top, max_side - h - top,
            left, max_side - w - left,
            cv2.BORDER_CONSTANT, value=0,
        )
        return cv2.resize(padded, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_LINEAR)

    def _augment(self, alt: np.ndarray, real: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the same geometric transform to both images."""
        if np.random.random() < 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = alt.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            alt = cv2.warpAffine(alt, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            real = cv2.warpAffine(real, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        return alt, real


# ---------------------------------------------------------------------------
# U-Net architecture
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    """Small U-Net for grayscale image enhancement.

    Total params: ~1.1M (very small, fast to train).
    """

    def __init__(self, in_ch: int = 1, out_ch: int = 1,
                 base: int = 32, depth: int = 4) -> None:
        super().__init__()
        # depth downs: produces chs[0..depth-1] skip features,
        # bottleneck doubles the last one
        chs = [base * (2 ** i) for i in range(depth)]
        # chs (depth=4): [32, 64, 128, 256]

        # Encoder: chs[i] -> chs[i+1] for i in 0..depth-2,
        # plus a final downsample? No, we have exactly `depth` downs
        # producing `depth` skip features, with last down output = chs[depth-1]
        self.downs = nn.ModuleList()
        for i in range(depth):
            in_c = in_ch if i == 0 else chs[i - 1]
            self.downs.append(DoubleConv(in_c, chs[i]))

        # Bottleneck: chs[depth-1] -> chs[depth-1] * 2
        bottleneck_ch = chs[-1] * 2
        self.bottleneck = DoubleConv(chs[-1], bottleneck_ch)

        # Decoder: at each level i (depth-1 down to 0):
        #   up input: prev_out (start at bottleneck_ch)
        #   up output: chs[i] (matches skip)
        #   after cat: chs[i] + chs[i] = 2*chs[i], then DoubleConv back to chs[i]
        self.ups = nn.ModuleList()
        cur = bottleneck_ch
        for i in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(cur, chs[i], 2, stride=2))
            self.ups.append(DoubleConv(chs[i] * 2, chs[i]))
            cur = chs[i]

        self.out_conv = nn.Conv2d(base, out_ch, 1)
        self.tanh = nn.Tanh()  # output in [-1, 1] to match embedding input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = torch.cat([x, skips[i // 2]], dim=1)
            x = self.ups[i + 1](x)

        return self.tanh(self.out_conv(x))


# ---------------------------------------------------------------------------
# Loss: L1 + perceptual (VGG features)
# ---------------------------------------------------------------------------

class VGGPerceptual(nn.Module):
    """Use a small VGG-style feature extractor for perceptual loss."""

    def __init__(self) -> None:
        super().__init__()
        # We use first 4 layers of VGG11 (very small)
        try:
            from torchvision.models import vgg11, VGG11_Weights
            v = vgg11(weights=VGG11_Weights.DEFAULT)
            self.features = nn.Sequential(*list(v.features.children())[:4])
            for p in self.features.parameters():
                p.requires_grad = False
        except Exception as e:
            print(f"Warning: VGG load failed ({e}), using random init")
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
            for p in self.features.parameters():
                p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VGG expects 3 channels; replicate
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)


class CombinedLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.vgg = VGGPerceptual().to(DEVICE)
        self.vgg.eval()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        loss_l1 = self.l1(pred, target)
        # Perceptual on [0, 1] range
        pred_01 = pred * 0.5 + 0.5
        target_01 = target * 0.5 + 0.5
        feat_pred = self.vgg(pred_01)
        feat_target = self.vgg(target_01)
        loss_perc = self.l1(feat_pred, feat_target)
        loss = (cfg.lambda_l1 * loss_l1 +
                cfg.lambda_perceptual * loss_perc)
        return {
            "loss": loss,
            "l1": loss_l1.item(),
            "perceptual": loss_perc.item(),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_l1 = 0.0
    total_perc = 0.0
    total_loss = 0.0
    n = 0
    for alt, real in loader:
        alt = alt.to(DEVICE, non_blocking=True)
        real = real.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=DEVICE):
            pred = model(alt)
            losses = criterion(pred, real)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        bs = alt.size(0)
        total_loss += losses["loss"].item() * bs
        total_l1 += losses["l1"] * bs
        total_perc += losses["perceptual"] * bs
        n += bs
    return {"loss": total_loss / n, "l1": total_l1 / n, "perc": total_perc / n}


@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_l1 = 0.0
    total_perc = 0.0
    total_loss = 0.0
    n = 0
    for alt, real in loader:
        alt = alt.to(DEVICE, non_blocking=True)
        real = real.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE):
            pred = model(alt)
            losses = criterion(pred, real)
        bs = alt.size(0)
        total_loss += losses["loss"].item() * bs
        total_l1 += losses["l1"] * bs
        total_perc += losses["perceptual"] * bs
        n += bs
    return {"loss": total_loss / n, "l1": total_l1 / n, "perc": total_perc / n}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    print(f"Device: {DEVICE}")
    print(f"Config: epochs={cfg.epochs}, batch={cfg.batch_size}, lr={cfg.lr}")
    print(f"  U-Net: base={cfg.base_channels}, depth={cfg.depth}")
    print(f"  Losses: L1={cfg.lambda_l1}, perceptual={cfg.lambda_perceptual}")
    print()

    # Build pairs and split
    print("Building Altered-Hard -> Real pairs...")
    pairs = build_pairs(cfg.root)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in perm]
    n_val = max(500, int(0.05 * len(pairs)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    train_ds = LatentPairDataset(train_pairs, cfg.image_size, augment=True)
    val_ds = LatentPairDataset(val_pairs, cfg.image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    print("\nBuilding U-Net...")
    model = UNet(in_ch=1, out_ch=1, base=cfg.base_channels, depth=cfg.depth).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    scaler = torch.GradScaler()
    criterion = CombinedLoss().to(DEVICE)

    best_loss = float("inf")
    best_epoch = -1
    t_start = time.time()

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, criterion)
        val_metrics = val_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"E{epoch + 1:2d}/{cfg.epochs} | "
            f"train L={train_metrics['loss']:.4f} "
            f"L1={train_metrics['l1']:.4f} "
            f"perc={train_metrics['perc']:.4f} | "
            f"val L={val_metrics['loss']:.4f} "
            f"L1={val_metrics['l1']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "config": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(cfg).items()},
            }, cfg.checkpoint_path)
            print(f"  -> Best saved (val L1: {val_metrics['l1']:.4f})")

    total = time.time() - t_start
    print(f"\nDone in {total / 60:.1f} min. Best epoch: {best_epoch + 1}, "
          f"val L1: {best_loss:.4f}")


if __name__ == "__main__":
    main()

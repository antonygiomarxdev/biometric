"""Train AFR-Net style fingerprint embedding with ArcFace loss."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))

from dataset import SOCOFingDataset, split_subjects
from model import AFRNetFingerprint, count_parameters


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    subdir = "Real"
    train_subjects = 540
    image_size = 224

    # Training
    epochs = 30
    batch_size = 32
    accum_steps = 2  # effective batch = 64
    lr_backbone = 1e-4
    lr_head = 1e-3
    weight_decay = 0.01
    warmup_epochs = 2

    # ArcFace
    s = 30.0
    m = 0.5

    # Data
    num_workers = 4

    # Paths
    checkpoint_dir = SPIKE06 / "checkpoints"
    best_model_path = SPIKE06 / "best_model.pt"

    seed = 42


cfg = Config()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Verification evaluation: TAR@FAR + EER + ROC AUC
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_verification_metrics(
    embs: torch.Tensor,
    subject_ids: torch.Tensor,
    n_impostor_pairs: int = 50000,
) -> dict[str, float]:
    """Compute TAR@FAR, EER, and ROC AUC on all genuine + sampled impostor pairs.

    Genuine pairs: same subject, different images.
    Impostor pairs: different subjects, sampled randomly.
    """
    n = embs.size(0)
    embs = F.normalize(embs, p=2, dim=1)

    # Group indices by subject
    subj_to_idx: dict[int, list[int]] = {}
    for i, s in enumerate(subject_ids.tolist()):
        subj_to_idx.setdefault(s, []).append(i)

    # Generate all genuine pairs (Capped to keep memory in check)
    genuine_scores: list[float] = []
    for s, idx_list in subj_to_idx.items():
        if len(idx_list) < 2:
            continue
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                sim = F.cosine_similarity(
                    embs[idx_list[i]:idx_list[i] + 1],
                    embs[idx_list[j]:idx_list[j] + 1],
                ).item()
                genuine_scores.append(sim)

    # Sample impostor pairs
    impostor_scores: list[float] = []
    rng = np.random.default_rng(42)
    all_subjects = list(subj_to_idx.keys())
    while len(impostor_scores) < n_impostor_pairs:
        s1, s2 = rng.choice(all_subjects, size=2, replace=False)
        i1 = rng.choice(subj_to_idx[s1])
        i2 = rng.choice(subj_to_idx[s2])
        sim = F.cosine_similarity(
            embs[i1:i1 + 1], embs[i2:i2 + 1],
        ).item()
        impostor_scores.append(sim)

    if not genuine_scores:
        return {"tar_far_001": 0.0, "tar_far_0001": 0.0,
                "eer": 1.0, "auc": 0.0, "n_genuine": 0, "n_impostor": 0}

    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    y_score = np.array(genuine_scores + impostor_scores)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))

    # EER: point where fpr == 1 - tpr
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    # TAR@FAR
    def tar_at_far(target_far: float) -> float:
        # Find smallest threshold where FPR <= target_far
        idx = np.searchsorted(-fpr, -target_far)  # fpr sorted ascending
        if idx >= len(fpr):
            idx = len(fpr) - 1
        # Walk back to find first tpr at or above target_far
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= target_far:
                return float(tpr[i])
        return float(tpr[0])

    return {
        "tar_far_01": tar_at_far(0.01),
        "tar_far_001": tar_at_far(0.001),
        "tar_far_0001": tar_at_far(0.0001),
        "eer": eer,
        "auc": roc_auc,
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores),
    }


@torch.no_grad()
def compute_rank_n(embs: torch.Tensor,
                   subject_ids: torch.Tensor,
                   top_k: int = 5) -> dict[str, float]:
    """Closed-set identification Rank-N."""
    n = embs.size(0)
    embs = F.normalize(embs, p=2, dim=1)
    # Compute full cosine similarity matrix
    sims = embs @ embs.T  # (N, N)

    rank1 = 0
    rank5 = 0
    for i in range(n):
        scores = sims[i].clone()
        scores[i] = -2  # exclude self
        order = scores.argsort(descending=True)
        top = order[:top_k]
        if subject_ids[top[0]].item() == subject_ids[i].item():
            rank1 += 1
        if subject_ids[i].item() in subject_ids[top].tolist():
            rank5 += 1
    return {"rank1": rank1 / n, "rank5": rank5 / n}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, epoch, scheduler):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE):
            out = model(images, targets)
            loss = out["loss"] / cfg.accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        bs = images.size(0)
        total_loss += out["loss"].item() * bs
        total_acc += out["acc"].item() * bs
        total += bs

    return {"loss": total_loss / total, "acc": total_acc / total}


@torch.no_grad()
def val_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0
    all_embs: list[torch.Tensor] = []
    all_sids: list[int] = []

    for images, targets in loader:
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type=DEVICE):
            out = model(images, targets)
            all_embs.append(out["embedding"].float().cpu())
            all_sids.extend(targets.cpu().tolist())
            bs = images.size(0)
            total_loss += out["loss"].item() * bs
            total_acc += out["acc"].item() * bs
            total += bs

    embs = torch.cat(all_embs)
    sids = torch.tensor(all_sids)

    metrics = {
        "loss": total_loss / total,
        "acc": total_acc / total,
    }
    metrics.update(compute_rank_n(embs, sids))
    metrics.update(compute_verification_metrics(embs, sids))
    return metrics, embs, sids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(cfg.seed)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Config: epochs={cfg.epochs}, batch={cfg.batch_size}, "
          f"accum={cfg.accum_steps}, eff_batch={cfg.batch_size * cfg.accum_steps}")
    print(f"  LR backbone={cfg.lr_backbone}, LR head={cfg.lr_head}, "
          f"s={cfg.s}, m={cfg.m}")
    print()

    # Dataset
    print("Loading SOCOFing...")
    ds = SOCOFingDataset(cfg.root, cfg.subdir)
    train_ds, val_ds = split_subjects(ds, cfg.train_subjects)
    train_ds.augment = True
    print(f"  Train: {len(train_ds)} images, {train_ds.num_classes} subjects")
    print(f"  Val:   {len(val_ds)} images, {val_ds.num_classes} subjects")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    print("\nBuilding model...")
    model = AFRNetFingerprint(
        num_classes=train_ds.num_classes,
        embedding_dim=512,
        s=cfg.s, m=cfg.m,
    ).to(DEVICE)
    print(f"  Params: {count_parameters(model)}")

    # Optimizer with different LR for backbone vs head
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "head" in name or "fusion" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

    # Warmup + cosine schedule
    steps_per_epoch = len(train_loader) // cfg.accum_steps
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.GradScaler()

    # Train
    print("\nTraining...")
    best_tar = -1.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    scaler, epoch, scheduler)
        val_metrics, embs, sids = val_epoch(model, val_loader)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"E{epoch + 1:2d}/{cfg.epochs} | "
            f"train L={train_metrics['loss']:.3f} A={train_metrics['acc']:.3f} | "
            f"val L={val_metrics['loss']:.3f} A={val_metrics['acc']:.3f} | "
            f"rank1={val_metrics['rank1']:.3f} rank5={val_metrics['rank5']:.3f} | "
            f"AUC={val_metrics['auc']:.3f} "
            f"TAR01={val_metrics['tar_far_01']:.3f} "
            f"TAR001={val_metrics['tar_far_001']:.3f} | "
            f"EER={val_metrics['eer']:.3f} | "
            f"lr={lr_now:.2e} {elapsed:.1f}s"
        )

        if val_metrics["tar_far_01"] > best_tar:
            best_tar = val_metrics["tar_far_01"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "config": vars(cfg),
            }, cfg.best_model_path)
            print(f"  → Best saved (TAR@FAR=0.01: {best_tar:.3f})")

    total = time.time() - t_start
    print(f"\nDone in {total / 60:.1f} min.")
    print(f"Best epoch: {best_epoch + 1}, TAR@FAR=0.01: {best_tar:.3f}")


if __name__ == "__main__":
    main()

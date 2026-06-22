"""Fine-tune AFR-Net on mixed Real + Altered-Hard for latent robustness.

Strategy:
- Load best checkpoint from Real-only training
- Mix Real (6K) + Altered-Hard (14K) for training (20K total)
- Same 540/60 subject split as original
- Curriculum augmentation: light -> heavy over epochs
- Low LR fine-tuning: 1e-5 backbone, 5e-5 head
- 20 epochs
- Evaluate on all 3 protocols (Easy/Medium/Hard) at end
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset, DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))

from model import AFRNetFingerprint, count_parameters


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    image_size = 224

    # Sources: Real (clean) + Altered-Hard (degraded)
    subdirs: list[str] = ["Real", "Altered/Altered-Hard"]

    epochs = 20
    batch_size = 32
    accum_steps = 2
    lr_backbone = 1e-5
    lr_head = 5e-5
    weight_decay = 0.01
    warmup_epochs = 2

    s = 30.0
    m = 0.5
    embedding_dim = 512
    num_workers = 4

    # Load this checkpoint and continue training
    base_checkpoint = SPIKE06 / "best_model.pt"
    best_model_path = SPIKE06 / "best_model_latent.pt"
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
# Combined dataset: Real + Altered-Hard
# ---------------------------------------------------------------------------

def get_finger_id(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split("__", 1)
    subject = parts[0]
    finger = parts[1] if len(parts) > 1 else ""
    for suffix in ["_CR", "_Obl", "_Zcut", "_OBL", "_CR.BMP",
                   "_BTH", "_BTR", "_BR", "_BL"]:
        finger = finger.replace(suffix, "")
    return f"{subject}__{finger}"


class MixedFingerprintDataset(Dataset):
    """Mixed Real + Altered dataset with curriculum augmentation."""

    def __init__(self, root: Path, subdirs: list[str],
                 subject_set: set[str] | None = None,
                 augment_level: float = 0.0) -> None:
        self.root = root
        self.augment_level = augment_level

        images: list[tuple[str, str]] = []
        for subdir in subdirs:
            data_dir = root / subdir
            paths = sorted(data_dir.rglob("*.BMP"))
            for p in paths:
                subj = p.stem.split("_")[0]
                if subject_set is None or subj in subject_set:
                    images.append((str(p), subj))

        self.subjects = sorted(set(s for _, s in images))
        self.subject_to_id = {s: i for i, s in enumerate(self.subjects)}

        self.images: list[tuple[str, int]] = [
            (p, self.subject_to_id[s]) for p, s in images
        ]
        self.num_classes = len(self.subjects)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, sid = self.images[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read: {path}")
        img = self._preprocess(img)
        if self.augment_level > 0:
            img = self._curriculum_augment(img)
        tensor = torch.from_numpy(img).float()
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor.unsqueeze(0), sid

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
        return cv2.resize(padded, (cfg.image_size, cfg.image_size),
                          interpolation=cv2.INTER_LINEAR)

    def _curriculum_augment(self, img: np.ndarray) -> np.ndarray:
        """Augment intensity grows with self.augment_level [0..1]."""
        level = self.augment_level

        # Rotation: up to ±180 * level
        if np.random.random() < 0.5 * level:
            angle = np.random.uniform(-180, 180) * level
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

        # Elastic: only at higher levels
        if level > 0.5 and np.random.random() < 0.4:
            h, w = img.shape
            strength = level * 5
            dx = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (15, 15), 5,
            ) * strength
            dy = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (15, 15), 5,
            ) * strength
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

        # Occlusion: only at higher levels
        if level > 0.3 and np.random.random() < 0.3 * level:
            h, w = img.shape
            for _ in range(np.random.randint(1, max(2, int(3 * level)))):
                occ_h = int(h * np.random.uniform(0.1, 0.3 * level))
                occ_w = int(w * np.random.uniform(0.1, 0.3 * level))
                top = np.random.randint(0, max(1, h - occ_h))
                left = np.random.randint(0, max(1, w - occ_w))
                color = np.random.choice([0, 255, 128])
                img[top:top + occ_h, left:left + occ_w] = color

        # Photometric
        if np.random.random() < 0.4 * level:
            factor = np.random.uniform(0.6, 1.4)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if np.random.random() < 0.3 * level:
            shift = np.random.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        if np.random.random() < 0.3 * level:
            k = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img


def split_subjects(dataset: MixedFingerprintDataset,
                   train_count: int = 540
                   ) -> tuple[MixedFingerprintDataset, MixedFingerprintDataset]:
    all_subjects = sorted(set(s for _, s in dataset.images))
    val_count = len(all_subjects) - train_count
    if val_count <= 0:
        raise ValueError(f"Need >{train_count} subjects, have {len(all_subjects)}")

    train_set = set(all_subjects[:train_count])
    val_set = set(all_subjects[train_count:])

    def _filter(subj_set, augment_level):
        ds = MixedFingerprintDataset.__new__(MixedFingerprintDataset)
        ds.root = dataset.root
        ds.augment_level = augment_level
        ds.images = [(p, s) for p, s in dataset.images if s in subj_set]
        ds.subjects = sorted(set(s for _, s in ds.images))
        ds.subject_to_id = {s: i for i, s in enumerate(ds.subjects)}
        ds.num_classes = len(ds.subjects)
        # Remap to contiguous
        ds.images = [(p, ds.subject_to_id[s]) for p, s in ds.images]
        return ds

    return (_filter(train_set, dataset.augment_level),
            _filter(val_set, 0.0))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(embs: torch.Tensor,
                    subject_ids: torch.Tensor) -> dict[str, float]:
    """Compute Rank-N + TAR@FAR on the dataset embeddings."""
    n = embs.size(0)
    embs_n = F.normalize(embs, p=2, dim=1)

    subj_to_idx: dict[int, list[int]] = {}
    for i, s in enumerate(subject_ids.tolist()):
        subj_to_idx.setdefault(s, []).append(i)

    # Rank-1 / Rank-5
    sims = embs_n @ embs_n.T
    rank1, rank5 = 0, 0
    for i in range(n):
        scores = sims[i].clone()
        scores[i] = -2
        order = scores.argsort(descending=True)
        top = order[:5]
        if subject_ids[top[0]].item() == subject_ids[i].item():
            rank1 += 1
        if subject_ids[i].item() in subject_ids[top].tolist():
            rank5 += 1

    # Genuine pairs for TAR@FAR
    genuine_scores = []
    for s, idx_list in subj_to_idx.items():
        if len(idx_list) < 2:
            continue
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                sim = F.cosine_similarity(
                    embs_n[idx_list[i]:idx_list[i] + 1],
                    embs_n[idx_list[j]:idx_list[j] + 1],
                ).item()
                genuine_scores.append(sim)

    impostor_scores = []
    rng = np.random.default_rng(42)
    all_subjects = list(subj_to_idx.keys())
    n_impostor = min(50000, len(genuine_scores) * 10)
    while len(impostor_scores) < n_impostor:
        s1, s2 = rng.choice(all_subjects, size=2, replace=False)
        i1 = rng.choice(subj_to_idx[s1])
        i2 = rng.choice(subj_to_idx[s2])
        sim = F.cosine_similarity(
            embs_n[i1:i1 + 1], embs_n[i2:i2 + 1],
        ).item()
        impostor_scores.append(sim)

    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    y_score = np.array(genuine_scores + impostor_scores)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    def tar_at_far(target):
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= target:
                return tpr[i]
        return tpr[0]

    return {
        "rank1": rank1 / n,
        "rank5": rank5 / n,
        "auc": roc_auc,
        "eer": eer,
        "tar_far_01": tar_at_far(0.01),
        "tar_far_001": tar_at_far(0.001),
    }


# ---------------------------------------------------------------------------
# Training
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
    metrics = {"loss": total_loss / total, "acc": total_acc / total}
    metrics.update(compute_metrics(embs, sids))
    return metrics


# ---------------------------------------------------------------------------
# Full protocol evaluation (Real vs Altered-Easy/Medium/Hard)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_protocol(model, label: str,
                      gallery_ds, probe_ds,
                      n_pairs: int = 2000) -> dict:
    """Same-finger verification: gallery=Real, probe=Altered-{X}."""
    from dataset import SOCOFingDataset

    def get_emb(ds, max_n=6000):
        loader = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)
        embs, paths = [], []
        n = 0
        for img, _ in loader:
            if n >= max_n:
                break
            img = img.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(device_type=DEVICE):
                out = model(img)
            embs.append(out["embedding"].float().cpu())
            n += img.size(0)
        return torch.cat(embs), [p[0] for p in ds.images][:n]

    g_embs, g_paths = get_emb(gallery_ds, 6000)
    p_embs, p_paths = get_emb(probe_ds, 3000)

    g_fids = {p: get_finger_id(p) for p in g_paths}
    p_fids = {p: get_finger_id(p) for p in p_paths}
    g_by_fid: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(g_paths):
        g_by_fid[g_fids[p]].append(i)

    g_embs_n = F.normalize(g_embs, p=2, dim=1)
    p_embs_n = F.normalize(p_embs, p=2, dim=1)

    valid = [i for i, p in enumerate(p_paths) if p_fids[p] in g_by_fid]
    sims = p_embs_n[valid] @ g_embs_n.T
    sims_np = sims.numpy()

    genuine, impostor = [], []
    rng = np.random.default_rng(42)
    for pi_idx, pi in enumerate(valid):
        fid = p_fids[p_paths[pi]]
        for gi in g_by_fid[fid]:
            genuine.append(sims_np[pi_idx, gi])
        non_match = [i for i in range(len(g_paths)) if g_fids[g_paths[i]] != fid]
        if non_match:
            impostor.append(sims_np[pi_idx, rng.choice(non_match)])

    genuine = np.array(genuine)
    impostor = np.array(impostor)

    y_true = np.array([1] * len(genuine) + [0] * len(impostor))
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] +
           fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2

    def tar_at_far(target):
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= target:
                return tpr[i]
        return tpr[0]

    return {
        "label": label,
        "auc": roc_auc,
        "eer": eer,
        "tar_far_01": tar_at_far(0.01),
        "tar_far_001": tar_at_far(0.001),
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
        "genuine": genuine,
        "impostor": impostor,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_seed(cfg.seed)

    print(f"Device: {DEVICE}")
    print(f"Config: epochs={cfg.epochs}, batch={cfg.batch_size}, "
          f"accum={cfg.accum_steps}")
    print(f"  LR backbone={cfg.lr_backbone}, LR head={cfg.lr_head}")
    print(f"  Sources: {cfg.subdirs}")
    print()

    # Dataset with curriculum (level 0 = no aug, will increase)
    print("Loading mixed dataset...")
    ds = MixedFingerprintDataset(cfg.root, cfg.subdirs, augment_level=0.5)
    train_ds, val_ds = split_subjects(ds, 540)
    print(f"  Total unique subjects: {ds.num_classes}")
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

    # Model: load checkpoint, keep same num_classes
    print("\nLoading base checkpoint...")
    ckpt = torch.load(cfg.base_checkpoint, map_location="cpu", weights_only=False)
    base_num_classes = ckpt["model_state"]["head.weight"].shape[0]
    model = AFRNetFingerprint(
        num_classes=base_num_classes,
        embedding_dim=cfg.embedding_dim,
        s=cfg.s, m=cfg.m,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")
    print(f"  Params: {count_parameters(model)}")

    # Optimizer
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if "head" in name or "fusion" in name:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.lr_backbone},
        {"params": head_params, "lr": cfg.lr_head},
    ], weight_decay=cfg.weight_decay)

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

    # Baseline eval on full protocol before fine-tuning
    print("\n--- BASELINE EVALUATION (before fine-tuning) ---")
    from dataset import SOCOFingDataset as SOCOFingOriginal
    root = cfg.root
    real_ds = SOCOFingOriginal(root, "Real", augment=False)
    baseline_results = {}
    for level in ["Easy", "Medium", "Hard"]:
        probe_ds = SOCOFingOriginal(root, f"Altered/Altered-{level}", augment=False)
        result = evaluate_protocol(model, f"Real vs Altered-{level}", real_ds, probe_ds)
        baseline_results[level] = result
        print(f"  {result['label']}: AUC={result['auc']:.4f} "
              f"TAR@0.01={result['tar_far_01']:.4f} "
              f"EER={result['eer']:.4f}")

    # Train
    print("\nFine-tuning...")
    best_tar = -1.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(cfg.epochs):
        # Increase curriculum augmentation over epochs
        progress = epoch / cfg.epochs
        aug_level = min(1.0, 0.3 + progress * 0.7)
        train_ds.augment_level = aug_level

        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    scaler, epoch, scheduler)
        val_metrics = val_epoch(model, val_loader)
        elapsed = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"E{epoch + 1:2d}/{cfg.epochs} | "
            f"aug={aug_level:.2f} | "
            f"train L={train_metrics['loss']:.3f} A={train_metrics['acc']:.3f} | "
            f"val L={val_metrics['loss']:.3f} A={val_metrics['acc']:.3f} | "
            f"rank1={val_metrics['rank1']:.3f} | "
            f"AUC={val_metrics['auc']:.3f} "
            f"TAR01={val_metrics['tar_far_01']:.3f} | "
            f"lr={lr_now:.2e} {elapsed:.1f}s"
        )

        if val_metrics["tar_far_01"] > best_tar:
            best_tar = val_metrics["tar_far_01"]
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "metrics": val_metrics,
                "config": {k: str(v) if isinstance(v, Path) else v
                          for k, v in vars(cfg).items()},
            }, cfg.best_model_path)
            print(f"  -> Best saved (TAR@FAR=0.01: {best_tar:.3f})")

    total = time.time() - t_start
    print(f"\nFine-tuning done in {total / 60:.1f} min.")
    print(f"Best epoch: {best_epoch + 1}, TAR@FAR=0.01: {best_tar:.3f}")

    # Final evaluation on all 3 protocols
    print("\n--- FINETUNED EVALUATION (after fine-tuning) ---")
    ckpt = torch.load(cfg.best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    finetuned_results = {}
    for level in ["Easy", "Medium", "Hard"]:
        probe_ds = SOCOFingOriginal(root, f"Altered/Altered-{level}", augment=False)
        result = evaluate_protocol(model, f"Real vs Altered-{level}", real_ds, probe_ds)
        finetuned_results[level] = result

    # Comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON: Baseline vs Finetuned (on mixed Real + Altered-Hard)")
    print(f"{'='*70}")
    print(f"{'Protocol':<24s} {'AUC':>8s} {'EER':>8s} {'TAR@0.01':>10s} {'TAR@0.001':>10s}")
    print("-" * 60)
    for level in ["Easy", "Medium", "Hard"]:
        b = baseline_results[level]
        f = finetuned_results[level]
        print(f"{b['label']:<24s}")
        print(f"  Baseline:  {b['auc']:>8.4f} {b['eer']:>8.4f} "
              f"{b['tar_far_01']:>10.4f} {b['tar_far_001']:>10.4f}")
        print(f"  Finetuned: {f['auc']:>8.4f} {f['eer']:>8.4f} "
              f"{f['tar_far_01']:>10.4f} {f['tar_far_001']:>10.4f}")
        delta_tar = (f['tar_far_01'] - b['tar_far_01']) * 100
        print(f"  Delta:                TAR@0.01: {delta_tar:+.2f}pp")
        print()

    # Save results
    out = SPIKE06 / "finetune_results.npz"
    np.savez(out,
             **{f"baseline_{k}_{lvl}": baseline_results[lvl][k]
                for lvl in ["Easy", "Medium", "Hard"]
                for k in ["genuine", "impostor"]},
             **{f"finetuned_{k}_{lvl}": finetuned_results[lvl][k]
                for lvl in ["Easy", "Medium", "Hard"]
                for k in ["genuine", "impostor"]},
             )
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()

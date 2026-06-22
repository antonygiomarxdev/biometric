"""Final evaluation of the trained AFR-Net model with detailed metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))

from dataset import SOCOFingDataset, split_subjects
from model import AFRNetFingerprint
from train import compute_verification_metrics, compute_rank_n


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = SPIKE06 / "best_model.pt"

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"Best epoch: {ckpt['epoch'] + 1}")
    print(f"Saved metrics: {ckpt['metrics']}")

    # Reconstruct model
    model = AFRNetFingerprint(
        num_classes=540,
        embedding_dim=512,
        s=30.0, m=0.5,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Dataset
    print("\nLoading val set...")
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    ds = SOCOFingDataset(root, "Real")
    train_ds, val_ds = split_subjects(ds, 540)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2, pin_memory=True)
    print(f"Val: {len(val_ds)} images, {val_ds.num_classes} subjects")

    # Extract all val embeddings
    print("\nExtracting embeddings...")
    embs_list, sids_list, paths_list = [], [], []
    with torch.no_grad():
        for img, sid in val_loader:
            img = img.to(DEVICE)
            with torch.amp.autocast(device_type=DEVICE):
                out = model(img)
            embs_list.append(out["embedding"].float().cpu())
            sids_list.extend(sid.tolist())

    embs = torch.cat(embs_list)
    sids = torch.tensor(sids_list)
    print(f"  Embeddings: {embs.shape}")

    # Compute metrics
    print("\n" + "=" * 60)
    print("VERIFICATION METRICS (TAR@FAR, EER, AUC)")
    print("=" * 60)
    metrics = compute_verification_metrics(embs, sids, n_impostor_pairs=100000)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s}: {v:.4f}")
        else:
            print(f"  {k:18s}: {v}")

    print("\n" + "=" * 60)
    print("IDENTIFICATION METRICS (Rank-N)")
    print("=" * 60)
    rank = compute_rank_n(embs, sids, top_k=10)
    for k, v in rank.items():
        print(f"  {k:6s}: {v:.4f}")

    # Cosine distribution
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY DISTRIBUTION")
    print("=" * 60)
    embs_n = F.normalize(embs, p=2, dim=1)
    # Genuine pairs
    subj_to_idx: dict[int, list[int]] = {}
    for i, s in enumerate(sids.tolist()):
        subj_to_idx.setdefault(s, []).append(i)
    genuine = []
    for s, idx_list in subj_to_idx.items():
        if len(idx_list) < 2:
            continue
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                genuine.append(
                    F.cosine_similarity(
                        embs_n[idx_list[i]:idx_list[i] + 1],
                        embs_n[idx_list[j]:idx_list[j] + 1],
                    ).item()
                )
    # Sample impostor
    rng = np.random.default_rng(42)
    all_subjs = list(subj_to_idx.keys())
    impostor = []
    while len(impostor) < 5000:
        s1, s2 = rng.choice(all_subjs, size=2, replace=False)
        i1 = rng.choice(subj_to_idx[s1])
        i2 = rng.choice(subj_to_idx[s2])
        impostor.append(
            F.cosine_similarity(
                embs_n[i1:i1 + 1], embs_n[i2:i2 + 1],
            ).item()
        )

    g = np.array(genuine)
    i = np.array(impostor)
    print(f"  Genuine pairs:  N={len(g)}, mean={g.mean():.4f}, std={g.std():.4f}, "
          f"range=[{g.min():.4f}, {g.max():.4f}]")
    print(f"  Impostor pairs: N={len(i)}, mean={i.mean():.4f}, std={i.std():.4f}, "
          f"range=[{i.min():.4f}, {i.max():.4f}]")
    print(f"  Separation (mean diff): {g.mean() - i.mean():.4f}")
    print(f"  Threshold @ EER:        {(g.max() + i.min()) / 2:.4f}")

    # Comparison with reference
    print("\n" + "=" * 60)
    print("CONTEXT (for comparison)")
    print("=" * 60)
    print("  Bozorth3 (NIST, no enhancement) on SOCOFing:  "
          "TAR@FAR=0.01 ~ 30-50% (literature)")
    print("  Verifinger (commercial COTS):                  "
          "TAR@FAR=0.01 ~ 95%+")
    print("  AFR-Net paper (clean conditions):              "
          "TAR@FAR=0.01 ~ 90%+")
    print("  Our baseline (untrained ImageNet):              "
          f"TAR@FAR=0.01 = 9.5%")
    print("  Our trained model (epoch 15):                   "
          f"TAR@FAR=0.01 = {metrics['tar_far_01']:.1%}")


if __name__ == "__main__":
    main()

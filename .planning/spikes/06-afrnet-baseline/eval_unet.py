"""Evaluate U-Net enhancement impact on fingerprint verification.

Compares two pipelines:
  A) baseline: probe -> embedding model
  B) enhanced: probe -> U-Net -> embedding model

On all 3 protocols: Real vs Altered-{Easy, Medium, Hard}
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))

from dataset import SOCOFingDataset
from model import AFRNetFingerprint
from unet_train import UNet


def get_finger_id(path: str) -> str:
    stem = Path(path).stem
    parts = stem.split("__", 1)
    subject = parts[0]
    finger = parts[1] if len(parts) > 1 else ""
    for suffix in ["_CR", "_Obl", "_Zcut", "_OBL", "_CR.BMP",
                   "_BTH", "_BTR", "_BR", "_BL"]:
        finger = finger.replace(suffix, "")
    return f"{subject}__{finger}"


def preprocess(img_gray: np.ndarray, size: int = 224) -> torch.Tensor:
    h, w = img_gray.shape
    max_side = max(h, w)
    top = (max_side - h) // 2
    left = (max_side - w) // 2
    padded = cv2.copyMakeBorder(
        img_gray, top, max_side - h - top,
        left, max_side - w - left,
        cv2.BORDER_CONSTANT, value=0,
    )
    resized = cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return (t - 0.5) / 0.5


@torch.no_grad()
def get_embeddings_with_optional_unet(
    embed_model, unet_model, ds, device,
    use_unet: bool, max_n: int = 6000,
) -> tuple[torch.Tensor, list[str]]:
    """Extract embeddings, optionally with U-Net enhancement."""
    loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    embs_list, paths_list = [], []
    n_seen = 0
    for i, (img, _) in enumerate(loader):
        if n_seen >= max_n:
            break
        img = img.to(device, non_blocking=True)
        if use_unet and unet_model is not None:
            img = unet_model(img)
        with torch.amp.autocast(device_type=device):
            out = embed_model(img)
        embs_list.append(out["embedding"].float().cpu())
        bs = img.size(0)
        start = i * 64
        for j in range(start, start + bs):
            if j < len(ds.images):
                paths_list.append(ds.images[j][0])
        n_seen += bs
    n_kept = len(paths_list)
    return torch.cat(embs_list)[:n_kept], paths_list


def evaluate_protocol(
    embed_model, unet_model, device,
    gallery_ds, probe_ds, label: str,
    use_unet_on_probe: bool, n_pairs: int = 2000,
) -> dict:
    """Same-finger verification: gallery=Real (no enhancement), probe=Altered (optionally enhanced)."""
    print(f"\n{'='*60}")
    print(f"PROTOCOL: {label} | use_unet_on_probe={use_unet_on_probe}")
    print(f"{'='*60}")

    # Gallery always uses clean (no U-Net on gallery)
    print("  Extracting gallery (Real, no U-Net)...")
    g_embs, g_paths = get_embeddings_with_optional_unet(
        embed_model, None, gallery_ds, device, use_unet=False, max_n=6000,
    )
    print(f"  Extracting probe (Altered, U-Net={use_unet_on_probe})...")
    p_embs, p_paths = get_embeddings_with_optional_unet(
        embed_model, unet_model, probe_ds, device, use_unet=use_unet_on_probe, max_n=3000,
    )
    print(f"  Gallery: {len(g_embs)}, Probe: {len(p_embs)}")

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

    print(f"  Genuine:  mean={genuine.mean():.4f} std={genuine.std():.4f}")
    print(f"  Impostor: mean={impostor.mean():.4f} std={impostor.std():.4f}")
    print(f"  Separation: {genuine.mean() - impostor.mean():.4f}")
    print(f"  AUC: {roc_auc:.4f}  EER: {eer:.4f}")
    print(f"  TAR@FAR=0.01:  {tar_at_far(0.01):.4f}")
    print(f"  TAR@FAR=0.001: {tar_at_far(0.001):.4f}")

    return {
        "label": label,
        "use_unet": use_unet_on_probe,
        "auc": roc_auc,
        "eer": eer,
        "tar_far_01": tar_at_far(0.01),
        "tar_far_001": tar_at_far(0.001),
        "genuine": genuine,
        "impostor": impostor,
        "genuine_mean": float(genuine.mean()),
        "impostor_mean": float(impostor.mean()),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    # Embedding model
    embed_ckpt = torch.load(SPIKE06 / "best_model.pt",
                            map_location=device, weights_only=False)
    embed_model = AFRNetFingerprint(
        num_classes=540, embedding_dim=512, s=30.0, m=0.5,
    ).to(device)
    embed_model.load_state_dict(embed_ckpt["model_state"])
    embed_model.eval()

    # U-Net
    unet_ckpt = torch.load(SPIKE06 / "unet_best.pt",
                           map_location=device, weights_only=False)
    unet_model = UNet(in_ch=1, out_ch=1, base=32, depth=4).to(device)
    unet_model.load_state_dict(unet_ckpt["model_state"])
    unet_model.eval()
    n_unet = sum(p.numel() for p in unet_model.parameters())
    print(f"  U-Net: {n_unet:,} params (val L1: {unet_ckpt['metrics']['l1']:.4f})")

    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    real_ds = SOCOFingDataset(root, "Real", augment=False)
    easy_ds = SOCOFingDataset(root, "Altered/Altered-Easy", augment=False)
    medium_ds = SOCOFingDataset(root, "Altered/Altered-Medium", augment=False)
    hard_ds = SOCOFingDataset(root, "Altered/Altered-Hard", augment=False)

    results_baseline = []
    results_enhanced = []
    for level, probe_ds in [("Easy", easy_ds), ("Medium", medium_ds), ("Hard", hard_ds)]:
        # Baseline
        b = evaluate_protocol(
            embed_model, unet_model, device,
            gallery_ds=real_ds, probe_ds=probe_ds,
            label=f"Real vs Altered-{level}",
            use_unet_on_probe=False,
        )
        results_baseline.append(b)
        # U-Net enhanced
        e = evaluate_protocol(
            embed_model, unet_model, device,
            gallery_ds=real_ds, probe_ds=probe_ds,
            label=f"Real vs Altered-{level}",
            use_unet_on_probe=True,
        )
        results_enhanced.append(e)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Baseline vs U-Net Enhanced")
    print(f"{'='*70}")
    print(f"{'Protocol':<26s} {'Metric':<10s} {'Baseline':>10s} {'U-Net':>10s} {'Delta':>8s}")
    print("-" * 70)
    for b, e in zip(results_baseline, results_enhanced):
        for metric, key in [("AUC", "auc"), ("EER", "eer"),
                            ("TAR@0.01", "tar_far_01"),
                            ("TAR@0.001", "tar_far_001")]:
            bv = b[key]
            ev = e[key]
            if key == "eer":
                delta = f"{(bv - ev) * 100:+.2f}pp"
            else:
                delta = f"{(ev - bv) * 100:+.2f}pp"
            print(f"{b['label']:<26s} {metric:<10s} {bv:>10.4f} {ev:>10.4f} {delta:>8s}")
        print()

    # Save
    out = SPIKE06 / "unet_eval_results.npz"
    np.savez(
        out,
        **{f"baseline_{b['label'].replace(' ', '_')}_genuine": b["genuine"]
           for b in results_baseline},
        **{f"baseline_{b['label'].replace(' ', '_')}_impostor": b["impostor"]
           for b in results_baseline},
        **{f"enhanced_{e['label'].replace(' ', '_')}_genuine": e["genuine"]
           for e in results_enhanced},
        **{f"enhanced_{e['label'].replace(' ', '_')}_impostor": e["impostor"]
           for e in results_enhanced},
    )
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

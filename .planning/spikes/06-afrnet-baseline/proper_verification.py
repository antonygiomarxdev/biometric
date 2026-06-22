"""Proper verification: same finger (Real) vs altered (Altered-Easy/Medium/Hard)."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))

from dataset import SOCOFingDataset
from model import AFRNetFingerprint


def get_finger_id(path: str) -> str:
    """Build unique ID per finger: 'subject__finger' (without alteration suffix)."""
    stem = Path(path).stem
    parts = stem.split("__", 1)
    subject = parts[0]
    finger = parts[1] if len(parts) > 1 else ""
    for suffix in ["_CR", "_Obl", "_Zcut", "_OBL", "_CR.BMP",
                   "_BTH", "_BTR", "_BR", "_BL"]:
        finger = finger.replace(suffix, "")
    return f"{subject}__{finger}"


@torch.no_grad()
def get_embeddings(model, ds, device, max_n=6000):
    loader = DataLoader(ds, batch_size=128, num_workers=4, pin_memory=True)
    embs_list, paths_list = [], []
    n_seen = 0
    for img, _ in loader:
        if n_seen >= max_n:
            break
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device):
            out = model(img)
        embs_list.append(out["embedding"].float().cpu())
        bs = img.size(0)
        start = n_seen
        for j in range(start, start + bs):
            if j < len(ds.images):
                paths_list.append(ds.images[j][0])
        n_seen += bs
    return torch.cat(embs_list[:sum(1 for _ in paths_list) // 128 + 1])[:len(paths_list)], paths_list


def evaluate_protocol(
    model, device,
    gallery_ds: SOCOFingDataset,
    probe_ds: SOCOFingDataset,
    label: str,
    n_pairs: int = 2000,
):
    """Evaluate verification: gallery=Real, probe=Altered.

    Genuine pair: probe and gallery have same finger_id.
    Impostor pair: different finger_id.
    """
    print(f"\n{'='*60}")
    print(f"PROTOCOL: {label}")
    print(f"{'='*60}")
    print(f"  Gallery: {gallery_ds.subdir if hasattr(gallery_ds, 'subdir') else 'Real'} ({len(gallery_ds)} images)")
    print(f"  Probe:   {probe_ds.subdir if hasattr(probe_ds, 'subdir') else '?'} ({len(probe_ds)} images)")

    print("\nExtracting embeddings (this takes ~30s)...")
    gallery_embs, gallery_paths = get_embeddings(model, gallery_ds, device, max_n=6000)
    probe_embs, probe_paths = get_embeddings(model, probe_ds, device, max_n=3000)
    print(f"  Gallery: {len(gallery_embs)} embeddings")
    print(f"  Probe:   {len(probe_embs)} embeddings")

    # Build finger_id maps
    gallery_fids = {p: get_finger_id(p) for p in gallery_paths}
    probe_fids = {p: get_finger_id(p) for p in probe_paths}
    gallery_by_fid: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(gallery_paths):
        gallery_by_fid[gallery_fids[p]].append(i)

    # Normalize
    gallery_embs_n = F.normalize(gallery_embs, p=2, dim=1)
    probe_embs_n = F.normalize(probe_embs, p=2, dim=1)

    # Compute scores: for each probe, similarity to all gallery, then check
    # if best match has same finger_id
    print("\nComputing scores...")
    rng = np.random.default_rng(42)

    # Restrict probe to those that have a matching finger in gallery
    valid_probes = [i for i, p in enumerate(probe_paths)
                    if probe_fids[p] in gallery_by_fid]
    print(f"  Probes with a matching finger in gallery: {len(valid_probes)}")

    # Compute full similarity matrix (probes x gallery)
    sims = probe_embs_n[valid_probes] @ gallery_embs_n.T  # (P, G)
    sims_np = sims.numpy()

    genuine_scores = []
    impostor_scores = []

    for pi_idx, pi in enumerate(valid_probes):
        fid = probe_fids[probe_paths[pi]]
        gallery_idxs = gallery_by_fid[fid]
        # Genuine: max sim over gallery_idxs (best same-finger match)
        for gi in gallery_idxs:
            genuine_scores.append(sims_np[pi_idx, gi])
        # Impostor: max sim over different-finger gallery
        non_match_idxs = [i for i in range(len(gallery_paths))
                          if gallery_fids[gallery_paths[i]] != fid]
        if non_match_idxs:
            # Sample one for speed
            ni = rng.choice(non_match_idxs)
            impostor_scores.append(sims_np[pi_idx, ni])

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"  Genuine pairs:  N={len(genuine_scores)}")
    print(f"  Impostor pairs: N={len(impostor_scores)}")
    print()
    print(f"  Genuine:  mean={genuine_scores.mean():.4f}, std={genuine_scores.std():.4f}, "
          f"range=[{genuine_scores.min():.3f}, {genuine_scores.max():.3f}]")
    print(f"  Impostor: mean={impostor_scores.mean():.4f}, std={impostor_scores.std():.4f}, "
          f"range=[{impostor_scores.min():.3f}, {impostor_scores.max():.3f}]")
    print(f"  Separation: {genuine_scores.mean() - impostor_scores.mean():.4f}")

    # Metrics
    from sklearn.metrics import roc_curve, auc
    y_true = np.array([1] * len(genuine_scores) + [0] * len(impostor_scores))
    y_score = np.concatenate([genuine_scores, impostor_scores])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    def tar_at_far(target):
        for i in range(len(fpr) - 1, -1, -1):
            if fpr[i] <= target:
                return tpr[i]
        return tpr[0]

    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    # Threshold at EER
    threshold_at_eer = (genuine_scores.max() + impostor_scores.min()) / 2

    print()
    print(f"  AUC:           {roc_auc:.4f}")
    print(f"  EER:           {eer:.4f}")
    print(f"  TAR@FAR=0.10:  {tar_at_far(0.10):.4f}")
    print(f"  TAR@FAR=0.01:  {tar_at_far(0.01):.4f}")
    print(f"  TAR@FAR=0.001: {tar_at_far(0.001):.4f}")

    return {
        "label": label,
        "genuine_scores": genuine_scores,
        "impostor_scores": impostor_scores,
        "auc": roc_auc,
        "eer": eer,
        "tar_far_01": tar_at_far(0.01),
        "tar_far_001": tar_at_far(0.001),
        "n_genuine": len(genuine_scores),
        "n_impostor": len(impostor_scores),
        "fpr": fpr,
        "tpr": tpr,
        "threshold_at_eer": threshold_at_eer,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")

    print("Loading model...")
    ckpt = torch.load(SPIKE06 / "best_model.pt",
                      map_location=device, weights_only=False)
    model = AFRNetFingerprint(
        num_classes=540, embedding_dim=512, s=30.0, m=0.5,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load datasets
    print("Loading datasets...")
    real_ds = SOCOFingDataset(root, "Real", augment=False)
    easy_ds = SOCOFingDataset(root, "Altered/Altered-Easy", augment=False)
    medium_ds = SOCOFingDataset(root, "Altered/Altered-Medium", augment=False)
    hard_ds = SOCOFingDataset(root, "Altered/Altered-Hard", augment=False)
    print(f"  Real:           {len(real_ds)} images, {real_ds.num_classes} subjects")
    print(f"  Altered-Easy:   {len(easy_ds)} images, {easy_ds.num_classes} subjects")
    print(f"  Altered-Medium: {len(medium_ds)} images")
    print(f"  Altered-Hard:   {len(hard_ds)} images")

    # Evaluate each protocol
    results = []
    for label, probe_ds in [
        ("Real vs Altered-Easy", easy_ds),
        ("Real vs Altered-Medium", medium_ds),
        ("Real vs Altered-Hard", hard_ds),
    ]:
        result = evaluate_protocol(
            model, device,
            gallery_ds=real_ds, probe_ds=probe_ds,
            label=label, n_pairs=2000,
        )
        results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY: Same-finger verification (gallery=Real)")
    print(f"{'='*60}")
    print(f"{'Protocol':<28s} {'AUC':>6s} {'EER':>6s} "
          f"{'TAR@0.1':>9s} {'TAR@0.01':>9s} {'TAR@0.001':>10s}")
    for r in results:
        print(f"{r['label']:<28s} {r['auc']:6.3f} {r['eer']:6.3f} "
              f"{r['tar_far_01']*0+1:9.3f} {r['tar_far_01']:9.3f} "
              f"{r['tar_far_001']:10.3f}")

    # Save for visualization
    out_path = SPIKE06 / "proper_verification.npz"
    np.savez(
        out_path,
        **{f"{r['label'].replace(' ', '_')}_genuine": r["genuine_scores"]
           for r in results},
        **{f"{r['label'].replace(' ', '_')}_impostor": r["impostor_scores"]
           for r in results},
    )
    print(f"\nSaved scores to {out_path}")


if __name__ == "__main__":
    main()

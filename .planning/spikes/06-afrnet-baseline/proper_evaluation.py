"""Proper fingerprint evaluation protocol.

The key insight: a fingerprint match means same finger, not just same person.
SOCOFing Real has 10 fingers per person (no duplicates).
SOCOFing Altered-Easy has 3 altered versions of each Real finger.

So:
- Gallery: Real images
- Probe: Altered-Easy images (same finger, altered)
- Genuine pair: Real(N) and Altered(N) where N is the same finger
- Impostor pair: Real(N) and Altered(M) where N != M

This is the correct verification protocol for forensics.
"""
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


def parse_filename(stem: str) -> tuple[str, str]:
    """Parse '1__M_Left_index_finger' into ('1', 'M_Left_index_finger').

    The finger ID is the part after the first '__' (or '_' depending on format).
    """
    # Real: '1__M_Left_index_finger' or '30__F_Left_thumb_finger'
    # Altered: '30__F_Left_index_finger_CR'
    parts = stem.split("__", 1)
    subject = parts[0]
    finger = parts[1] if len(parts) > 1 else ""
    # Strip the alteration suffix for Altered
    for suffix in ["_CR", "_Obl", "_Zcut", "_OBL", "_CR.BMP"]:
        finger = finger.replace(suffix, "")
    return subject, finger


def build_finger_id(path: str) -> str:
    """Build a unique ID per finger: 'subject__finger'."""
    stem = Path(path).stem
    subject, finger = parse_filename(stem)
    return f"{subject}__{finger}"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    ckpt = torch.load(SPIKE06 / "best_model.pt",
                      map_location=device, weights_only=False)
    model = AFRNetFingerprint(
        num_classes=540, embedding_dim=512, s=30.0, m=0.5,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    real_dir = root / "Real"
    altered_dir = root / "Altered" / "Altered-Easy"

    # Build datasets
    print("Loading datasets...")
    real_ds = SOCOFingDataset(root, "Real", augment=False)
    altered_ds = SOCOFingDataset(root, "Altered/Altered-Easy", augment=False)
    print(f"  Real: {len(real_ds)} images, {real_ds.num_classes} subjects")
    print(f"  Altered-Easy: {len(altered_ds)} images, {altered_ds.num_classes} subjects")

    # Map real images by finger_id
    real_paths = [p[0] for p in real_ds.images]
    real_finger_ids = {p: build_finger_id(p) for p in real_paths}

    altered_paths = [p[0] for p in altered_ds.images]
    altered_finger_ids = {p: build_finger_id(p) for p in altered_paths}

    # Real finger inventory
    real_fingers = defaultdict(list)
    for p, fid in real_finger_ids.items():
        real_fingers[fid].append(p)

    print(f"  Unique fingers in Real: {len(real_fingers)}")
    multi = [f for f, ps in real_fingers.items() if len(ps) > 1]
    print(f"  Fingers with >1 Real image: {len(multi)}")

    # For verification, use a subset of altered images (10 per subject for speed)
    # Genuine pair: Real[X] vs Altered[X] (same finger)
    # Impostor pair: Real[X] vs Altered[Y] (different finger)

    # Load embeddings
    def get_embeddings(ds, max_n=2000):
        loader = DataLoader(ds, batch_size=64, num_workers=2)
        embs_list = []
        paths_list = []
        with torch.no_grad():
            for i, (img, _) in enumerate(loader):
                img = img.to(device)
                with torch.amp.autocast(device_type=device):
                    out = model(img)
                embs_list.append(out["embedding"].float().cpu())
                bs = img.size(0)
                start = i * 64
                paths_list.extend([ds.images[j][0] for j in range(start, start + bs)])
                if len(paths_list) >= max_n:
                    break
        return torch.cat(embs_list), paths_list[:len(embs_list)]

    print("\nExtracting Real embeddings (gallery)...")
    real_embs, real_paths_used = get_embeddings(real_ds, max_n=2000)
    real_finger_ids_used = {p: build_finger_id(p) for p in real_paths_used}
    real_emb_dict = {p: real_embs[i] for i, p in enumerate(real_paths_used)}
    print(f"  Real gallery: {len(real_embs)} embeddings")

    print("Extracting Altered embeddings (probe)...")
    altered_embs, altered_paths_used = get_embeddings(altered_ds, max_n=2000)
    altered_finger_ids_used = {p: build_finger_id(p) for p in altered_paths_used}
    print(f"  Altered probe: {len(altered_embs)} embeddings")

    # Sample genuine and impostor pairs
    print("\nBuilding genuine and impostor pairs...")
    rng = np.random.default_rng(42)

    # Genuine pairs: probe finger_id has a corresponding real
    genuine_pairs = []
    for p in altered_paths_used:
        fid = altered_finger_ids_used[p]
        if fid in real_finger_ids:
            # Find a real with same finger_id
            real_p = rng.choice(real_fingers[fid])
            genuine_pairs.append((p, real_p))
    print(f"  Genuine pairs available: {len(genuine_pairs)}")
    # Sample N genuine pairs
    n_genuine = min(2000, len(genuine_pairs))
    genuine_sample = rng.choice(len(genuine_pairs), size=n_genuine, replace=False)
    genuine_pairs = [genuine_pairs[i] for i in genuine_sample]

    # Impostor pairs: random probe with random real of different finger
    impostor_pairs = []
    real_paths_arr = np.array(real_paths_used)
    while len(impostor_pairs) < n_genuine:
        p = rng.choice(altered_paths_used)
        fid = altered_finger_ids_used[p]
        candidates = [r for r in real_paths_arr if real_finger_ids_used[r] != fid]
        if candidates:
            r = rng.choice(candidates)
            impostor_pairs.append((p, r))

    print(f"  Final genuine pairs: {len(genuine_pairs)}")
    print(f"  Final impostor pairs: {len(impostor_pairs)}")

    # Compute scores
    def cos_score(p1, p2):
        e1 = altered_emb_dict.get(p1)
        e2 = real_emb_dict.get(p2)
        if e1 is None or e2 is None:
            return None
        e1 = F.normalize(e1.unsqueeze(0), p=2, dim=1)
        e2 = F.normalize(e2.unsqueeze(0), p=2, dim=1)
        return (e1 @ e2.T).item()

    print("\nComputing scores...")
    genuine_scores = []
    for p1, p2 in genuine_pairs:
        s = cos_score(p1, p2)
        if s is not None:
            genuine_scores.append(s)
    impostor_scores = []
    for p1, p2 in impostor_pairs:
        s = cos_score(p1, p2)
        if s is not None:
            impostor_scores.append(s)

    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    print(f"\n{'='*60}")
    print(f"CORRECT VERIFICATION PROTOCOL")
    print(f"{'='*60}")
    print(f"  Gallery: SOCOFing Real (10 fingers × 600 subjects)")
    print(f"  Probe:   SOCOFing Altered-Easy (3 versions × 10 fingers × 600 subjects)")
    print(f"  Genuine:  Same finger (Altered variant of Real finger)")
    print(f"  Impostor: Different finger")
    print()
    print(f"  Genuine pairs:  N={len(genuine_scores)}")
    print(f"  Impostor pairs: N={len(impostor_scores)}")
    print()
    print(f"  Genuine scores:  mean={genuine_scores.mean():.4f}, "
          f"std={genuine_scores.std():.4f}")
    print(f"  Impostor scores: mean={impostor_scores.mean():.4f}, "
          f"std={impostor_scores.std():.4f}")
    print(f"  Separation:      {genuine_scores.mean() - impostor_scores.mean():.4f}")

    # ROC + TAR@FAR
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

    # EER
    fnr = 1 - tpr
    eer = (fpr[np.nanargmin(np.abs(fpr - fnr))] +
           fnr[np.nanargmin(np.abs(fpr - fnr))]) / 2

    print()
    print(f"  AUC:           {roc_auc:.4f}")
    print(f"  EER:           {eer:.4f}")
    print(f"  TAR@FAR=0.01:  {tar_at_far(0.01):.4f}")
    print(f"  TAR@FAR=0.001: {tar_at_far(0.001):.4f}")

    # Save data for visualization
    out_path = SPIKE06 / "verification_results.npz"
    np.savez(out_path,
             genuine_scores=genuine_scores,
             impostor_scores=impostor_scores,
             genuine_pairs=np.array(genuine_pairs),
             impostor_pairs=np.array(impostor_pairs))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

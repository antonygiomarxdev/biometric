"""Visualize U-Net enhancement results.

Shows side-by-side: Altered -> U-Net -> Real (ground truth) and
score distribution improvement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def preprocess_for_display(img_gray: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img_gray.shape
    max_side = max(h, w)
    top = (max_side - h) // 2
    left = (max_side - w) // 2
    padded = cv2.copyMakeBorder(
        img_gray, top, max_side - h - top,
        left, max_side - w - left,
        cv2.BORDER_CONSTANT, value=0,
    )
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR)


def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """Convert normalized [-1, 1] tensor to uint8 numpy array."""
    img = t.detach().cpu().numpy()
    if img.ndim == 4:
        img = img[0]
    img = (img * 0.5 + 0.5) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8).squeeze()


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = SPIKE06 / "visualizations"
    out_dir.mkdir(exist_ok=True)

    print("Loading models...")
    embed_ckpt = torch.load(SPIKE06 / "best_model.pt",
                            map_location=device, weights_only=False)
    embed_model = AFRNetFingerprint(
        num_classes=540, embedding_dim=512, s=30.0, m=0.5,
    ).to(device)
    embed_model.load_state_dict(embed_ckpt["model_state"])
    embed_model.eval()

    unet_ckpt = torch.load(SPIKE06 / "unet_best.pt",
                           map_location=device, weights_only=False)
    unet_model = UNet(in_ch=1, out_ch=1, base=32, depth=4).to(device)
    unet_model.load_state_dict(unet_ckpt["model_state"])
    unet_model.eval()

    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    real_ds = SOCOFingDataset(root, "Real", augment=False)
    hard_ds = SOCOFingDataset(root, "Altered/Altered-Hard", augment=False)

    # ----------------------------------------------------------------
    # Visualization 16: Altered -> U-Net -> Real (triplet)
    # ----------------------------------------------------------------
    print("\n16. U-Net enhancement samples (Altered -> U-Net -> Real)...")
    rng = np.random.default_rng(0)
    sample_idxs = rng.choice(len(hard_ds), size=8, replace=False)

    fig, axes = plt.subplots(3, 8, figsize=(20, 7))
    for col, idx in enumerate(sample_idxs):
        altered_path, _ = hard_ds.images[idx]
        real_fid = get_finger_id(altered_path)

        # Find corresponding Real image
        subj, finger = real_fid.split("__", 1)
        target_stem = f"{subj}__{finger}"
        real_path = None
        for rp, _ in real_ds.images:
            if Path(rp).stem == target_stem:
                real_path = rp
                break

        # Altered (top)
        alt_img = cv2.imread(altered_path, cv2.IMREAD_GRAYSCALE)
        alt_pre = preprocess_for_display(alt_img)
        alt_t = torch.from_numpy(alt_pre).float().unsqueeze(0).unsqueeze(0) / 255.0
        alt_t = (alt_t - 0.5) / 0.5
        alt_t = alt_t.to(device)

        # U-Net enhanced (middle)
        with torch.amp.autocast(device_type=device):
            enh_t = unet_model(alt_t)
        enh_img = tensor_to_img(enh_t)

        # Real (bottom)
        if real_path:
            real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
            real_pre = preprocess_for_display(real_img)
        else:
            real_pre = np.zeros_like(alt_pre)

        # Compute similarity of enhanced probe vs Real
        with torch.amp.autocast(device_type=device):
            emb_enh = embed_model(enh_t)["embedding"]
            emb_real = embed_model(
                torch.from_numpy(real_pre).float().unsqueeze(0).unsqueeze(0).to(device)
                / 255.0 * 2 - 1
            )["embedding"]
        sim = F.cosine_similarity(emb_enh, emb_real).item()

        # Compute baseline similarity (altered -> Real)
        with torch.amp.autocast(device_type=device):
            emb_alt = embed_model(alt_t)["embedding"]
        sim_baseline = F.cosine_similarity(emb_alt, emb_real).item()

        # Show
        axes[0, col].imshow(alt_pre, cmap='gray')
        axes[0, col].set_title(f"Altered\n({alt_img.shape[0]}x{alt_img.shape[1]})\n"
                              f"sim={sim_baseline:.3f}",
                              fontsize=8, color='red')
        axes[0, col].axis('off')

        axes[1, col].imshow(enh_img, cmap='gray')
        axes[1, col].set_title(f"U-Net enhanced\n(224x224)\n"
                              f"sim={sim:.3f}",
                              fontsize=8, color='green')
        axes[1, col].axis('off')

        axes[2, col].imshow(real_pre, cmap='gray')
        axes[2, col].set_title(f"Real (target)\nFID: {real_fid}",
                              fontsize=8, color='black')
        axes[2, col].axis('off')

    fig.suptitle("U-Net Enhancement: Altered-Hard -> Cleaned-up -> Real "
                 "(green: sim improved)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "16_unet_enhancement.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> 16_unet_enhancement.png")

    # ----------------------------------------------------------------
    # Visualization 17: Score distributions: baseline vs U-Net
    # ----------------------------------------------------------------
    print("\n17. Score distributions: baseline vs U-Net enhanced...")
    data = np.load(SPIKE06 / "unet_eval_results.npz")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, level in zip(axes, ["Easy", "Medium", "Hard"]):
        for suffix, color, style, label in [
            ("baseline", "gray", "--", "Baseline"),
            ("enhanced", "blue", "-", "U-Net enhanced"),
        ]:
            g = data[f"{suffix}_Real_vs_Altered-{level}_genuine"]
            i = data[f"{suffix}_Real_vs_Altered-{level}_impostor"]
            ax.hist(g, bins=60, range=(-0.3, 1.0), alpha=0.45,
                    color=color, density=True, histtype="step", linewidth=2,
                    linestyle=style, label=f"{label} genuine (μ={g.mean():.3f})")
            ax.hist(i, bins=60, range=(-0.3, 1.0), alpha=0.30,
                    color=color, density=True, histtype="step", linewidth=1.5,
                    linestyle=style)
        ax.set_title(f"Altered-{level}")
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Score distributions: Baseline (dashed) vs U-Net Enhanced (solid)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "17_score_distributions_unet.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> 17_score_distributions_unet.png")

    # ----------------------------------------------------------------
    # Visualization 18: ROC baseline vs U-Net (all 3 protocols)
    # ----------------------------------------------------------------
    print("\n18. ROC comparison: baseline vs U-Net...")
    fig, ax = plt.subplots(figsize=(9, 7))
    for level, color in [("Easy", "green"), ("Medium", "orange"), ("Hard", "red")]:
        for suffix, style in [("baseline", ":"), ("enhanced", "-")]:
            g = data[f"{suffix}_Real_vs_Altered-{level}_genuine"]
            i = data[f"{suffix}_Real_vs_Altered-{level}_impostor"]
            y_true = np.array([1] * len(g) + [0] * len(i))
            y_score = np.concatenate([g, i])
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            lbl = f"Altered-{level} ({'U-Net' if suffix == 'enhanced' else 'baseline'})"
            ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=2, label=lbl)

            # Mark TAR@FAR=0.01
            for k in range(len(fpr) - 1, -1, -1):
                if fpr[k] <= 0.01:
                    ax.scatter([fpr[k]], [tpr[k]], color=color, s=50, zorder=5,
                               marker='o' if suffix == 'enhanced' else 'x')
                    break

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xscale('log')
    ax.set_xlabel('False Accept Rate (FAR)')
    ax.set_ylabel('True Accept Rate (TAR)')
    ax.set_title('ROC: Baseline (dotted, X) vs U-Net Enhanced (solid, O)\n'
                 'Same-finger verification, gallery=Real, probe=Altered')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(out_dir / "18_roc_unet_vs_baseline.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> 18_roc_unet_vs_baseline.png")

    # ----------------------------------------------------------------
    # Visualization 19: TAR@FAR improvement bars
    # ----------------------------------------------------------------
    print("\n19. TAR@FAR improvement bar chart...")
    protocols = ["Easy", "Medium", "Hard"]
    baseline_tar = []
    enhanced_tar = []
    for level in protocols:
        b = data[f"baseline_Real_vs_Altered-{level}_genuine"]
        bi = data[f"baseline_Real_vs_Altered-{level}_impostor"]
        e = data[f"enhanced_Real_vs_Altered-{level}_genuine"]
        ei = data[f"enhanced_Real_vs_Altered-{level}_impostor"]
        from sklearn.metrics import roc_curve
        fpr_b, tpr_b, _ = roc_curve(
            np.array([1] * len(b) + [0] * len(bi)),
            np.concatenate([b, bi])
        )
        fpr_e, tpr_e, _ = roc_curve(
            np.array([1] * len(e) + [0] * len(ei)),
            np.concatenate([e, ei])
        )
        def tar01(fpr, tpr):
            for k in range(len(fpr) - 1, -1, -1):
                if fpr[k] <= 0.01:
                    return tpr[k]
            return tpr[0]
        baseline_tar.append(tar01(fpr_b, tpr_b) * 100)
        enhanced_tar.append(tar01(fpr_e, tpr_e) * 100)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(protocols))
    width = 0.35
    bars1 = ax.bar(x - width/2, baseline_tar, width, label='Baseline',
                   color='gray', alpha=0.7)
    bars2 = ax.bar(x + width/2, enhanced_tar, width, label='U-Net enhanced',
                   color='blue', alpha=0.7)
    for bar, val in zip(bars1, baseline_tar):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.2f}%', ha='center', fontsize=9)
    for bar, val in zip(bars2, enhanced_tar):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.2f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel('TAR@FAR=0.01 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Altered-{p}' for p in protocols])
    ax.set_title('TAR@FAR=0.01: Baseline vs U-Net Enhanced\n'
                 '(higher is better)', fontsize=12)
    ax.set_ylim([95, 100.5])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / "19_tar_improvement.png", dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> 19_tar_improvement.png")

    print(f"\nAll U-Net visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()

"""Visual analysis of trained AFR-Net model.

Generates:
1. ROC curve
2. Cosine similarity distribution (genuine vs impostor)
3. t-SNE of embeddings
4. Top-5 retrievals gallery (correct + error cases)
5. Learning curves
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

SPIKE06 = Path(__file__).parent
sys.path.insert(0, str(SPIKE06))

from dataset import SOCOFingDataset, split_subjects
from model import AFRNetFingerprint


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AFRNetFingerprint(
        num_classes=540, embedding_dim=512, s=30.0, m=0.5,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def extract_all(model, val_loader, device):
    embs, sids, paths = [], [], []
    with torch.no_grad():
        for img, sid in val_loader:
            img = img.to(device)
            with torch.amp.autocast(device_type=device):
                out = model(img)
            embs.append(out["embedding"].float().cpu())
            sids.extend(sid.tolist())
    # We need paths - reload from the val_ds
    return torch.cat(embs), torch.tensor(sids)


def parse_train_log(log_path: Path) -> dict[str, list]:
    """Parse train.log for per-epoch metrics."""
    epochs_data = []
    with open(log_path) as f:
        for line in f:
            m = re.match(
                r"E\s*(\d+)/\d+ \| "
                r"train L=([\d.]+) A=([\d.]+) \| "
                r"val L=([\d.]+) A=([\d.]+) \| "
                r"rank1=([\d.]+) rank5=([\d.]+) \| "
                r"AUC=([\d.]+) TAR01=([\d.]+) TAR001=([\d.]+) \| "
                r"EER=([\d.]+) \| "
                r"lr=([\d.eE-]+) ([\d.]+)s",
                line.strip()
            )
            if m:
                epochs_data.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "train_acc": float(m.group(3)),
                    "val_loss": float(m.group(4)),
                    "val_acc": float(m.group(5)),
                    "rank1": float(m.group(6)),
                    "rank5": float(m.group(7)),
                    "auc": float(m.group(8)),
                    "tar_far_01": float(m.group(9)),
                    "tar_far_001": float(m.group(10)),
                    "eer": float(m.group(11)),
                })
    return {"epochs": epochs_data}


def compute_all_pairs(embs: torch.Tensor, sids: torch.Tensor):
    """Return (genuine_scores, impostor_scores) lists."""
    embs_n = F.normalize(embs, p=2, dim=1)
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

    rng = np.random.default_rng(42)
    all_subjs = list(subj_to_idx.keys())
    impostor = []
    while len(impostor) < 10000:
        s1, s2 = rng.choice(all_subjs, size=2, replace=False)
        i1 = rng.choice(subj_to_idx[s1])
        i2 = rng.choice(subj_to_idx[s2])
        impostor.append(
            F.cosine_similarity(
                embs_n[i1:i1 + 1], embs_n[i2:i2 + 1],
            ).item()
        )
    return np.array(genuine), np.array(impostor)


# ---------------------------------------------------------------------------
# Visualization 1: ROC curve
# ---------------------------------------------------------------------------

def plot_roc(embs, sids, save_path):
    genuine, impostor = compute_all_pairs(embs, sids)
    y_true = np.array([1] * len(genuine) + [0] * len(impostor))
    y_score = np.concatenate([genuine, impostor])
    fpr, tpr, thr = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'AFR-Net (AUC = {roc_auc:.3f})')
    # Mark TAR@FAR points
    for far_target in [0.01, 0.001, 0.0001]:
        idx = np.searchsorted(fpr, far_target)
        if idx < len(fpr):
            ax.scatter([far_target], [tpr[idx]], s=100, zorder=5)
            ax.annotate(
                f'TAR@FAR={far_target}\nTAR={tpr[idx]:.2f}',
                xy=(far_target, tpr[idx]),
                xytext=(far_target * 1.5, tpr[idx] * 0.85),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray')
            )
    # EER line
    eer_fpr = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, label='Random (AUC=0.5)')
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    ax.set_xscale('log')
    ax.set_xlabel('False Accept Rate (FAR)')
    ax.set_ylabel('True Accept Rate (TAR)')
    ax.set_title('ROC Curve - AFR-Net on SOCOFing Val Set\n'
                 f'(60 unseen subjects, {len(genuine)} genuine, '
                 f'{len(impostor)} impostor pairs)')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 2: Cosine similarity histogram
# ---------------------------------------------------------------------------

def plot_score_distribution(genuine, impostor, save_path, threshold=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-0.5, 1.0, 80)
    ax.hist(genuine, bins=bins, alpha=0.6, color='green',
            label=f'Genuine (N={len(genuine)}, mean={genuine.mean():.3f})',
            density=True)
    ax.hist(impostor, bins=bins, alpha=0.6, color='red',
            label=f'Impostor (N={len(impostor)}, mean={impostor.mean():.3f})',
            density=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold @ EER = {threshold:.3f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution: Genuine vs Impostor Pairs')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 3: t-SNE of embeddings
# ---------------------------------------------------------------------------

def plot_tsne(embs, sids, save_path, n_show=15):
    """t-SNE colored by subject. Show only the n_show most populated subjects."""
    embs_n = F.normalize(embs, p=2, dim=1).numpy()

    # Pick the n_show subjects with the most images
    subj_counts = {s.item(): (sids == s).sum().item() for s in sids.unique()}
    top_subjs = sorted(subj_counts.items(), key=lambda x: -x[1])[:n_show]
    top_subj_set = {s for s, _ in top_subjs}

    # Filter to those subjects
    mask = torch.tensor([s.item() in top_subj_set for s in sids])
    embs_show = embs_n[mask.numpy()]
    sids_show = sids[mask].numpy()
    subj_list = sorted(top_subj_set)

    print(f"  t-SNE on {len(embs_show)} embeddings from {len(subj_list)} subjects...")

    tsne = TSNE(n_components=2, perplexity=15, random_state=42,
                max_iter=1000, init='pca')
    coords = tsne.fit_transform(embs_show)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.get_cmap('tab20')
    for i, subj in enumerate(subj_list):
        m = sids_show == subj
        color = cmap(i / len(subj_list))
        ax.scatter(coords[m, 0], coords[m, 1], c=[color],
                   label=f'Subj {subj}', s=80, alpha=0.8,
                   edgecolors='white', linewidth=0.5)
    ax.set_title(f't-SNE of AFR-Net Embeddings (Val Set)\n'
                 f'Top {n_show} most populated subjects',
                 fontsize=14)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=9, ncol=1, title='Subject ID')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 4: Top-5 retrievals gallery
# ---------------------------------------------------------------------------

def plot_retrievals(embs, sids, paths, save_path, n_queries=8, top_k=5):
    """For n_queries random probes, show the top-k retrievals side by side."""
    embs_n = F.normalize(embs, p=2, dim=1)
    sims = embs_n @ embs_n.T  # (N, N)

    rng = np.random.default_rng(7)
    # Pick queries from subjects with at least 2 images
    subj_to_idx: dict[int, list[int]] = {}
    for i, s in enumerate(sids.tolist()):
        subj_to_idx.setdefault(s, []).append(i)
    valid_subjs = [s for s, idx in subj_to_idx.items() if len(idx) >= 3]
    chosen_subjs = rng.choice(valid_subjs, size=n_queries, replace=False)
    query_idxs = [rng.choice(subj_to_idx[s]) for s in chosen_subjs]

    fig = plt.figure(figsize=(16, 2.2 * n_queries))
    gs = GridSpec(n_queries, top_k + 1, figure=fig,
                  width_ratios=[1.5] + [1] * top_k, wspace=0.05, hspace=0.3)

    for row, qi in enumerate(query_idxs):
        # Query
        ax = fig.add_subplot(gs[row, 0])
        img = cv2.imread(paths[qi], cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Query\nSubj {sids[qi].item()}", fontsize=9)
        ax.axis('off')
        # Border color = green for query
        for spine in ax.spines.values():
            spine.set_edgecolor('blue')
            spine.set_linewidth(3)

        # Top-k retrievals
        scores = sims[qi].clone()
        scores[qi] = -2
        order = scores.argsort(descending=True)[:top_k]
        for k, ri in enumerate(order):
            ax = fig.add_subplot(gs[row, k + 1])
            img = cv2.imread(paths[ri], cv2.IMREAD_GRAYSCALE)
            ax.imshow(img, cmap='gray')
            match = sids[ri].item() == sids[qi].item()
            color = 'green' if match else 'red'
            label = f"Subj {sids[ri].item()}\nsim={scores[ri]:.3f}"
            ax.set_title(label, fontsize=8, color=color)
            ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)

    fig.suptitle("Top-5 Retrievals per Query (green=correct, red=error)",
                 fontsize=13, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 5: Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(log_data, save_path):
    epochs = [e["epoch"] for e in log_data["epochs"]]
    train_loss = [e["train_loss"] for e in log_data["epochs"]]
    val_loss = [e["val_loss"] for e in log_data["epochs"]]
    rank1 = [e["rank1"] for e in log_data["epochs"]]
    tar = [e["tar_far_01"] for e in log_data["epochs"]]
    auc = [e["auc"] for e in log_data["epochs"]]
    eer = [e["eer"] for e in log_data["epochs"]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', label='Train loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Val loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rank-1 + TAR
    ax = axes[0, 1]
    ax.plot(epochs, rank1, 'b-', label='Rank-1', linewidth=2)
    ax.plot(epochs, tar, 'g-', label='TAR@FAR=0.01', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Identification & Verification Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Mark best
    best_idx = int(np.argmax(tar))
    ax.axvline(epochs[best_idx], color='orange', linestyle='--', alpha=0.5,
               label=f'Best (epoch {epochs[best_idx]})')
    ax.legend()

    # AUC
    ax = axes[1, 0]
    ax.plot(epochs, auc, 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('ROC AUC over Training')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3)

    # EER
    ax = axes[1, 1]
    ax.plot(epochs, [e * 100 for e in eer], 'brown', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('EER (%)')
    ax.set_title('Equal Error Rate (lower is better)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 6: Confusion-style — distribution of top-1 sim by correctness
# ---------------------------------------------------------------------------

def plot_top1_confidence(embs, sids, paths, save_path, n_show=12):
    """For n_show random queries, show top-1 retrieval and confidence."""
    embs_n = F.normalize(embs, p=2, dim=1)
    sims = embs_n @ embs_n.T

    rng = np.random.default_rng(11)
    # Pick queries that span correct + incorrect outcomes
    n_total = sims.size(0)
    query_idxs = rng.choice(n_total, size=min(n_show, n_total), replace=False)

    fig = plt.figure(figsize=(16, 2.2 * len(query_idxs)))
    gs = GridSpec(len(query_idxs), 3, figure=fig,
                  width_ratios=[1, 1, 2], wspace=0.1, hspace=0.3)

    correct = 0
    for row, qi in enumerate(query_idxs):
        scores = sims[qi].clone()
        scores[qi] = -2
        order = scores.argsort(descending=True)
        top1 = order[0]
        match = sids[top1].item() == sids[qi].item()
        if match:
            correct += 1
        color = 'green' if match else 'red'

        # Query
        ax = fig.add_subplot(gs[row, 0])
        img = cv2.imread(paths[qi], cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Q: Subj {sids[qi].item()}", fontsize=8)
        ax.axis('off')
        for s in ax.spines.values():
            s.set_edgecolor('blue')
            s.set_linewidth(2)

        # Top-1
        ax = fig.add_subplot(gs[row, 1])
        img = cv2.imread(paths[top1], cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray')
        ax.set_title(f"R1: Subj {sids[top1].item()}", fontsize=8, color=color)
        ax.axis('off')
        for s in ax.spines.values():
            s.set_edgecolor(color)
            s.set_linewidth(2)

        # Sim distribution: top-10 + threshold
        ax = fig.add_subplot(gs[row, 2])
        top10_scores = scores[order[:10]].numpy()
        top10_subjs = [sids[order[k]].item() for k in range(10)]
        colors = [color if t == sids[qi].item() else 'gray' for t in top10_subjs]
        ax.bar(range(10), top10_scores, color=colors)
        ax.axhline(0.249, color='black', linestyle='--', alpha=0.5,
                   label='Threshold @ EER')
        ax.set_xticks(range(10))
        ax.set_xticklabels([f"s{s}" for s in top10_subjs], fontsize=7,
                           rotation=0)
        ax.set_ylim([0, 1.0])
        ax.set_title(f"Top-10 sim scores ({'✓' if match else '✗'})",
                     fontsize=8, color=color)
        ax.legend(fontsize=7)

    acc = correct / len(query_idxs)
    fig.suptitle(
        f"Top-1 Retrieval Detail ({correct}/{len(query_idxs)} correct = {acc:.0%})",
        fontsize=13, y=1.0
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = SPIKE06 / "visualizations"
    out_dir.mkdir(exist_ok=True)

    print("Loading model...")
    model, ckpt = load_model(SPIKE06 / "best_model.pt", device)

    print("Loading val set...")
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    ds = SOCOFingDataset(root, "Real")
    train_ds, val_ds = split_subjects(ds, 540)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=2,
                            pin_memory=True)

    print("Extracting embeddings...")
    embs, sids = extract_all(model, val_loader, device)
    paths = [item[0] for item in val_ds.images]
    print(f"  {embs.shape}, {len(paths)} paths")

    print("\n1. ROC curve...")
    plot_roc(embs, sids, out_dir / "01_roc_curve.png")

    print("\n2. Score distribution...")
    genuine, impostor = compute_all_pairs(embs, sids)
    plot_score_distribution(genuine, impostor,
                            out_dir / "02_score_distribution.png",
                            threshold=0.249)

    print("\n3. t-SNE of embeddings...")
    plot_tsne(embs, sids, out_dir / "03_tsne_embeddings.png", n_show=15)

    print("\n4. Top-5 retrievals gallery...")
    plot_retrievals(embs, sids, paths,
                    out_dir / "04_top5_retrievals.png",
                    n_queries=8, top_k=5)

    print("\n5. Top-1 retrieval detail...")
    plot_top1_confidence(embs, sids, paths,
                         out_dir / "05_top1_detail.png", n_show=12)

    print("\n6. Learning curves...")
    log_data = parse_train_log(SPIKE06 / "train.log")
    if log_data["epochs"]:
        plot_learning_curves(log_data, out_dir / "06_learning_curves.png")
    else:
        print("  No log data parsed")

    print(f"\nAll visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()

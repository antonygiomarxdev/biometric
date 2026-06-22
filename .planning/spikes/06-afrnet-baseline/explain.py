"""Explainable visualizations: GradCAM, attention maps, SIFT keypoints.

Goal: show WHY the model matches two fingerprints, not just THAT it matches.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
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
    return model


# ---------------------------------------------------------------------------
# GradCAM for the CNN branch
# ---------------------------------------------------------------------------

class GradCAMHook:
    """Capture activations and gradients from a target layer."""

    def __init__(self, layer: torch.nn.Module):
        self.activations = None
        self.gradients = None
        self.fwd = layer.register_forward_hook(self._fwd)
        self.bwd = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, mod, inp, out):
        self.activations = out.detach()

    def _bwd(self, mod, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self.fwd.remove()
        self.bwd.remove()

    def compute(self) -> np.ndarray:
        """Returns a heatmap (H, W) normalized to [0, 1]."""
        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def compute_gradcam_for_query_match(
    model, query_tensor: torch.Tensor, target_cnn_features: torch.Tensor,
    device: str, layer_name: str = "cnn.stages.3.blocks.2.conv_dw",
) -> np.ndarray:
    """Compute GradCAM for matching the query to the target's CNN features.

    target_cnn_features should be the 768-D CNN features (not the full embedding).
    """
    model.zero_grad()
    query_tensor = query_tensor.unsqueeze(0).to(device).requires_grad_(True)

    # Attach hook to the target layer (strip the "cnn." prefix for lookup)
    layer_key = layer_name.replace("cnn.", "")
    target_layer = dict(model.cnn.named_modules())[layer_key]
    hook = GradCAMHook(target_layer)

    cnn_feat = model.cnn(query_tensor)

    # Loss: similarity to target CNN features
    cnn_feat_norm = F.normalize(cnn_feat, p=2, dim=1)
    target_norm = F.normalize(target_cnn_features.unsqueeze(0).to(device), p=2, dim=1)
    sim = (cnn_feat_norm * target_norm).sum()
    sim.backward()

    cam = hook.compute()
    hook.remove()
    return cam


# ---------------------------------------------------------------------------
# SIFT keypoint detection
# ---------------------------------------------------------------------------

def detect_sift_keypoints(img: np.ndarray, n_keypoints: int = 30) -> tuple:
    """Detect SIFT keypoints and return (keypoints, descriptors, image_with_kp)."""
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_kp = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, descriptors, img_kp


def match_sift_keypoints(
    img1: np.ndarray, img2: np.ndarray, n_keypoints: int = 50,
) -> tuple:
    """Match SIFT keypoints between two images. Returns (img_matches, n_good_matches)."""
    sift = cv2.SIFT_create(nfeatures=n_keypoints)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None, 0, len(kp1) if kp1 else 0, len(kp2) if kp2 else 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good[:30], None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return img_matches, len(good), len(kp1), len(kp2)


# ---------------------------------------------------------------------------
# Attention map for the ViT branch
# ---------------------------------------------------------------------------

def get_vit_attention(model, x: torch.Tensor) -> np.ndarray:
    """Extract attention map from the last ViT block. Returns a 14x14 grid."""
    # Need to do a forward pass with attention output capture
    # The ViT model from timm supports forward with output_attentions
    try:
        # We need to access internal attention — patch and call
        # Patch the model to return attentions
        vit = model.vit
        # Add attention output capability
        for blk in vit.blocks:
            if not hasattr(blk, "_attn_output_captured"):
                # timm's Attention has attn_drop, so we need to register a hook
                pass
        # Simpler: just hook into the last block's attn module
        attn_module = vit.blocks[-1].attn
        captured = {}
        def hook(module, inp, out):
            # Attention output is (B, num_heads, N, N) — we want the mean over heads
            if isinstance(out, tuple) and len(out) > 1:
                captured["attn"] = out[1]
        h = attn_module.register_forward_hook(hook)
        with torch.no_grad():
            _ = vit(x)
        h.remove()
        if "attn" in captured:
            attn = captured["attn"][0]  # (num_heads, N, N)
            attn = attn.mean(dim=0)  # average over heads: (N, N)
            # CLS token attention to all patches
            cls_attn = attn[0, 1:]  # (196,) for 14x14
            h_side = int(np.sqrt(cls_attn.shape[0]))
            return cls_attn.reshape(h_side, h_side).cpu().numpy()
    except Exception as e:
        print(f"  Attention extraction failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Visualization 1: GradCAM per query-match pair
# ---------------------------------------------------------------------------

def plot_gradcam_pairs(model, embs, cnn_feats, sids, paths, val_loader, device,
                       save_path, n_pairs=6,
                       layer_name="cnn.stages.3.blocks.2.conv_dw"):
    """Show query + top-1 match with GradCAM heatmap overlay."""
    # Get embeddings and find query-match pairs
    embs_n = F.normalize(embs, p=2, dim=1)
    sims = embs_n @ embs_n.T

    rng = np.random.default_rng(13)
    # Pick queries that have at least one correct match
    n_total = sims.size(0)
    # Sample random queries
    query_idxs = sorted(rng.choice(n_total, size=n_pairs, replace=False))

    fig = plt.figure(figsize=(14, 4 * n_pairs))
    gs = GridSpec(n_pairs, 5, figure=fig,
                  width_ratios=[1, 1, 1, 1, 1.2], wspace=0.1, hspace=0.25)

    for row, qi in enumerate(query_idxs):
        scores = sims[qi].clone()
        scores[qi] = -2
        order = scores.argsort(descending=True)
        top1 = order[0]
        match = sids[top1].item() == sids[qi].item()

        # Read images
        q_img = cv2.imread(paths[qi], cv2.IMREAD_GRAYSCALE)
        m_img = cv2.imread(paths[top1], cv2.IMREAD_GRAYSCALE)
        q_resized = cv2.resize(q_img, (224, 224))
        m_resized = cv2.resize(m_img, (224, 224))

        # Get raw image tensor for GradCAM
        # Use the validation set's preprocessing
        q_tensor = val_loader.dataset[qi][0].to(device)
        m_tensor = val_loader.dataset[top1][0].to(device)

        # Query GradCAM (relative to its own CNN features)
        cam_q = compute_gradcam_for_query_match(
            model, q_tensor, cnn_feats[qi].to(device), device, layer_name
        )
        # Match GradCAM (relative to the query's CNN features)
        cam_m = compute_gradcam_for_query_match(
            model, m_tensor, cnn_feats[qi].to(device), device, layer_name
        )

        # Resize CAMs to 224x224
        cam_q_resized = cv2.resize(cam_q, (224, 224))
        cam_m_resized = cv2.resize(cam_m, (224, 224))

        # Overlay heatmap
        def overlay(img, cam, alpha=0.5):
            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam), cv2.COLORMAP_JET
            )
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            return (alpha * heatmap + (1 - alpha) * img_rgb).astype(np.uint8)

        # Column 0: query
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(q_resized, cmap='gray')
        ax.set_title(f"Query\nSubj {sids[qi].item()}", fontsize=9)
        ax.axis('off')

        # Column 1: query with GradCAM
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(overlay(q_resized, cam_q_resized))
        ax.set_title("Query\n+ GradCAM", fontsize=9, color='blue')
        ax.axis('off')

        # Column 2: top-1 match
        ax = fig.add_subplot(gs[row, 2])
        ax.imshow(m_resized, cmap='gray')
        match_str = f"Subj {sids[top1].item()}"
        color = 'green' if match else 'red'
        ax.set_title(f"Top-1 ({'✓' if match else '✗'})\n{match_str}", fontsize=9, color=color)
        ax.axis('off')

        # Column 3: match with GradCAM
        ax = fig.add_subplot(gs[row, 3])
        ax.imshow(overlay(m_resized, cam_m_resized))
        ax.set_title("Top-1\n+ GradCAM", fontsize=9, color=color)
        ax.axis('off')

        # Column 4: SIFT keypoints side by side
        ax = fig.add_subplot(gs[row, 4])
        img_kp_q = detect_sift_keypoints(q_resized)[2]
        img_kp_m = detect_sift_keypoints(m_resized)[2]
        ax.imshow(np.hstack([img_kp_q, img_kp_m]))
        ax.set_title("SIFT keypoints\n(classical minutiae)", fontsize=9, color='gray')
        ax.axis('off')

    fig.suptitle(
        f"Explainable Matching: What the model looks at vs classical keypoints\n"
        f"({'green=match' if match else 'red=mismatch'} based on subject ID)",
        fontsize=12, y=0.99
    )
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 2: SIFT matching
# ---------------------------------------------------------------------------

def plot_sift_matching(embs, sids, paths, save_path, n_pairs=6):
    """Show SIFT keypoint matching between query and top-1."""
    embs_n = F.normalize(embs, p=2, dim=1)
    sims = embs_n @ embs_n.T

    rng = np.random.default_rng(17)
    n_total = sims.size(0)
    query_idxs = sorted(rng.choice(n_total, size=n_pairs, replace=False))

    fig, axes = plt.subplots(n_pairs, 1, figsize=(16, 3.5 * n_pairs))
    if n_pairs == 1:
        axes = [axes]

    for row, qi in enumerate(query_idxs):
        scores = sims[qi].clone()
        scores[qi] = -2
        top1 = scores.argsort(descending=True)[0]
        match = sids[top1].item() == sids[qi].item()

        q_img = cv2.imread(paths[qi], cv2.IMREAD_GRAYSCALE)
        m_img = cv2.imread(paths[top1], cv2.IMREAD_GRAYSCALE)
        q_resized = cv2.resize(q_img, (224, 224))
        m_resized = cv2.resize(m_img, (224, 224))

        img_matches, n_good, n_q, n_m = match_sift_keypoints(
            q_resized, m_resized, n_keypoints=100
        )
        if img_matches is None:
            axes[row].text(0.5, 0.5, "Not enough keypoints",
                           ha='center', va='center')
            axes[row].axis('off')
            continue

        color = 'green' if match else 'red'
        axes[row].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        title = (
            f"Subj {sids[qi].item()} → Subj {sids[top1].item()} "
            f"({'✓' if match else '✗'}, sim={scores[top1]:.3f}) | "
            f"SIFT: {n_good} good matches ({n_q} & {n_m} keypoints)"
        )
        axes[row].set_title(title, fontsize=11, color=color, fontweight='bold')
        axes[row].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 3: ViT attention vs CNN focus
# ---------------------------------------------------------------------------

def plot_attention_vs_gradcam(model, embs, cnn_feats, sids, paths, val_loader,
                              device, save_path, n_show=6):
    """Side-by-side: ViT attention map vs CNN GradCAM."""
    embs_n = F.normalize(embs, p=2, dim=1)
    sims = embs_n @ embs_n.T

    rng = np.random.default_rng(23)
    n_total = sims.size(0)
    query_idxs = sorted(rng.choice(n_total, size=n_show, replace=False))

    fig, axes = plt.subplots(n_show, 3, figsize=(12, 3.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, qi in enumerate(query_idxs):
        q_img = cv2.imread(paths[qi], cv2.IMREAD_GRAYSCALE)
        q_resized = cv2.resize(q_img, (224, 224))
        q_tensor = val_loader.dataset[qi][0].to(device)

        # CNN GradCAM
        cam = compute_gradcam_for_query_match(
            model, q_tensor, cnn_feats[qi].to(device), device,
            "cnn.stages.3.blocks.2.conv_dw"
        )
        cam_resized = cv2.resize(cam, (224, 224))

        # ViT attention
        # Prepare batch with the single image
        with torch.no_grad():
            attn_map = get_vit_attention(
                model, q_tensor.unsqueeze(0).to(device)
            )

        # Column 0: original
        axes[row, 0].imshow(q_resized, cmap='gray')
        axes[row, 0].set_title(f"Query Subj {sids[qi].item()}", fontsize=10)
        axes[row, 0].axis('off')

        # Column 1: CNN GradCAM
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(q_resized, cv2.COLOR_BGR2RGB)
        overlay_cnn = (0.5 * heatmap + 0.5 * img_rgb).astype(np.uint8)
        axes[row, 1].imshow(overlay_cnn)
        axes[row, 1].set_title("CNN Branch (ConvNeXt)\nGradCAM", fontsize=10)
        axes[row, 1].axis('off')

        # Column 2: ViT attention
        if attn_map is not None:
            attn_resized = cv2.resize(attn_map, (224, 224))
            attn_resized = (attn_resized - attn_resized.min()) / (
                attn_resized.max() - attn_resized.min() + 1e-8
            )
            heatmap_vit = cv2.applyColorMap(
                np.uint8(255 * attn_resized), cv2.COLOR_BGR2RGB
            )
            overlay_vit = (0.5 * heatmap_vit + 0.5 * img_rgb).astype(np.uint8)
            axes[row, 2].imshow(overlay_vit)
            axes[row, 2].set_title("ViT Branch\nCLS Attention", fontsize=10)
        else:
            axes[row, 2].text(0.5, 0.5, "No attention",
                              ha='center', va='center')
        axes[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → {save_path}")


# ---------------------------------------------------------------------------
# Visualization 4: Minutiae overlay comparison
# ---------------------------------------------------------------------------

def plot_minutiae_comparison(embs, sids, paths, save_path, n_show=4):
    """Compare classical minutiae detection with model's effective regions."""
    embs_n = F.normalize(embs, p=2, dim=1)
    sims = embs_n @ embs_n.T

    rng = np.random.default_rng(29)
    n_total = sims.size(0)
    query_idxs = sorted(rng.choice(n_total, size=n_show, replace=False))

    fig, axes = plt.subplots(n_show, 2, figsize=(12, 3.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]

    for row, qi in enumerate(query_idxs):
        scores = sims[qi].clone()
        scores[qi] = -2
        top1 = scores.argsort(descending=True)[0]
        match = sids[top1].item() == sids[qi].item()

        q_img = cv2.imread(paths[qi], cv2.IMREAD_GRAYSCALE)
        m_img = cv2.imread(paths[top1], cv2.IMREAD_GRAYSCALE)
        q_resized = cv2.resize(q_img, (224, 224))
        m_resized = cv2.resize(m_img, (224, 224))

        # SIFT on both
        kp1, des1, kp_img1 = detect_sift_keypoints(q_resized, n_keypoints=40)
        kp2, des2, kp_img2 = detect_sift_keypoints(m_resized, n_keypoints=40)

        # SIFT matches
        if des1 is not None and des2 is not None and len(des1) >= 2 and len(des2) >= 2:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]
            img_matches = cv2.drawMatches(
                q_resized, kp1, m_resized, kp2, good[:20], None,
                matchColor=(0, 255, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
        else:
            img_matches = np.hstack([
                cv2.cvtColor(q_resized, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor(m_resized, cv2.COLOR_GRAY2RGB),
            ])

        axes[row, 0].imshow(np.hstack([
            cv2.cvtColor(q_resized, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(m_resized, cv2.COLOR_GRAY2RGB),
        ]))
        axes[row, 0].set_title(
            f"Subj {sids[qi].item()}  vs  Subj {sids[top1].item()}\n"
            f"({len(kp1)} vs {len(kp2)} keypoints)",
            fontsize=10
        )
        axes[row, 0].axis('off')

        axes[row, 1].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        n_matched = len(good) if des1 is not None and des2 is not None and len(good) > 0 else 0
        title_color = 'green' if match else 'red'
        axes[row, 1].set_title(
            f"SIFT matches: {n_matched}\n"
            f"DeepPrint sim: {scores[top1]:.3f} ({'✓' if match else '✗'})",
            fontsize=10, color=title_color
        )
        axes[row, 1].axis('off')

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
    model = load_model(SPIKE06 / "best_model.pt", device)

    print("Loading val set...")
    root = Path("/home/ksante/dev/biometric/apps/backend/static/SOCOFing")
    ds = SOCOFingDataset(root, "Real")
    train_ds, val_ds = split_subjects(ds, 540)
    val_loader = DataLoader(val_ds, batch_size=8, num_workers=2, pin_memory=True)

    print("Extracting embeddings...")
    embs_list, cnn_list, vit_list, sids_list = [], [], [], []
    with torch.no_grad():
        for img, sid in val_loader:
            img = img.to(device)
            with torch.amp.autocast(device_type=device):
                out = model(img)
            embs_list.append(out["embedding"].float().cpu())
            # Also extract CNN and ViT features separately for GradCAM
            cnn_feat = model.cnn(img)
            vit_feat = model.vit(img)
            cnn_list.append(cnn_feat.float().cpu())
            vit_list.append(vit_feat.float().cpu())
            sids_list.extend(sid.tolist())
    embs = torch.cat(embs_list)
    cnn_feats = torch.cat(cnn_list)
    vit_feats = torch.cat(vit_list)
    sids = torch.tensor(sids_list)
    paths = [item[0] for item in val_ds.images]
    print(f"  embeddings: {embs.shape}, cnn: {cnn_feats.shape}, vit: {vit_feats.shape}")

    print("\n1. GradCAM + SIFT side-by-side...")
    plot_gradcam_pairs(model, embs, cnn_feats, sids, paths, val_loader, device,
                       out_dir / "07_gradcam_explainable.png", n_pairs=6)

    print("\n2. SIFT keypoint matching...")
    plot_sift_matching(embs, sids, paths,
                      out_dir / "08_sift_matching.png", n_pairs=6)

    print("\n3. ViT attention vs CNN GradCAM...")
    plot_attention_vs_gradcam(model, embs, cnn_feats, sids, paths, val_loader,
                              device, out_dir / "09_attention_vs_gradcam.png",
                              n_show=6)

    print("\n4. Classical vs deep matching comparison...")
    plot_minutiae_comparison(embs, sids, paths,
                             out_dir / "10_classical_vs_deep.png", n_show=4)

    print(f"\nAll explainable visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()

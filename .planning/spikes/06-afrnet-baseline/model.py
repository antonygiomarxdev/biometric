"""AFR-Net style hybrid fingerprint embedding model.

Architecture (from 2023 paper + adapted for 2025/2026 best practices):
- CNN branch: ConvNeXt-Tiny pre-trained on ImageNet (768-D features)
- ViT branch: ViT-Tiny/16 pre-trained on ImageNet (192-D features)
- Fusion: Linear(960, 512) + BN + L2 normalization
- Loss: ArcFace (s=30, m=0.5) on the 512-D embedding
- Input: grayscale (1 channel), 224x224
"""
from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ArcFace head
# ---------------------------------------------------------------------------

class ArcFaceHead(nn.Module):
    """Linear layer with ArcFace angular margin for metric learning.

    The forward pass takes the embedding and target class id, and returns
    logits where the target class logit has an additive angular margin m
    and the whole vector is scaled by s.
    """

    def __init__(self, in_features: int, num_classes: int,
                 s: float = 30.0, m: float = 0.5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_normal_(self.weight)
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(x, w).clamp(-1.0, 1.0)
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2).clamp(0.0, 1.0)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cond = cos_theta > self.th
        cos_theta_m = torch.where(cond, cos_theta_m, cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1), 1)
        return (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.s


# ---------------------------------------------------------------------------
# AFR-Net style model
# ---------------------------------------------------------------------------

class AFRNetFingerprint(nn.Module):
    """ConvNeXt-Tiny + ViT-Tiny hybrid fingerprint embedding.

    Backbones are loaded with ImageNet pre-trained weights. The first conv
    is converted to accept 1-channel grayscale input.
    """

    def __init__(self, num_classes: int, embedding_dim: int = 512,
                 s: float = 30.0, m: float = 0.5,
                 use_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # CNN branch: ConvNeXt-Tiny, output 768-D
        self.cnn = timm.create_model(
            "convnext_tiny", pretrained=True, in_chans=1, num_classes=0,
        )
        # ViT branch: ViT-Tiny, output 192-D
        self.vit = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, in_chans=1, num_classes=0,
        )

        cnn_dim = self.cnn.num_features  # 768
        vit_dim = self.vit.num_features  # 192

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        # ArcFace head
        self.head = ArcFaceHead(embedding_dim, num_classes, s=s, m=m)

        if use_gradient_checkpointing:
            self.cnn.set_grad_checkpointing(True)
            self.vit.set_grad_checkpointing(True)

    def extract_features(self, x: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run both backbones and return concatenated features."""
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        return cnn_feat, vit_feat

    def forward(self, x: torch.Tensor,
                target: torch.Tensor | None = None
                ) -> dict[str, torch.Tensor]:
        cnn_feat, vit_feat = self.extract_features(x)
        # Concatenate
        fused = torch.cat([cnn_feat, vit_feat], dim=1)
        emb_raw = self.fusion(fused)
        # L2-normalize the embedding
        emb = F.normalize(emb_raw, p=2, dim=1)

        result = {
            "embedding": emb,
            "cnn_features": cnn_feat,
            "vit_features": vit_feat,
        }
        if target is not None:
            logits = self.head(emb, target)
            loss = F.cross_entropy(logits, target)
            acc = (logits.argmax(1) == target).float().mean()
            result["loss"] = loss
            result["logits"] = logits
            result["acc"] = acc
        return result


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count parameters in each sub-component."""
    return {
        "cnn": sum(p.numel() for p in model.cnn.parameters()),
        "vit": sum(p.numel() for p in model.vit.parameters()),
        "fusion": sum(p.numel() for p in model.fusion.parameters()),
        "head": sum(p.numel() for p in model.head.parameters()),
        "total": sum(p.numel() for p in model.parameters()),
    }

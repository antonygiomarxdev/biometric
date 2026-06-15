"""Train binary CNN patch classifier for minutiae detection.

Architecture: small CNN (4 conv blocks + GAP + Linear).
Input: (1, 160, 160) grayscale patch
Output: (1,) sigmoid probability (real minutia vs noise)

Training data: patches.npz + labels.npz from generate_patch_dataset.py
Augmentation: random horizontal/vertical flip, random rotation +/-15 deg
Class imbalance: handled via pos_weight in BCEWithLogitsLoss

Exports the trained model to ONNX (data/models/extract.onnx).

Usage:
    cd apps/backend && PYTHONPATH=. python3 scripts/train_patch_classifier.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import onnx
import onnxruntime as ort

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"

PATCH_SIZE = 160
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class PatchDataset(Dataset):
    """Dataset that applies on-the-fly augmentation to 160x160 patches."""

    def __init__(self, patches: np.ndarray, labels: np.ndarray, augment: bool) -> None:
        self.patches = patches
        self.labels = labels.astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        patch = self.patches[idx]
        label = self.labels[idx]
        if self.augment:
            patch = self._augment(patch)
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - 0.5) / 0.5
        return (
            torch.from_numpy(patch).unsqueeze(0),
            torch.tensor(label, dtype=torch.float32),
        )

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.5:
            patch = patch[:, ::-1].copy()
        if np.random.random() < 0.5:
            patch = patch[::-1, :].copy()
        k = np.random.randint(0, 4)
        if k > 0:
            patch = np.rot90(patch, k=k).copy()
        if np.random.random() < 0.3:
            angle = np.random.uniform(-15, 15)
            from cv2 import warpAffine, getRotationMatrix2D
            M = getRotationMatrix2D((PATCH_SIZE // 2, PATCH_SIZE // 2), angle, 1.0)
            patch = warpAffine(patch, M, (PATCH_SIZE, PATCH_SIZE), flags=3, borderMode=4)
        if np.random.random() < 0.3:
            factor = np.random.uniform(0.9, 1.1)
            patch = np.clip(patch * factor, 0, 255).astype(np.uint8)
        return patch


class PatchClassifier(nn.Module):
    """Small CNN: 4 conv blocks + GAP + Linear classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_preds: list[float] = []
    all_labels: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            logits = model(patches).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            losses.append(loss.item())
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    preds = np.array(all_preds) >= 0.5
    truth = np.array(all_labels) >= 0.5
    tp = int(((preds == 1) & (truth == 1)).sum())
    fp = int(((preds == 1) & (truth == 0)).sum())
    fn = int(((preds == 0) & (truth == 1)).sum())
    tn = int(((preds == 0) & (truth == 0)).sum())
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_patches = np.load(TRAINING_DIR / "train_patches.npz")["arr_0"]
    train_labels = np.load(TRAINING_DIR / "train_labels.npz")["arr_0"]
    test_patches = np.load(TRAINING_DIR / "test_patches.npz")["arr_0"]
    test_labels = np.load(TRAINING_DIR / "test_labels.npz")["arr_0"]
    print(
        f"Loaded: train {train_patches.shape} ({train_labels.sum()} pos / "
        f"{len(train_labels)-train_labels.sum()} neg), "
        f"test {test_patches.shape} ({test_labels.sum()} pos / "
        f"{len(test_labels)-test_labels.sum()} neg)"
    )

    train_ds = PatchDataset(train_patches, train_labels, augment=True)
    test_ds = PatchDataset(test_patches, test_labels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = PatchClassifier().to(device)
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - train_labels.sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_losses: list[float] = []
        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(patches).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses))
        val_metrics = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "time_s": elapsed,
            }
        )
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | train_loss {train_loss:.4f} | "
            f"val_loss {val_metrics['loss']:.4f} | val_acc {val_metrics['accuracy']:.4f} | "
            f"val_f1 {val_metrics['f1']:.4f} | {elapsed:.1f}s"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, test_loader, device)
    print("\n=== Final test metrics ===")
    print(json.dumps(final_metrics, indent=2))

    pt_path = MODELS_DIR / "extract.pt"
    torch.save(best_state, pt_path)
    print(f"Saved PyTorch checkpoint to {pt_path}")

    onnx_path = MODELS_DIR / "extract.onnx"
    model.eval()
    dummy = torch.randn(1, 1, PATCH_SIZE, PATCH_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported ONNX to {onnx_path}")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_input = np.random.rand(2, 1, PATCH_SIZE, PATCH_SIZE).astype(np.float32)
    test_input = (test_input - 0.5) / 0.5
    out = sess.run(None, {"input": test_input})
    print(f"ONNX inference OK: input {test_input.shape} -> output {out[0].shape}")

    log = {
        "history": history,
        "final_metrics": final_metrics,
        "pos_weight": float(pos_weight.item()),
        "patch_size": PATCH_SIZE,
        "epochs_trained": len(history),
        "device": str(device),
    }
    with open(TRAINING_DIR / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {TRAINING_DIR / 'train_log.json'}")


if __name__ == "__main__":
    main()

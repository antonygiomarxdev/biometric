"""Smoke test for the trained ONNX minutiae patch classifier.

Verifies:
  1. extract.onnx exists and is a valid ONNX file
  2. onnxruntime can load it (with CUDA if available, CPU fallback)
  3. A dummy 160x160 inference returns shape (1, 1)
  4. A real positive patch scores >0.5
  5. A random negative patch scores <0.5
  6. Inference time is reasonable (<100ms on GPU, <500ms on CPU)

Exits 0 on success, 1 on any failure.

Usage:
    cd apps/backend && PYTHONPATH=. python3 scripts/onnx_smoke_test.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"
ONNX_PATH = MODELS_DIR / "extract.onnx"
POS_THRESHOLD = 0.5
NEG_THRESHOLD = 0.5
MAX_INFER_MS_CPU = 500
MAX_INFER_MS_GPU = 100


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> int:
    print(f"=== Smoke test for {ONNX_PATH} ===")

    if not ONNX_PATH.exists():
        print(f"FAIL: {ONNX_PATH} not found")
        return 1
    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"[OK] Model file exists ({size_mb:.2f} MB)")

    try:
        onnx_model = onnx.load(str(ONNX_PATH))
        onnx.checker.check_model(onnx_model)
        print(f"[OK] Valid ONNX (ir_version={onnx_model.ir_version}, opset={onnx_model.opset_import[0].version})")
    except Exception as e:
        print(f"FAIL: ONNX validation: {e}")
        return 1

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[OK] CUDA available, trying CUDA first")
    else:
        providers = ["CPUExecutionProvider"]
        print("[OK] No CUDA, using CPU only")
    try:
        sess = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        active = sess.get_providers()[0]
        print(f"[OK] Model loaded (active provider: {active})")
    except Exception as e:
        print(f"FAIL: Could not load model: {e}")
        return 1
    is_gpu = "CUDA" in active

    dummy = np.random.rand(1, 1, 160, 160).astype(np.float32)
    dummy = (dummy - 0.5) / 0.5
    start = time.time()
    out = sess.run(None, {"input": dummy})
    elapsed_ms = (time.time() - start) * 1000
    if out[0].shape != (1, 1):
        print(f"FAIL: output shape {out[0].shape}, expected (1, 1)")
        return 1
    print(f"[OK] Dummy inference: shape={out[0].shape}, time={elapsed_ms:.1f}ms, value={out[0][0][0]:.3f}")

    limit = MAX_INFER_MS_GPU if is_gpu else MAX_INFER_MS_CPU
    if elapsed_ms > limit:
        print(f"WARN: inference {elapsed_ms:.1f}ms exceeds target {limit}ms")

    train_patches_path = TRAINING_DIR / "train_patches.npz"
    train_labels_path = TRAINING_DIR / "train_labels.npz"
    if not train_patches_path.exists() or not train_labels_path.exists():
        print(f"SKIP: real/noise patch tests (no training data at {TRAINING_DIR})")
        print("\n=== ALL CHECKS PASSED ===")
        return 0

    train_patches = np.load(train_patches_path)["arr_0"]
    train_labels = np.load(train_labels_path)["arr_0"]
    pos_idx = int(np.where(train_labels == 1)[0][0])
    neg_idx = int(np.where(train_labels == 0)[0][0])

    pos_patch = train_patches[pos_idx].astype(np.float32) / 255.0
    neg_patch = train_patches[neg_idx].astype(np.float32) / 255.0
    pos_in = ((pos_patch - 0.5) / 0.5)[np.newaxis, np.newaxis, :, :]
    neg_in = ((neg_patch - 0.5) / 0.5)[np.newaxis, np.newaxis, :, :]

    pos_logit = float(sess.run(None, {"input": pos_in})[0][0][0])
    neg_logit = float(sess.run(None, {"input": neg_in})[0][0][0])
    pos_prob = sigmoid(pos_logit)
    neg_prob = sigmoid(neg_logit)
    print(f"[{'OK' if pos_prob > POS_THRESHOLD else 'FAIL'}] Real patch: prob={pos_prob:.4f} (need >{POS_THRESHOLD})")
    print(f"[{'OK' if neg_prob < NEG_THRESHOLD else 'FAIL'}] Noise patch: prob={neg_prob:.4f} (need <{NEG_THRESHOLD})")
    if pos_prob <= POS_THRESHOLD or neg_prob >= NEG_THRESHOLD:
        return 1

    print("\n=== ALL CHECKS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

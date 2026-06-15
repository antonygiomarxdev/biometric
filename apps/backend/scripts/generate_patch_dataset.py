"""Generate patch dataset for minutiae classifier training.

For each SOCOFing Real image:
  1. Load grayscale, enhance+resize via the production pipeline
  2. Compute ground-truth minutiae (Crossing Number on binarized+skeletonized)
  3. Extract 160x160 patches centered on each true minutia (label=1)
  4. Sample random non-minutia points; extract patches (label=0)
  5. Save patches.npz, labels.npz, and train/test split

Uses multiprocessing to parallelize the slow Gabor enhancement.
Caches enhanced images to disk to avoid re-enhancement.

Usage:
    cd apps/backend && PYTHONPATH=. python3 scripts/generate_patch_dataset.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

from src.processing.enhancer import create_enhancer

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = DATA_DIR / "models"
TRAINING_DIR = DATA_DIR / "training"
ENHANCED_CACHE_DIR = TRAINING_DIR / "enhanced_cache"
SOCOFING_REAL = Path(__file__).resolve().parents[1] / "static" / "SOCOFing" / "Real"

PATCH_SIZE = 160
HALF = PATCH_SIZE // 2
N_TRAIN_IMAGES = 80
N_TEST_IMAGES = 20
NEGATIVES_PER_IMAGE = 30
MIN_DIST_FROM_TRUE = 16
RANDOM_SEED = 42
N_WORKERS = 4


def _enhance_one(path_str: str) -> tuple[str, np.ndarray | None]:
    """Worker function: enhance one image, return (path_str, enhanced_array_or_none)."""
    path = Path(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (path_str, None)
    try:
        enhanced = create_enhancer().enhance(img, resize=True)
    except Exception:
        return (path_str, None)
    return (path_str, enhanced)


def _enhance_all_parallel(paths: list[Path]) -> dict[str, np.ndarray]:
    """Parallel enhance all images, cache to disk, return path -> enhanced array."""
    ENHANCED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, np.ndarray] = {}
    todo: list[Path] = []
    for p in paths:
        cache_path = ENHANCED_CACHE_DIR / (p.stem + ".npy")
        if cache_path.exists():
            results[str(p)] = np.load(cache_path)
        else:
            todo.append(p)

    if todo:
        print(f"  Enhancing {len(todo)} images with {N_WORKERS} workers...")
        start = time.time()
        with mp.Pool(N_WORKERS) as pool:
            for i, (path_str, enhanced) in enumerate(pool.imap_unordered(
                _enhance_one, [str(p) for p in todo], chunksize=4
            )):
                if enhanced is not None:
                    np.save(ENHANCED_CACHE_DIR / (Path(path_str).stem + ".npy"), enhanced)
                    results[path_str] = enhanced
                if (i + 1) % 20 == 0:
                    print(f"    {i+1}/{len(todo)} ({time.time()-start:.1f}s)")
        print(f"  Enhancement done in {time.time()-start:.1f}s")

    return results


def compute_ground_truth_minutiae(enhanced: np.ndarray) -> list[tuple[int, int]]:
    """Return (x, y) positions of true minutiae from CN on binarized+skeletonized image."""
    binary = enhanced > 127
    skel = skeletonize(binary)
    h, w = skel.shape
    pts: list[tuple[int, int]] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not skel[y, x]:
                continue
            n = (
                int(skel[y - 1, x]),
                int(skel[y - 1, x + 1]),
                int(skel[y, x + 1]),
                int(skel[y + 1, x + 1]),
                int(skel[y + 1, x]),
                int(skel[y + 1, x - 1]),
                int(skel[y, x - 1]),
                int(skel[y - 1, x - 1]),
            )
            transitions = sum(
                1 for i in range(8) if n[i] == 0 and n[(i + 1) % 8] == 1
            )
            if transitions == 1 or transitions >= 3:
                pts.append((x, y))
    return pts


def extract_patch(image: np.ndarray, x: int, y: int, half: int = HALF) -> np.ndarray | None:
    """Extract a square patch centered on (x, y) with reflection padding."""
    h, w = image.shape
    x0, x1 = x - half, x + half
    y0, y1 = y - half, y + half
    if x1 <= 0 or x0 >= w or y1 <= 0 or y0 >= h:
        return None
    pad_left = max(0, -x0)
    pad_right = max(0, x1 - w)
    pad_top = max(0, -y0)
    pad_bottom = max(0, y1 - h)
    if any(p > 0 for p in (pad_left, pad_right, pad_top, pad_bottom)):
        image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="reflect",
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    return image[y0:y1, x0:x1].copy()


def sample_negatives(
    image_shape: tuple[int, int],
    true_pts: list[tuple[int, int]],
    n: int,
    min_dist: int,
    rng: random.Random,
    half: int = HALF,
) -> list[tuple[int, int]]:
    """Sample n random (x, y) points NOT within min_dist of any true minutia."""
    h, w = image_shape
    true_set = set(true_pts)
    pts: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = n * 20
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        x = rng.randint(half, w - half - 1)
        y = rng.randint(half, h - half - 1)
        too_close = False
        for tx, ty in true_pts:
            if (x - tx) ** 2 + (y - ty) ** 2 < min_dist ** 2:
                too_close = True
                break
        if too_close:
            continue
        if (x, y) in true_set:
            continue
        pts.append((x, y))
    return pts


def process_image(
    enhanced: np.ndarray,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[int]]:
    """Compute ground truth + extract patches for one enhanced image."""
    true_pts = compute_ground_truth_minutiae(enhanced)
    neg_pts = sample_negatives(
        enhanced.shape, true_pts, NEGATIVES_PER_IMAGE, MIN_DIST_FROM_TRUE, rng
    )

    patches: list[np.ndarray] = []
    labels: list[int] = []
    for x, y in true_pts:
        p = extract_patch(enhanced, x, y)
        if p is None or p.shape != (PATCH_SIZE, PATCH_SIZE):
            continue
        patches.append(p)
        labels.append(1)
    for x, y in neg_pts:
        p = extract_patch(enhanced, x, y)
        if p is None or p.shape != (PATCH_SIZE, PATCH_SIZE):
            continue
        patches.append(p)
        labels.append(0)
    return patches, labels


def main() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(SOCOFING_REAL.glob("*.BMP"))
    if not image_paths:
        raise FileNotFoundError(f"No BMP files in {SOCOFING_REAL}")
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(image_paths)
    selected = image_paths[: N_TRAIN_IMAGES + N_TEST_IMAGES]
    train_paths = selected[:N_TRAIN_IMAGES]
    test_paths = selected[N_TRAIN_IMAGES:]

    print(f"Total available: {len(image_paths)}")
    print(f"Train: {len(train_paths)}, Test: {len(test_paths)}")

    enhanced_map = _enhance_all_parallel(selected)
    print(f"Enhanced: {len(enhanced_map)}/{len(selected)} images")

    train_patches: list[np.ndarray] = []
    train_labels: list[int] = []
    test_patches: list[np.ndarray] = []
    test_labels: list[int] = []

    start = time.time()
    for i, p in enumerate(train_paths):
        enhanced = enhanced_map.get(str(p))
        if enhanced is None:
            continue
        patches, labels = process_image(enhanced, rng)
        train_patches.extend(patches)
        train_labels.extend(labels)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - start
            print(
                f"  train {i+1}/{len(train_paths)} | "
                f"pos={sum(train_labels)} neg={len(train_labels)-sum(train_labels)} | "
                f"{elapsed:.1f}s"
            )

    for i, p in enumerate(test_paths):
        enhanced = enhanced_map.get(str(p))
        if enhanced is None:
            continue
        patches, labels = process_image(enhanced, rng)
        test_patches.extend(patches)
        test_labels.extend(labels)
        if (i + 1) % 10 == 0:
            print(f"  test {i+1}/{len(test_paths)}")

    train_patches_arr = np.stack(train_patches).astype(np.uint8)
    train_labels_arr = np.array(train_labels, dtype=np.uint8)
    test_patches_arr = np.stack(test_patches).astype(np.uint8)
    test_labels_arr = np.array(test_labels, dtype=np.uint8)

    np.savez_compressed(TRAINING_DIR / "train_patches.npz", train_patches_arr)
    np.savez_compressed(TRAINING_DIR / "train_labels.npz", train_labels_arr)
    np.savez_compressed(TRAINING_DIR / "test_patches.npz", test_patches_arr)
    np.savez_compressed(TRAINING_DIR / "test_labels.npz", test_labels_arr)

    metadata = {
        "patch_size": PATCH_SIZE,
        "n_train_images": len(train_paths),
        "n_test_images": len(test_paths),
        "train_positive": int(train_labels_arr.sum()),
        "train_negative": int(len(train_labels_arr) - train_labels_arr.sum()),
        "test_positive": int(test_labels_arr.sum()),
        "test_negative": int(len(test_labels_arr) - test_labels_arr.sum()),
        "train_patch_shape": list(train_patches_arr.shape),
        "test_patch_shape": list(test_patches_arr.shape),
        "random_seed": RANDOM_SEED,
        "n_workers": N_WORKERS,
    }
    with open(TRAINING_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Dataset summary ===")
    print(json.dumps(metadata, indent=2))
    print(f"\nSaved to {TRAINING_DIR}")


if __name__ == "__main__":
    main()

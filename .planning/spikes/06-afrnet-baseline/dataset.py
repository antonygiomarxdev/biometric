"""SOCOFing dataset loader with heavy augmentation for latent robustness.

Following the research findings, we use aggressive augmentations to simulate
the conditions of latent prints:
- Large rotation ranges (±180°)
- Elastic deformation
- Random occlusion (simulating partial prints)
- Contrast/brightness jitter (simulating dry/wet fingers)
- Additive Gaussian noise (simulating sensor noise)
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SOCOFingDataset(Dataset):
    """SOCOFing dataset with subject-id mapping and heavy augmentation."""

    def __init__(self, root: Path, subdir: str = "Real",
                 augment: bool = False) -> None:
        self.root = Path(root) / subdir
        self.augment = augment

        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")

        paths = sorted(self.root.rglob("*.BMP"))
        if not paths:
            raise FileNotFoundError(f"No BMP files in {self.root}")

        # Group by subject (prefix before first underscore)
        subject_images: dict[str, list[Path]] = defaultdict(list)
        for p in paths:
            subj = p.stem.split("_")[0]
            subject_images[subj].append(p)

        # Sort subjects for reproducible ordering
        self.subjects = sorted(subject_images.keys())
        self.subject_to_id = {s: i for i, s in enumerate(self.subjects)}

        # Build image list: (path, subject_id)
        self.images: list[tuple[str, int]] = []
        for subj in self.subjects:
            sid = self.subject_to_id[subj]
            for p in subject_images[subj]:
                self.images.append((str(p), sid))

        self.num_classes = len(self.subjects)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, sid = self.images[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read: {path}")
        img = self._preprocess(img)
        if self.augment:
            img = self._augment(img)
        tensor = torch.from_numpy(img).float()
        # Normalize to mean=0.5, std=0.5 (i.e. [-1, 1] from [0, 1])
        tensor = (tensor / 255.0 - 0.5) / 0.5
        return tensor.unsqueeze(0), sid

    def _preprocess(self, img: np.ndarray, size: int = 224) -> np.ndarray:
        """Pad to square, resize to target size."""
        h, w = img.shape
        max_side = max(h, w)
        top = (max_side - h) // 2
        left = (max_side - w) // 2
        padded = cv2.copyMakeBorder(
            img, top, max_side - h - top,
            left, max_side - w - left,
            cv2.BORDER_CONSTANT, value=0,
        )
        return cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Heavy augmentation pipeline for latent-print robustness.

        Order matters: geometric first, then photometric.
        """
        img = self._aug_rotate(img)
        img = self._aug_elastic(img)
        img = self._aug_occlude(img)
        img = self._aug_photometric(img)
        img = self._aug_noise(img)
        return img

    def _aug_rotate(self, img: np.ndarray) -> np.ndarray:
        """Random rotation in [-180, 180] (latents can be at any angle)."""
        if random.random() < 0.7:
            angle = random.uniform(-180, 180)
            h, w = img.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        return img

    def _aug_elastic(self, img: np.ndarray) -> np.ndarray:
        """Elastic deformation simulating skin distortion."""
        if random.random() < 0.5:
            h, w = img.shape
            # Random displacement field, smoothed by Gaussian
            dx = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (15, 15), 5,
            ) * 4
            dy = cv2.GaussianBlur(
                (np.random.rand(h, w) * 2 - 1).astype(np.float32),
                (15, 15), 5,
            ) * 4
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = (x + dx).astype(np.float32)
            map_y = (y + dy).astype(np.float32)
            img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
        return img

    def _aug_occlude(self, img: np.ndarray) -> np.ndarray:
        """Random occlusion (simulating partial latent prints)."""
        if random.random() < 0.5:
            h, w = img.shape
            # 1-3 rectangular occlusions
            for _ in range(random.randint(1, 3)):
                occ_h = int(h * random.uniform(0.1, 0.4))
                occ_w = int(w * random.uniform(0.1, 0.4))
                top = random.randint(0, max(0, h - occ_h))
                left = random.randint(0, max(0, w - occ_w))
                color = random.choice([0, 255, 128])
                img[top:top + occ_h, left:left + occ_w] = color
        return img

    def _aug_photometric(self, img: np.ndarray) -> np.ndarray:
        """Contrast, brightness, and blur (dry/wet fingers, sensor artefacts)."""
        if random.random() < 0.5:
            # Contrast
            factor = random.uniform(0.5, 1.5)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        if random.random() < 0.4:
            # Brightness
            shift = random.uniform(-40, 40)
            img = np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)
        if random.random() < 0.3:
            # Gaussian blur
            k = random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def _aug_noise(self, img: np.ndarray) -> np.ndarray:
        """Additive Gaussian noise (sensor noise)."""
        if random.random() < 0.3:
            sigma = random.uniform(2, 15)
            noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img


def split_subjects(dataset: SOCOFingDataset,
                   train_count: int = 540
                   ) -> tuple[SOCOFingDataset, SOCOFingDataset]:
    """Split dataset by subject: first `train_count` for training, rest for val.

    Subject IDs in each split are re-mapped to contiguous 0..N-1 range.
    """
    all_subjects = sorted(set(s for _, s in dataset.images))
    val_count = len(all_subjects) - train_count
    if val_count <= 0:
        raise ValueError(f"Need >{train_count} subjects, have {len(all_subjects)}")

    train_set = set(all_subjects[:train_count])
    val_set = set(all_subjects[train_count:])

    def _filter(subj_set, remap):
        ds = SOCOFingDataset.__new__(SOCOFingDataset)
        ds.root = dataset.root
        ds.augment = False
        ds.images = [(p, remap[s]) for p, s in dataset.images
                     if s in subj_set]
        ds.subjects = [dataset.subjects[s] for s in sorted(subj_set)]
        ds.subject_to_id = {k: remap[v] for k, v in dataset.subject_to_id.items()
                            if v in subj_set}
        ds.num_classes = len(ds.subject_to_id)
        return ds

    train_map = {old: new for new, old in enumerate(sorted(train_set))}
    val_map = {old: new for new, old in enumerate(sorted(val_set))}

    return _filter(train_set, train_map), _filter(val_set, val_map)

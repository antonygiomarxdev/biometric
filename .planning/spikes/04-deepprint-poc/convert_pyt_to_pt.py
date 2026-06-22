"""Convert DeepPrint model from TF-Hub-style .pyt to standard PyTorch .pt.

The upstream distributes the model as a TF-Hub-style zip containing a
pickled PyTorch state_dict with CUDA tensors. Newer PyTorch versions
can't load this directly.

This script reads the .pyt once, materializes the tensors from the
data/ files (raw float32 blobs), and saves a clean .pt file that
torch.load() can read normally.

Strategy:
- Intercept _rebuild_tensor_v2 to capture (offset, size, stride) for
  each tensor BEFORE torch builds it
- Use the size to read the right slice of the storage's data
- Replace each tensor's content with the actual weights

After running this once, we never touch the .pyt again.
"""
from __future__ import annotations

import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


ZIP_PATH = Path("/home/ksante/Downloads/best_model.pyt")
OUT_PATH = Path("/home/ksante/Downloads/best_model.pt")


# Global: (tensor_id) -> (storage_key, offset, size, stride)
_tensor_meta: dict[int, tuple[str, int, tuple, tuple]] = {}
# Global: (storage_key) -> raw bytes
_storage_cache: dict[str, bytes] = {}


class _Storage:
    """Real FloatStorage of the right numel.

    We don't pre-fill it; we replace each tensor's data after
    unpickling using the captured (offset, size, stride) metadata.
    """

    def __init__(self, numel: int) -> None:
        self._untyped_storage = torch.FloatStorage(numel)
        self.dtype = torch.float32
        self._key: str = ""
        self._numel = numel

    def set_key(self, key: str) -> None:
        self._key = key


def _persistent_id_to_storage(pid: tuple) -> tuple[str, int]:
    return pid[2], pid[4]


def _read_storage(zip_file: zipfile.ZipFile, storage_key: str) -> bytes:
    """Read and cache the raw float32 blob for a given storage key."""
    if storage_key not in _storage_cache:
        _storage_cache[storage_key] = zip_file.read(f"model/data/{storage_key}")
    return _storage_cache[storage_key]


def main() -> None:
    if not ZIP_PATH.exists():
        sys.exit(f"ERROR: {ZIP_PATH} does not exist")

    print(f"Reading state_dict metadata from {ZIP_PATH}...")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        metadata_blob = zf.read("model/data.pkl")

        # Intercept _rebuild_tensor_v2 to capture metadata before building
        original_rebuild_v2 = torch._utils._rebuild_tensor_v2

        def _patched_rebuild_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
            tensor = original_rebuild_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata)
            if isinstance(storage, _Storage) and storage._key:
                _tensor_meta[id(tensor)] = (storage._key, storage_offset, tuple(size), tuple(stride))
            return tensor

        torch._utils._rebuild_tensor_v2 = _patched_rebuild_v2

        class _Unpickler(pickle.Unpickler):
            def persistent_load(self, pid):
                key, numel = _persistent_id_to_storage(pid)
                s = _Storage(numel)
                s.set_key(key)
                return s

        sd_meta = _Unpickler(io.BytesIO(metadata_blob)).load()
        torch._utils._rebuild_tensor_v2 = original_rebuild_v2

        if not isinstance(sd_meta, dict) or "model_state_dict" not in sd_meta:
            sys.exit("ERROR: unexpected metadata structure (no model_state_dict)")

        inner = sd_meta["model_state_dict"]
        print(f"Found {len(inner)} parameter entries, captured {len(_tensor_meta)} tensor metadata")

        print(f"Replacing tensor data from data/ files...")
        state_dict: dict[str, torch.Tensor] = {}
        skipped = 0
        for key, tensor in inner.items():
            meta = _tensor_meta.get(id(tensor))
            if meta is None:
                skipped += 1
                continue
            storage_key, offset, size, stride = meta

            raw_bytes = _read_storage(zf, storage_key)
            full_arr = np.frombuffer(raw_bytes, dtype=np.float32)
            expected_numel = int(np.prod(size))
            if full_arr.size < expected_numel:
                sys.exit(f"ERROR: storage {storage_key} too small for {key}")

            tensor_data = full_arr[:expected_numel].reshape(size).copy()
            new_tensor = torch.from_numpy(tensor_data)
            state_dict[key] = new_tensor

        print(f"Materialized {len(state_dict)} tensors ({skipped} skipped)")

        print(f"Saving clean .pt to {OUT_PATH}...")
        torch.save(state_dict, OUT_PATH)
        print(f"Done. File size: {OUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

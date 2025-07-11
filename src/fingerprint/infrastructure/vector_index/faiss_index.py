import os
import numpy as np
import faiss


class FingerprintVectorIndex:
    """FAISS-based vector index for fingerprint embeddings."""

    def __init__(self, index_path: str, dim: int):
        self.index_path = index_path
        self.dim = dim
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)

    def add(self, vector: np.ndarray) -> int:
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        idx = self.index.ntotal
        self.index.add(vector)
        faiss.write_index(self.index, self.index_path)
        return idx

    def search(self, vector: np.ndarray, k: int = 1):
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(vector, k)
        return indices[0].tolist(), distances[0].tolist()

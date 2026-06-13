"""
Hybrid Repository and Matching.
Clean Code: IMatcher and Repository implementation with Async/Batch support.
"""

from typing import List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.core.config import config
from src.core.interfaces import IMatcher
from src.core.metrics import timed
from src.core.types import MatchResult, NormalizedFingerprint
from src.processing.vectorizer import MinutiaeVectorizer
from src.storage.database import FingerprintRecord, db_manager
from src.storage.vector_index import VectorIndex, vector_index


class FingerprintRepository(IMatcher):
    """
    Repository implementing hybrid matching logic (L2 + Cosine).
    """

    def __init__(self, index: VectorIndex = vector_index):
        self.db_manager = db_manager
        self.vector_index = index
        # Weight configuration for combined score
        self.w_l2 = 0.7
        self.w_cos = 0.3
        self.threshold_l2 = 2000.0

    @timed("register_fingerprint")
    def register(
        self,
        fp: NormalizedFingerprint,
        person_id: str,
        name: str,
        doc: str,
        image_path: Optional[str] = None,
        minutiae_data: Optional[dict] = None,
    ) -> int:
        """Register a normalized fingerprint."""
        vector = fp.vector
        # Normalize vector to fixed dimension (256) for index compatibility
        vector = MinutiaeVectorizer.pad_vector(vector, config.vector_dimension)
        idx_id = self.vector_index.add(vector)

        session = self.db_manager.get_session()
        try:
            record = FingerprintRecord(
                person_id=person_id,
                name=name,
                document=doc,
                vector_index=idx_id,
                num_minutiae=len(fp.minutiae),
                image_path=image_path,
                minutiae_data=minutiae_data,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    async def match(self, probe: np.ndarray, top_k: int = 5) -> MatchResult:
        """Single asynchronous matching."""
        # In a real app we would use run_in_executor for blocking DB operations
        # For simplicity we call directly, but we prepare the async signature
        return self._match_sync(probe, top_k)

    async def match_batch(
        self, probes: List[np.ndarray], top_k: int = 5
    ) -> List[MatchResult]:
        """Batch matching."""
        # Parallelism simulation for batch
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self._match_sync, p, top_k) for p in probes]
        return await asyncio.gather(*tasks)

    def identify(self, fp: NormalizedFingerprint, top_k: int = 5) -> MatchResult:
        """Identify a fingerprint synchronously (Wrapper for compatibility)."""
        vector = fp.vector
        # Normalize vector to fixed dimension (256) for index compatibility
        vector = MinutiaeVectorizer.pad_vector(vector, config.vector_dimension)
        return self._match_sync(vector, top_k)

    def count(self) -> int:
        """Count the number of records in the database."""
        session = self.db_manager.get_session()
        try:
            return session.query(FingerprintRecord).count()
        finally:
            session.close()

    def _match_sync(self, probe: np.ndarray, top_k: int) -> MatchResult:
        # Normalize query vector to fixed dimension (256) for compatibility
        # Although vector_index.search also normalizes, doing it here is better for consistency
        if len(probe) != config.vector_dimension:
            probe = MinutiaeVectorizer.pad_vector(probe, config.vector_dimension)

        # 1. Fast index-based search (L2)
        # Fetch more candidates (2x) for re-ranking
        candidates_k = top_k * 2
        ids, l2_dists = self.vector_index.search(probe, k=candidates_k)

        if not ids:
            return self._empty_result()

        # 2. Re-ranking with hybrid metric
        best_res = None
        best_score = -1.0

        # Batch vector retrieval for Cosine computation
        candidates_vectors = self.vector_index.get_batch_by_ids(ids)

        for i, (idx, l2_dist) in enumerate(zip(ids, l2_dists)):
            # Compute normalized L2 score
            norm_l2 = l2_dist / self.threshold_l2
            score_l2 = np.exp(-norm_l2 * 2.0)

            # Real Cosine distance calculation
            cand_vec = candidates_vectors[i]

            cosine_dist = 1.0
            if cand_vec is not None:
                # Cosine distance = 1 - cosine_similarity
                # similarity = (A . B) / (||A|| ||B||)
                dot = np.dot(probe, cand_vec)
                norm_a = np.linalg.norm(probe)
                norm_b = np.linalg.norm(cand_vec)
                if norm_a > 0 and norm_b > 0:
                    sim = dot / (norm_a * norm_b)
                    cosine_dist = 1 - sim

            # Combined score (similarity, not distance)
            # Cosine similarity ranges from -1 to 1. Normalize to 0-1 (if vectors are positive)
            score_cos = 1.0 - cosine_dist  # Similarity

            combined = (self.w_l2 * score_l2) + (self.w_cos * score_cos)

            if combined > best_score:
                best_score = combined
                best_res = (idx, l2_dist, cosine_dist, score_l2, combined)

        if not best_res:
            return self._empty_result()

        idx, l2, cos, conf, comb = best_res

        # Retrieve metadata
        record = self.get_by_vector_index(idx)

        # Final decision based on threshold
        is_match = (
            l2 < self.threshold_l2
        )  # Hard L2 threshold remains the main guard

        return MatchResult(
            matched=is_match,
            person_id=record.person_id if record else None,
            score=comb,
            confidence=conf,
            l2_distance=l2,
            cosine_distance=cos,
            combined_score=comb,
            metadata={"name": record.name, "doc": record.document} if record else {},
        )

    def _empty_result(self) -> MatchResult:
        """Return an empty result when no match is found.

        Uses a very large numeric value instead of inf for JSON compatibility.
        """
        # Use a very large but finite value for JSON compatibility
        # 1e10 is large enough to represent "no match"
        MAX_DISTANCE = 1e10
        return MatchResult(
            matched=False,
            person_id=None,
            score=0.0,
            confidence=0.0,
            l2_distance=MAX_DISTANCE,
            cosine_distance=1.0,
            combined_score=0.0,
        )

    def get_by_person_id(self, person_id: str) -> Optional[FingerprintRecord]:
        """Retrieve a record by person ID."""
        session = self.db_manager.get_session()
        try:
            return (
                session.query(FingerprintRecord)
                .filter(FingerprintRecord.person_id == person_id)
                .first()
            )
        finally:
            session.close()

    def get_by_vector_index(self, idx: int) -> Optional[FingerprintRecord]:
        session = self.db_manager.get_session()
        try:
            return (
                session.query(FingerprintRecord)
                .filter(FingerprintRecord.vector_index == idx)
                .first()
            )
        finally:
            session.close()


# Global instance
repository = FingerprintRepository()

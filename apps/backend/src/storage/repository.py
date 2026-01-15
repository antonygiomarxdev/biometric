"""
Repositorio y Matching Híbrido.
Clean Code: Implementación de IMatcher y Repositorio con soporte Async/Batch.
"""

import asyncio
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
    Repositorio que implementa lógica de matching híbrido (L2 + Coseno).
    """

    def __init__(self, index: VectorIndex = vector_index):
        self.db_manager = db_manager
        self.vector_index = index
        # Configuración de pesos para score combinado
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
        """Registra una huella normalizada."""
        vector = fp.vector
        # Normalizar vector a dimensión fija (256) para compatibilidad con el índice
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
        """Matching unitario asíncrono."""
        # En una app real usaríamos run_in_executor para operaciones DB bloqueantes
        # Por simplicidad aquí llamamos directo, pero preparamos la firma async
        return self._match_sync(probe, top_k)

    async def match_batch(
        self, probes: List[np.ndarray], top_k: int = 5
    ) -> List[MatchResult]:
        """Matching batch."""
        # Simulación de paralelismo para batch
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self._match_sync, p, top_k) for p in probes]
        return await asyncio.gather(*tasks)

    def identify(self, fp: NormalizedFingerprint, top_k: int = 5) -> MatchResult:
        """Identifica una huella de forma síncrona (Wrapper para compatibilidad)."""
        vector = fp.vector
        # Normalizar vector a dimensión fija (256) para compatibilidad con el índice
        vector = MinutiaeVectorizer.pad_vector(vector, config.vector_dimension)
        return self._match_sync(vector, top_k)

    def count(self) -> int:
        """Cuenta el número de registros en la base de datos."""
        session = self.db_manager.get_session()
        try:
            return session.query(FingerprintRecord).count()
        finally:
            session.close()

    def _match_sync(self, probe: np.ndarray, top_k: int) -> MatchResult:
        # Normalizar vector de búsqueda a dimensión fija (256) para compatibilidad
        # Aunque vector_index.search también normaliza, es mejor hacerlo aquí para consistencia
        if len(probe) != config.vector_dimension:
            probe = MinutiaeVectorizer.pad_vector(probe, config.vector_dimension)

        # 1. Búsqueda rápida por índice (L2)
        # Traemos más candidatos (2x) para re-ranking
        candidates_k = top_k * 2
        ids, l2_dists = self.vector_index.search(probe, k=candidates_k)

        if not ids:
            return self._empty_result()

        # 2. Re-ranking con métrica híbrida
        best_res = None
        best_score = -1.0

        # Recuperación Batch de vectores para cálculo de Coseno
        candidates_vectors = self.vector_index.get_batch_by_ids(ids)

        for i, (idx, l2_dist) in enumerate(zip(ids, l2_dists)):
            # Calcular score L2 normalizado
            norm_l2 = l2_dist / self.threshold_l2
            score_l2 = np.exp(-norm_l2 * 2.0)

            # Cálculo de distancia Coseno Real
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

            # Score combinado (similitud, no distancia)
            # Cosine similarity va de -1 a 1. Normalizamos a 0-1 (si son vectores positivos)
            score_cos = 1.0 - cosine_dist  # Similitud

            combined = (self.w_l2 * score_l2) + (self.w_cos * score_cos)

            if combined > best_score:
                best_score = combined
                best_res = (idx, l2_dist, cosine_dist, score_l2, combined)

        if not best_res:
            return self._empty_result()

        idx, l2, cos, conf, comb = best_res

        # Recuperar metadatos
        record = self.get_by_vector_index(idx)

        # Decisión final basada en threshold
        is_match = (
            l2 < self.threshold_l2
        )  # Hard threshold en L2 sigue siendo la guardia principal

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
        """Retorna un resultado vacío cuando no se encuentra match.

        Usa un valor numérico muy grande en lugar de inf para compatibilidad JSON.
        """
        # Usar un valor muy grande pero finito para compatibilidad con JSON
        # 1e10 es suficientemente grande para representar "sin match"
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
        """Recupera un registro por ID de persona."""
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


# Instancia global
repository = FingerprintRepository()

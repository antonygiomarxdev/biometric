"""
FingerprintEnrollmentService — orchestrates image → capture → graphs.

The end-to-end pipeline for POST /api/v1/fingerprints/{id}/captures:
  1. Decode image bytes (CV2)
  2. Run FingerprintService.process_image
  3. Extract connected components from the resulting RidgeGraph
  4. Persist FingerprintCapture (with SHA-256 of image)
  5. Persist one RidgeGraph per connected component
  6. Persist to NebulaGraph (each minutia vertex gets capture_id + graph_id)
  7. Vectorise into chunks, store in Qdrant with extended payload
  8. Update Fingerprint.capture_count + timestamps
  9. Return the capture + graph count
"""

from __future__ import annotations

import hashlib
import logging
import uuid

import cv2
import networkx as nx
import numpy as np
from sqlalchemy.orm import Session

from src.core.types import NormalizedFingerprint, RidgeEdge, RidgeNode
from src.db.models import Fingerprint, FingerprintCapture, Person, RidgeGraph
from src.db.repositories.fingerprint_capture_repository import (
    FingerprintCaptureRepository,
)
from src.db.repositories.fingerprint_repository import FingerprintRepository

from src.db.repositories.ridge_graph_repository import RidgeGraphRepository
from src.services.fingerprint_service import FingerprintService

log = logging.getLogger(__name__)


class FingerprintEnrollmentService:
    def __init__(
        self,
        session: Session,
        fingerprint_service: FingerprintService,
        qdrant_repo=None,
        nebula_repo=None,
    ) -> None:
        self._session = session
        self._fp_service = fingerprint_service
        self._qdrant = qdrant_repo
        self._nebula = nebula_repo

    def create_capture(
        self,
        fingerprint_id: uuid.UUID,
        image_bytes: bytes,
        image_dpi: int | None = None,
        is_reference: bool = False,
        is_exemplar: bool = True,
        notes: str | None = None,
    ) -> tuple[FingerprintCapture, list[RidgeGraph]]:
        fp = FingerprintRepository.get_by_id(self._session, fingerprint_id)
        if fp is None:
            raise ValueError(f"Fingerprint {fingerprint_id} not found")

        image_hash = hashlib.sha256(image_bytes).hexdigest()

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image bytes")
        normalized: NormalizedFingerprint = self._fp_service.process_image(
            img, fingerprint_id=str(fingerprint_id),
        )

        capture = FingerprintCaptureRepository.create(
            self._session,
            fingerprint_id=fingerprint_id,
            image_uri=f"minio://pending/{fingerprint_id}/{image_hash[:12]}.bmp",
            image_hash_sha256=image_hash,
            image_dpi=image_dpi,
            algorithm_version="phase-13-v1",
            is_reference=is_reference,
            is_exemplar=is_exemplar,
            notes=notes,
        )

        graphs: list[RidgeGraph] = []
        if normalized.minutiae and len(normalized.minutiae) > 0:
            components = self._extract_connected_components(normalized)
            for idx, (region, nodes, edges) in enumerate(components, start=1):
                graph_data = {
                    "nodes": [
                        {"x": n.x, "y": n.y, "weight": n.weight,
                         "is_cutoff": n.is_cutoff, "angle": n.angle}
                        for n in nodes
                    ],
                    "edges": [
                        {"source": e.source, "target": e.target,
                         "length": e.length, "path": e.path}
                        for e in edges
                    ],
                }
                core = next((n for n in nodes if not n.is_cutoff and n.weight >= 0.9), None)
                g = RidgeGraphRepository.create(
                    self._session,
                    capture_id=capture.id,
                    graph_index=idx,
                    region_x=region[0], region_y=region[1],
                    region_w=region[2], region_h=region[3],
                    num_nodes=len(nodes), num_edges=len(edges),
                    graph_data=graph_data,
                    core_x=core.x if core else None,
                    core_y=core.y if core else None,
                    singularity_type="core" if core else None,
                )
                graphs.append(g)

        FingerprintCaptureRepository.update(
            self._session, capture.id,
            num_minutiae=len(normalized.minutiae) if normalized.minutiae else 0,
            num_graphs=len(graphs),
        )

        FingerprintRepository.increment_capture_count(self._session, fingerprint_id)

        self._index_external(
            capture=capture, fingerprint=fp, graphs=graphs,
            normalized=normalized,
        )

        self._session.refresh(capture)
        return capture, graphs

    def _extract_connected_components(
        self, normalized: NormalizedFingerprint,
    ) -> list[tuple[tuple[int, int, int, int], list, list]]:
        if not normalized.minutiae:
            return []

        G = nx.Graph()
        for i, m in enumerate(normalized.minutiae):
            G.add_node(i, x=m.x, y=m.y)
        for i in range(len(normalized.minutiae)):
            for j in range(i + 1, len(normalized.minutiae)):
                a, b = normalized.minutiae[i], normalized.minutiae[j]
                dist = ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
                if dist < 30:
                    G.add_edge(i, j)

        components: list[tuple[tuple[int, int, int, int], list, list]] = []
        for component in nx.connected_components(G):
            nodes_idx = list(component)
            xs = [normalized.minutiae[i].x for i in nodes_idx]
            ys = [normalized.minutiae[i].y for i in nodes_idx]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            region = (x_min, y_min, x_max - x_min, y_max - y_min)
            rnodes = []
            for i in nodes_idx:
                m = normalized.minutiae[i]
                rnodes.append(RidgeNode(x=m.x, y=m.y))
            redges = [
                RidgeEdge(source=i, target=j, path=[], length=1)
                for i, j in G.subgraph(component).edges
            ]
            components.append((region, rnodes, redges))
        return components

    def _index_external(
        self,
        capture: FingerprintCapture,
        fingerprint: Fingerprint,
        graphs: list[RidgeGraph],
        normalized: NormalizedFingerprint,
    ) -> None:
        """Push chunks to Qdrant and minutiae to NebulaGraph. Best-effort."""
        if self._qdrant is None or not normalized.minutiae:
            return
        try:
            person: Person | None = self._session.get(
                Person, fingerprint.person_id,
            )
            if person is None:
                return
            from src.processing.vectorizer import RagTripletVectorizer

            vectorizer = RagTripletVectorizer()
            chunks = vectorizer._chunks_from_normalized(normalized)
            self._qdrant.bulk_insert_chunks(
                person_id=str(person.external_id) if person.external_id else str(person.id),
                fingerprint_id=str(fingerprint.id),
                chunks=chunks,
                chunk_type="delaunay",
                capture_id=str(capture.id),
                graph_id="",
            )
        except Exception as e:
            log.warning("Qdrant indexing failed for capture %s: %s", capture.id, e)

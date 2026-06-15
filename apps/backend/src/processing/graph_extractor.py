from __future__ import annotations

import logging

import cv2
import networkx as nx
import numpy as np
import sknw
from skimage.morphology import skeletonize

from src.core.interfaces import IPipelineStep, PipelineContext
from src.core.types import RidgeEdge, RidgeGraph, RidgeNode

logger = logging.getLogger(__name__)


class RidgeGraphExtractor(IPipelineStep):
    def __init__(self, binary_threshold: int | None = None) -> None:
        self.binary_threshold = binary_threshold

    def process(self, ctx: PipelineContext) -> None:
        source = ctx.enhanced_image if ctx.enhanced_image is not None else ctx.raw_image

        if source.ndim == 3:
            source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        if self.binary_threshold is not None:
            _, binary = cv2.threshold(source, self.binary_threshold, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_bool = binary > 0

        white_pixels = int(binary_bool.sum())
        if white_pixels < 10:
            logger.warning("RidgeGraphExtractor: image too dark to extract skeleton")
            ctx.ridge_graph = RidgeGraph(nodes=[], edges=[])
            return

        skel_bool = skeletonize(binary_bool)
        skel = skel_bool.astype(np.uint8)

        skel_pixels = int(skel.sum())
        if skel_pixels < 5:
            logger.warning("RidgeGraphExtractor: skeleton too sparse")
            ctx.ridge_graph = RidgeGraph(nodes=[], edges=[])
            return

        nx_graph: nx.Graph = sknw.build_sknw(skel)

        nodes: list[RidgeNode] = []
        for nid in nx_graph.nodes:
            pts = nx_graph.nodes[nid].get("o", np.array([0, 0]))
            nodes.append(RidgeNode(x=int(pts[1]), y=int(pts[0])))

        edges: list[RidgeEdge] = []
        for u, v, data in nx_graph.edges(data=True):
            pts = data.get("pts", np.empty((0, 2), dtype=np.int16))
            path = [(int(p[1]), int(p[0])) for p in pts]
            length = int(data.get("weight", len(pts) - 1))
            edges.append(RidgeEdge(source=int(u), target=int(v), path=path, length=length))

        ctx.ridge_graph = RidgeGraph(nodes=nodes, edges=edges)
        logger.info(
            "RidgeGraphExtractor: %d nodes, %d edges from %d skeleton pixels",
            len(nodes), len(edges), skel_pixels,
        )

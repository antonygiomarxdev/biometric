from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import sknw

from src.core.interfaces import IPipelineStep, PipelineContext
from src.core.types import RidgeEdge, RidgeGraph, RidgeNode

logger = logging.getLogger(__name__)


class RidgeGraphExtractor(IPipelineStep):
    """
    Construye la topología de la huella a partir de un esqueleto pre-calculado.
    
    Toma `ctx.skeleton` y usa sknw para extraer un Grafo donde los Nodos
    son bifurcaciones/terminaciones y las Aristas son las crestas físicas.
    """
    def process(self, ctx: PipelineContext) -> None:
        if ctx.skeleton is None:
            logger.warning("RidgeGraphExtractor: ctx.skeleton is missing. SkeletonizationStep must run first.")
            ctx.ridge_graph = RidgeGraph(nodes=[], edges=[])
            return

        skel_pixels = int(ctx.skeleton.sum())
        if skel_pixels < 5:
            logger.warning("RidgeGraphExtractor: skeleton too sparse")
            ctx.ridge_graph = RidgeGraph(nodes=[], edges=[])
            return

        # sknw requiere un array con valores > 0
        nx_graph: nx.Graph = sknw.build_sknw(ctx.skeleton)

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

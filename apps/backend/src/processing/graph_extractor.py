from __future__ import annotations

import logging

import cv2
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
    Además, enriquece los nodos con pesos forenses (Gaussianos desde el Core)
    y detecta recortes artificiales del sensor (Cutoffs).
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

        # 1. Preparar métricas globales para Weights y Cutoffs
        ys, xs = np.where(ctx.skeleton > 0)
        
        # Convex Hull (la envoltura elástica matemática)
        points = np.column_stack((xs, ys)).astype(np.int32)
        hull = cv2.convexHull(points) if len(points) > 3 else None
        
        # Centro / Core
        if ctx.core is not None:
            cx, cy = ctx.core
        else:
            # Fallback al centroide de masa si no hay core detectado
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            
        # Campana de Gauss auto-escalable (cubre el 99% de la huella a 3 sigmas)
        h, w = ctx.skeleton.shape
        sigma = max(h, w) / 4.0

        # 2. Construir Grafo Base
        nx_graph: nx.Graph = sknw.build_sknw(ctx.skeleton)

        # 3. Mapear e hidratar Nodos (Weights + Cutoffs)
        nodes: list[RidgeNode] = []
        for nid in nx_graph.nodes:
            pts = nx_graph.nodes[nid].get("o", np.array([0, 0]))
            x, y = int(pts[1]), int(pts[0])
            
            # --- Forensic Weight (Gaussiana) ---
            dist_sq = (x - cx)**2 + (y - cy)**2
            weight = float(np.exp(-dist_sq / (2 * sigma**2)))
            weight = max(0.01, round(weight, 3))
            
            # --- Boundary Cutoff Detection ---
            is_cutoff = False
            # Las bifurcaciones no son cortes, solo las terminaciones (grado 1)
            if nx_graph.degree(nid) == 1 and hull is not None:
                # Distancia al límite matemático de la huella
                dist_to_hull = cv2.pointPolygonTest(hull, (x, y), measureDist=True)
                # Si está a menos de 5 píxeles del límite, es el final artificial del escáner
                if dist_to_hull < 5.0:
                    is_cutoff = True
            
            nodes.append(RidgeNode(x=x, y=y, weight=weight, is_cutoff=is_cutoff))

        # 4. Mapear Aristas
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

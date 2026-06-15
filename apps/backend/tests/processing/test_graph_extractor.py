from __future__ import annotations

import numpy as np
import pytest

from src.core.interfaces import PipelineContext
from src.core.types import RidgeEdge, RidgeGraph, RidgeNode


@pytest.fixture
def cross_skeleton() -> np.ndarray:
    img = np.zeros((50, 50), dtype=np.uint8)
    img[25, 10:40] = 255
    img[10:40, 25] = 255
    return img


@pytest.fixture
def blank_image() -> np.ndarray:
    return np.zeros((50, 50), dtype=np.uint8)


@pytest.fixture
def vertical_ridges() -> np.ndarray:
    img = np.zeros((50, 50), dtype=np.uint8)
    for x in range(5, 50, 7):
        img[:, x : x + 3] = 255
    return img


class TestRidgeGraphExtractor:
    def test_extract_cross_finds_center_junction(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=cross_skeleton)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5
        assert ctx.ridge_graph.num_edges >= 4

    def test_blank_image_returns_empty(self, blank_image: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=blank_image)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.is_empty()

    def test_vertical_ridges_produces_graph(self, vertical_ridges: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=vertical_ridges)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes > 0
        assert ctx.ridge_graph.num_edges > 0

    def test_nodes_have_correct_types(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=cross_skeleton)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        for node in ctx.ridge_graph.nodes:
            assert isinstance(node, RidgeNode)
            assert isinstance(node.x, int)
            assert isinstance(node.y, int)

    def test_edges_have_correct_types(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=cross_skeleton)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        for edge in ctx.ridge_graph.edges:
            assert isinstance(edge, RidgeEdge)
            assert isinstance(edge.source, int)
            assert isinstance(edge.target, int)
            assert isinstance(edge.path, list)
            assert len(edge.path) > 0
            assert isinstance(edge.length, int)
            assert edge.length > 0

    def test_edge_path_contains_ridge_points(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=cross_skeleton)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        for edge in ctx.ridge_graph.edges:
            for px, py in edge.path:
                assert cross_skeleton[py, px] > 0

    def test_uses_enhanced_image_when_available(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(
            raw_image=np.zeros((50, 50), dtype=np.uint8),
            enhanced_image=cross_skeleton,
        )
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5

    def test_custom_threshold(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        extractor = RidgeGraphExtractor(binary_threshold=127)
        ctx = PipelineContext(raw_image=cross_skeleton)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5

    def test_3channel_image(self) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor

        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        rgb[25, 10:40, :] = 255
        rgb[10:40, 25, :] = 255

        extractor = RidgeGraphExtractor()
        ctx = PipelineContext(raw_image=rgb)
        extractor.process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5

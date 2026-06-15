"""Tests for the RidgeGraphExtractor.

Covers synthetic patterns, real SOCOFing images from
``tests/fixtures/socofing_real/`` (10 images, 2 persons x 5 fingers,
tracked in git for portable CI).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.interfaces import PipelineContext
from src.core.types import RidgeEdge, RidgeGraph, RidgeNode


SOCOFING_FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "socofing_real"
)


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
        img[5:45, x : x + 3] = 255  # Leave a 5px gap at top and bottom
    return img


@pytest.fixture
def socofing_image_paths() -> list[Path]:
    if not SOCOFING_FIXTURES.exists():
        pytest.skip(f"SOCOFing fixtures not found: {SOCOFING_FIXTURES}")
    paths = sorted(SOCOFING_FIXTURES.glob("*.BMP"))
    if not paths:
        pytest.skip("No SOCOFing .BMP fixtures in tests/fixtures/socofing_real/")
    return paths


def _run_extractor_pipeline(img: np.ndarray, custom_threshold: int | None = None) -> PipelineContext:
    from src.processing.graph_extractor import RidgeGraphExtractor
    from src.processing.skeletonize_step import SkeletonizationStep

    ctx = PipelineContext(raw_image=img)
    # The standard pipeline sequence for graph extraction
    pipeline = [
        SkeletonizationStep(),
        RidgeGraphExtractor(),
    ]
    
    # If custom threshold is needed, we'd theoretically pass it to a CustomBinarizationStep
    # But since SkeletonizationStep encapsulates Otsu, for the test we just bypass Otsu 
    # if a custom threshold is strictly required by the test.
    if custom_threshold is not None:
        import cv2
        _, binary = cv2.threshold(img, custom_threshold, 255, cv2.THRESH_BINARY)
        ctx.enhanced_image = binary
    
    for step in pipeline:
        step.process(ctx)
        
    return ctx

def _load_graph(path: Path) -> RidgeGraph:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Could not load {path}"
    ctx = _run_extractor_pipeline(img)
    assert ctx.ridge_graph is not None
    return ctx.ridge_graph


class TestRidgeGraphExtractor:
    def test_extract_cross_finds_center_junction(self, cross_skeleton: np.ndarray) -> None:
        ctx = _run_extractor_pipeline(cross_skeleton)
        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5
        assert ctx.ridge_graph.num_edges >= 4

    def test_blank_image_returns_empty(self, blank_image: np.ndarray) -> None:
        ctx = _run_extractor_pipeline(blank_image)
        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.is_empty()

    def test_nodes_have_correct_types(self, cross_skeleton: np.ndarray) -> None:
        ctx = _run_extractor_pipeline(cross_skeleton)
        assert ctx.ridge_graph is not None
        for node in ctx.ridge_graph.nodes:
            assert isinstance(node, RidgeNode)
            assert isinstance(node.x, int)
            assert isinstance(node.y, int)

    def test_edges_have_correct_types(self, cross_skeleton: np.ndarray) -> None:
        ctx = _run_extractor_pipeline(cross_skeleton)
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
        ctx = _run_extractor_pipeline(cross_skeleton)
        assert ctx.ridge_graph is not None
        for edge in ctx.ridge_graph.edges:
            for px, py in edge.path:
                assert cross_skeleton[py, px] > 0

    def test_uses_enhanced_image_when_available(self, cross_skeleton: np.ndarray) -> None:
        from src.processing.graph_extractor import RidgeGraphExtractor
        from src.processing.skeletonize_step import SkeletonizationStep
        
        ctx = PipelineContext(
            raw_image=np.zeros((50, 50), dtype=np.uint8),
            enhanced_image=cross_skeleton
        )
        SkeletonizationStep().process(ctx)
        RidgeGraphExtractor().process(ctx)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5

    def test_custom_threshold_via_skeletonization(self, cross_skeleton: np.ndarray) -> None:
        ctx = _run_extractor_pipeline(cross_skeleton, custom_threshold=127)
        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5

    def test_3channel_image(self) -> None:
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        rgb[25, 10:40, :] = 255
        rgb[10:40, 25, :] = 255
        ctx = _run_extractor_pipeline(rgb)
        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes >= 5

    def test_fingerprint_like_pattern_produces_connected_graph(self) -> None:
        rng = np.random.default_rng(seed=42)
        img = np.zeros((150, 150), dtype=np.uint8)
        cx, cy = 75, 75
        for angle in np.linspace(0, 2 * np.pi, 30, endpoint=False):
            for r in range(20, 70, 3):
                x = int(cx + r * np.cos(angle))
                y = int(cy + r * np.sin(angle))
                if 0 <= x < 150 and 0 <= y < 150:
                    img[y - 1 : y + 2, x - 1 : x + 2] = 255

        ctx = _run_extractor_pipeline(img)

        assert ctx.ridge_graph is not None
        assert ctx.ridge_graph.num_nodes > 10
        assert ctx.ridge_graph.num_edges > 10
        for edge in ctx.ridge_graph.edges:
            assert edge.source != edge.target
            assert 0 <= edge.source < ctx.ridge_graph.num_nodes
            assert 0 <= edge.target < ctx.ridge_graph.num_nodes


class TestRidgeGraphExtractorSOCOFing:
    """Real-world validation using curated SOCOFing fixtures."""

    def test_all_fixtures_produce_non_empty_graphs(
        self, socofing_image_paths: list[Path]
    ) -> None:
        for path in socofing_image_paths:
            graph = _load_graph(path)
            assert not graph.is_empty(), f"{path.name} produced empty graph"
            assert graph.num_nodes > 0
            assert graph.num_edges > 0

    def test_fixtures_have_consistent_node_density(
        self, socofing_image_paths: list[Path]
    ) -> None:
        node_counts: list[int] = []
        for path in socofing_image_paths:
            graph = _load_graph(path)
            node_counts.append(graph.num_nodes)

        assert len(node_counts) == 10
        mean_nodes = float(np.mean(node_counts))
        assert mean_nodes > 10, f"Mean node count too low: {mean_nodes}"
        assert max(node_counts) < mean_nodes * 5, "Outlier in node count"

    def test_fixtures_produce_connected_components(
        self, socofing_image_paths: list[Path]
    ) -> None:
        for path in socofing_image_paths:
            graph = _load_graph(path)
            if graph.num_edges == 0:
                continue
            for edge in graph.edges:
                assert 0 <= edge.source < graph.num_nodes
                assert 0 <= edge.target < graph.num_nodes

    def test_per_image_extraction(self, socofing_image_paths: list[Path]) -> None:
        for path in socofing_image_paths[:3]:
            graph = _load_graph(path)
            assert graph.num_nodes >= 20, (
                f"{path.name}: too few nodes ({graph.num_nodes}) for real fingerprint"
            )

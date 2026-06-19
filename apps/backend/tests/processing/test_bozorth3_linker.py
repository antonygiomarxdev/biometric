"""Unit tests for the NBIS Bozorth3-style pair linker."""

from src.processing.bozorth3_linker import Bozorth3Linker


def _make_probe_pair(
    i: int,
    j: int,
    mi_x: float,
    mi_y: float,
    mi_angle: float,
    mj_x: float,
    mj_y: float,
    mj_angle: float,
) -> dict:
    return {
        "i": i,
        "j": j,
        "mi_x": mi_x,
        "mi_y": mi_y,
        "mi_angle": mi_angle,
        "mj_x": mj_x,
        "mj_y": mj_y,
        "mj_angle": mj_angle,
        "dx": mj_x - mi_x,
        "dy": mj_y - mi_y,
        "dtheta": mj_angle - mi_angle,
        "distance": ((mj_x - mi_x) ** 2 + (mj_y - mi_y) ** 2) ** 0.5,
        "type_pair": 0,
    }


def _make_hit(
    query_pair_index: int,
    person_id: str,
    mi_x: float,
    mi_y: float,
    mi_angle: float,
    mj_x: float,
    mj_y: float,
    mj_angle: float,
    similarity: float = 0.9,
) -> dict:
    return {
        "query_pair_index": query_pair_index,
        "person_id": person_id,
        "similarity": similarity,
        "mi_x": mi_x,
        "mi_y": mi_y,
        "mi_angle": mi_angle,
        "mj_x": mj_x,
        "mj_y": mj_y,
        "mj_angle": mj_angle,
        "fingerprint_id": "fp1",
        "capture_id": "cap1",
    }


def _transform(dx: float = 0.0, dy: float = 0.0, dtheta: float = 0.0) -> tuple[float, float, float]:
    """Shorthand for expected transformation."""
    return (dx, dy, dtheta)


class TestBozorth3Linker:
    """Tests for Bozorth3Linker with synthetic pair data."""

    def test_empty_inputs(self) -> None:
        """Empty probe pairs or hits returns empty list."""
        linker = Bozorth3Linker()
        assert linker.link([], []) == []
        assert linker.link([_make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1)], []) == []

    def test_single_hit(self) -> None:
        """Single hit per person should produce a singleton component."""
        linker = Bozorth3Linker()
        probe = [_make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1)]
        hits = [
            _make_hit(0, "person_A", 0.15, 0.12, 0.05, 0.35, 0.32, 0.15),
        ]
        results = linker.link(probe, hits)
        assert len(results) == 1
        assert results[0]["person_id"] == "person_A"
        assert results[0]["validated_count"] == 1
        # Default saturation=30 → score = 1/30 ≈ 0.0333
        assert results[0]["score"] == round(1 / 30, 4)

    def test_two_compatible_hits(self) -> None:
        """Two hits with similar transformations should be linked."""
        linker = Bozorth3Linker(dx_tol=0.05, dy_tol=0.05, dtheta_tol=0.3)
        # Two probe pairs at different positions
        probe = [
            _make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1),
            _make_probe_pair(2, 3, 0.5, 0.5, 0.2, 0.7, 0.7, 0.3),
        ]
        # Both map to person_A with a similar shift (+0.05, +0.02, +0.05 rad)
        hits = [
            _make_hit(0, "person_A", 0.15, 0.12, 0.05, 0.35, 0.32, 0.15),
            _make_hit(1, "person_A", 0.55, 0.52, 0.25, 0.75, 0.72, 0.35),
        ]
        results = linker.link(probe, hits)
        assert len(results) == 1
        assert results[0]["validated_count"] == 2

    def test_incompatible_hits_separate_components(self) -> None:
        """Two hits with very different transformations should NOT link."""
        linker = Bozorth3Linker(dx_tol=0.02, dy_tol=0.02, dtheta_tol=0.1)
        probe = [
            _make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1),
            _make_probe_pair(2, 3, 0.5, 0.5, 0.2, 0.7, 0.7, 0.3),
        ]
        hits = [
            # Small shift
            _make_hit(0, "person_A", 0.11, 0.11, 0.01, 0.31, 0.31, 0.11),
            # Huge shift (0.4, 0.4) — way outside 0.02 tolerance
            _make_hit(1, "person_A", 0.9, 0.9, 0.6, 1.1, 1.1, 0.7),
        ]
        results = linker.link(probe, hits)
        assert len(results) == 1
        # Two components of size 1 each (no linking), largest = 1
        assert results[0]["validated_count"] == 1

    def test_two_candidates_ranked(self) -> None:
        """Multiple candidates are ranked by component size then score."""
        linker = Bozorth3Linker(saturation=10)
        probe = [
            _make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1),
            _make_probe_pair(2, 3, 0.5, 0.5, 0.0, 0.7, 0.7, 0.1),
            _make_probe_pair(4, 5, 0.2, 0.8, 0.0, 0.4, 1.0, 0.1),
        ]
        hits = [
            # person_A: 2 compatible hits (score = 2/10 = 0.2)
            _make_hit(0, "person_A", 0.15, 0.12, 0.02, 0.35, 0.32, 0.12),
            _make_hit(1, "person_A", 0.55, 0.52, 0.02, 0.75, 0.72, 0.12),
            # person_B: 1 hit only (score = 1/10 = 0.1)
            _make_hit(2, "person_B", 0.25, 0.85, 0.03, 0.45, 1.05, 0.13),
        ]
        results = linker.link(probe, hits)
        assert len(results) == 2
        assert results[0]["person_id"] == "person_A"
        assert results[0]["score"] == 0.2
        assert results[1]["person_id"] == "person_B"
        assert results[1]["score"] == 0.1

    def test_top_k_limits_results(self) -> None:
        """top_k parameter limits the number of returned candidates."""
        linker = Bozorth3Linker()
        probe = [_make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1)]
        hits = [
            _make_hit(0, "person_A", 0.15, 0.12, 0.05, 0.35, 0.32, 0.15),
            _make_hit(0, "person_B", 0.12, 0.11, 0.03, 0.32, 0.31, 0.13),
            _make_hit(0, "person_C", 0.18, 0.14, 0.06, 0.38, 0.34, 0.16),
        ]
        results = linker.link(probe, hits, top_k=2)
        assert len(results) == 2

    def test_component_hits_sorted_by_similarity(self) -> None:
        """Supporting pairs within the largest component are sorted by similarity desc."""
        linker = Bozorth3Linker(dx_tol=0.05, dy_tol=0.05, dtheta_tol=0.3)
        probe = [
            _make_probe_pair(0, 1, 0.1, 0.1, 0.0, 0.3, 0.3, 0.1),
            _make_probe_pair(2, 3, 0.5, 0.5, 0.2, 0.7, 0.7, 0.3),
            _make_probe_pair(4, 5, 0.2, 0.8, 0.0, 0.4, 1.0, 0.1),
        ]
        hits = [
            _make_hit(0, "person_A", 0.15, 0.12, 0.05, 0.35, 0.32, 0.15, similarity=0.7),
            _make_hit(1, "person_A", 0.55, 0.52, 0.25, 0.75, 0.72, 0.35, similarity=0.9),
            _make_hit(2, "person_A", 0.25, 0.82, 0.05, 0.45, 1.02, 0.15, similarity=0.8),
        ]
        results = linker.link(probe, hits)
        assert len(results) == 1
        assert results[0]["validated_count"] == 3
        # Supporting pairs should be sorted by similarity desc
        sp = results[0]["supporting_pairs"]
        assert len(sp) == 3
        assert sp[0]["similarity"] == 0.9
        assert sp[1]["similarity"] == 0.8
        assert sp[2]["similarity"] == 0.7

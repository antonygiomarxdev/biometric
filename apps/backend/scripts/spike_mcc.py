"""
MCC v4 — ridge flow structure per cylinder cell (not skeleton pixels).

Features per cell: mean orientation, frequency, dominant angle.
Rotation-invariant because cylinder aligned to minutia's orientation.
Scale-normalized because ridge frequency is relative to image size.
"""

import cv2, math, random, sys, time, numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from src.core.interfaces import PipelineContext
from src.processing.enhancer import create_enhancer
from src.processing.gabor import QualityMaskStep
from src.processing.pre_hooks import OrientationFieldAnalyzer, SingularityDetector
from src.processing.skeletonize_step import SkeletonizationStep
from src.processing.spurious_filter import SkeletonCleanerStep
from src.processing.graph_extractor import RidgeGraphExtractor

SOCOF = Path(__file__).resolve().parents[1] / "static" / "SOCOFing" / "Real"
SECTORS, RINGS = 12, 3
FEATURES_PER_CELL = 3  # mean_orient, ridge_count, mean_freq
DIM = SECTORS * RINGS * FEATURES_PER_CELL  # 108D
COL = "mcc_flow"
N = 10

random.seed(42); np.random.seed(42)


def pipeline(img: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray | None, np.ndarray | None]:
    ctx = PipelineContext(raw_image=img, fingerprint_id="x")
    enh = create_enhancer()
    enhanced = enh.enhance(img, resize=True)
    ctx.enhanced_image = enhanced
    ctx.preprocessed_image = enhanced

    OrientationFieldAnalyzer().process(ctx)
    QualityMaskStep().process(ctx)
    orient = ctx.orientation_field
    freq = ctx.freq_image

    SingularityDetector(roi_radius=140).process(ctx)
    SkeletonizationStep(min_island_size=20).process(ctx)
    SkeletonCleanerStep().process(ctx)
    RidgeGraphExtractor().process(ctx)

    rg = ctx.ridge_graph
    if rg is None:
        return [], np.zeros((1,1)), None, None
    nodes = [{"x": float(n.x), "y": float(n.y), "angle": float(n.angle)} for n in rg.nodes]
    skel = ctx.skeleton
    return nodes, skel if skel is not None else np.zeros((1,1)), orient, freq


def _count_ridges(skeleton: np.ndarray, points: list[tuple[float, float]]) -> int:
    """Count ridge crossings along a line defined by points. Robust to stretching."""
    if len(points) < 2:
        return 0
    prev = 0
    crossings = 0
    # Sample points along the path and count transitions
    for px, py in points:
        x, y = int(px), int(py)
        if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1]:
            val = 1 if skeleton[y, x] > 0 else 0
            if val != prev:
                crossings += val
                prev = val
    return crossings


def cylinders(nodes: list[dict], skeleton: np.ndarray, orient: np.ndarray | None,
              freq: np.ndarray | None) -> list[list[float]]:
    """Build MCC cylinders capturing RIDGE STRUCTURE (not pixel density).

    Per cell: [dominant_orientation, ridge_crossings, mean_frequency].
    All aligned to minutia orientation → rotation-invariant.
    Ridge crossings via skeleton → stretch-invariant.
    """
    rr = [25, 55, 95]
    h, w = skeleton.shape

    if orient is None or freq is None or skeleton.sum() == 0:
        return [[0.0] * DIM] * len(nodes)

    descs = []
    for m in nodes:
        mx, my, ma = m["x"], m["y"], m["angle"]
        features = []

        # Block scaling: orientation field is at lower resolution
        orient_scale_y = orient.shape[0] / h
        orient_scale_x = orient.shape[1] / w
        freq_scale_y = freq.shape[0] / h if freq is not None else 0
        freq_scale_x = freq.shape[1] / w if freq is not None else 0

        for ri in range(RINGS):
            inner_r = rr[ri] * 0.7
            outer_r = rr[ri] * 1.3
            for si in range(SECTORS):
                # Sample points in this angular sector + radial ring
                sector_center = (si + 0.5) * 2 * math.pi / SECTORS
                sample_r = (inner_r + outer_r) / 2

                # World coordinates of the sampling point
                world_angle = ma + sector_center  # Add minutia orientation
                sx = mx + sample_r * math.cos(world_angle)
                sy = my + sample_r * math.sin(world_angle)
                six, siy = int(sx), int(sy)

                # 1. Ridge count: trace from minutia → sampling point
                num_steps = max(1, int(sample_r / 3))
                path = [(mx + (sx - mx) * t / num_steps, my + (sy - my) * t / num_steps)
                        for t in range(num_steps + 1)]
                n_ridges = _count_ridges(skeleton, path)

                # 2. Dominant orientation (block-level → map pixel coords)
                oby = min(int(sy * orient_scale_y), orient.shape[0] - 1)
                obx = min(int(sx * orient_scale_x), orient.shape[1] - 1)
                dom_orient = float(orient[oby, obx])
                rel_orient = (dom_orient - ma + math.pi) % (2 * math.pi)

                # 3. Ridge spacing (frequency)
                if freq is not None:
                    fby = min(int(sy * freq_scale_y), freq.shape[0] - 1)
                    fbx = min(int(sx * freq_scale_x), freq.shape[1] - 1)
                    ridge_spacing = float(freq[fby, fbx])
                    if ridge_spacing <= 0:
                        ridge_spacing = 0.0
                else:
                    ridge_spacing = 0.0

                features += [rel_orient / (2 * math.pi), min(n_ridges, 10) / 10.0,
                             min(ridge_spacing, 1.0)]

        # L2 normalize
        vec = np.array(features, dtype=np.float32)
        norm = np.sqrt(np.sum(vec ** 2)) + 1e-10
        descs.append((vec / norm).tolist())
    return descs


def run():
    imgs = sorted(SOCOF.glob("*.BMP"))
    selected = random.sample(imgs, N * 2)

    client = QdrantClient(host="localhost", port=6333)
    try:
        client.delete_collection(COL)
    except Exception:
        pass
    client.create_collection(COL, vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))

    enrolled = []
    pid = 0
    t0 = time.monotonic()
    for p in selected:
        if len(enrolled) >= N:
            break
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        nodes, skel, orient, freq2 = pipeline(img)
        if len(nodes) < 8:
            continue
        descs = cylinders(nodes, skel, orient, freq2)
        if len(descs) < 5:
            continue
        client.upsert(COL, points=[qm.PointStruct(id=pid + i, vector=d, payload={"fp": p.stem}) for i, d in enumerate(descs)])
        pid += len(descs)
        enrolled.append((p.stem, img))
        print(f"  [{len(enrolled)}/{N}] {p.stem}: {len(descs)} descs ({time.monotonic() - t0:.0f}s)")
    print(f"\nEnrolled {len(enrolled)} prints ({time.monotonic() - t0:.0f}s)")

    print(f"\n{'Minutiae':>8s}  {'Rank-1':>7s}  {'Rank-5':>7s}  {'Rank-10':>7s}  {'AvgTime':>8s}")
    print("-" * 48)

    for n_minutiae in [3, 5, 8, 15]:
        hits = {1: 0, 5: 0, 10: 0}
        t_tot = 0.0
        total = 0
        for fid, fimg in enrolled:
            nodes, skel, orient, freq2 = pipeline(fimg)
            if len(nodes) < n_minutiae:
                continue
            # Take a random subset of minutiae (simulates partial latent)
            idx = random.sample(range(len(nodes)), n_minutiae)
            # Add noise to simulate real-world distortion
            perturbed = [{"x": nodes[i]["x"] + random.gauss(0, 3),
                          "y": nodes[i]["y"] + random.gauss(0, 3),
                          "angle": (nodes[i]["angle"] + random.gauss(0, 0.1)) % (2 * math.pi)}
                         for i in idx]
            pdescs = cylinders(perturbed, skel, orient, freq2)
            if len(pdescs) < 3:
                continue

            t1 = time.monotonic()
            scores = defaultdict(float)
            for d in pdescs:
                for hit in client.query_points(collection_name=COL, query=d, limit=5, with_payload=True).points:
                    fid2 = (hit.payload or {}).get("fp", "")
                    if fid2:
                        scores[fid2] += float(hit.score)
            ranked = sorted(scores, key=lambda k: scores[k], reverse=True)
            t_tot += time.monotonic() - t1
            try:
                rank = ranked.index(fid) + 1
            except ValueError:
                rank = -1
            for k in hits:
                if 0 < rank <= k:
                    hits[k] += 1
            total += 1
        avg = t_tot / max(total, 1) * 1000
        print(f"  {n_minutiae:>8d}  {hits[1] / max(total, 1) * 100:6.1f}%  {hits[5] / max(total, 1) * 100:6.1f}%  {hits[10] / max(total, 1) * 100:6.1f}%  {avg:6.0f}ms")


if __name__ == "__main__":
    run()

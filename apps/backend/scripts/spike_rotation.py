"""Test rotation invariance with canvas expansion (no cropping)."""
import math, cv2, numpy as np, sys, time, random
from collections import defaultdict
sys.path.insert(0, '.')
from qdrant_client import QdrantClient
from scripts.spike_mcc import pipeline, cylinders

COL = 'mcc_flow'
client = QdrantClient(host='localhost', port=6333)
scroll = client.scroll(COL, limit=5000, with_payload=True)

cyl_counts = defaultdict(int)
for p in scroll[0]:
    fid = (p.payload or {}).get('fp', '')
    if fid: cyl_counts[fid] += 1

target = scroll[0][0].payload.get('fp')
print(f'Target: {target}')
print('Caching minutiae...')

enrolled = {}
for fid in cyl_counts:
    nodes, _, _, _ = pipeline(cv2.imread(f'static/SOCOFing/Real/{fid}.BMP', 0))
    enrolled[fid] = nodes
    print(f'  {fid}: {len(nodes)}')

def rot_img(img, deg):
    h, w = img.shape
    th = math.radians(deg)
    ct, st = abs(math.cos(th)), abs(math.sin(th))
    nw, nh = int(w * ct + h * st), int(w * st + h * ct)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    M[0, 2] += (nw / 2 - w / 2)
    M[1, 2] += (nh / 2 - h / 2)
    return cv2.warpAffine(img, M, (nw, nh), borderValue=255)

def coarse(descs):
    sc = defaultdict(float)
    for d in descs:
        for h in client.query_points(collection_name=COL, query=d, limit=5, with_payload=True).points:
            f = (h.payload or {}).get('fp', '')
            if f and cyl_counts.get(f, 0) > 0:
                sc[f] += float(h.score) / cyl_counts[f]
    return sorted(sc, key=lambda k: sc[k], reverse=True)

def verify(pn, cn):
    if len(pn) < 2 or len(cn) < 2:
        return 0.0, 0
    pp = np.array([[n['x'], n['y']] for n in pn])
    cp = np.array([[n['x'], n['y']] for n in cn])
    best = 0
    for _ in range(min(300, len(pp) * len(cp))):
        pi = random.randrange(len(pp))
        ci = np.argmin(np.hypot(cp[:, 0] - pp[pi, 0], cp[:, 1] - pp[pi, 1]))
        pi2 = random.randrange(len(pp))
        if pi2 == pi: continue
        ci2 = np.argmin(np.hypot(cp[:, 0] - pp[pi2, 0], cp[:, 1] - pp[pi2, 1]))
        th = math.atan2(cp[ci2, 1] - cp[ci, 1], cp[ci2, 0] - cp[ci, 0]) - math.atan2(pp[pi2, 1] - pp[pi, 1], pp[pi2, 0] - pp[pi, 0])
        ct, st = math.cos(th), math.sin(th)
        cd = math.hypot(cp[ci2, 0] - cp[ci, 0], cp[ci2, 1] - cp[ci, 1])
        pd = math.hypot(pp[pi2, 0] - pp[pi, 0], pp[pi2, 1] - pp[pi, 1])
        s = cd / max(pd, 1e-10)
        rx, ry = pp[pi, 0] * ct * s - pp[pi, 1] * st * s, pp[pi, 0] * st * s + pp[pi, 1] * ct * s
        tx, ty = cp[ci, 0] - rx, cp[ci, 1] - ry
        il = 0
        for p2 in range(len(pp)):
            rpx = pp[p2, 0] * ct * s - pp[p2, 1] * st * s + tx
            rpy = pp[p2, 0] * st * s + pp[p2, 1] * ct * s + ty
            if np.min(np.hypot(cp[:, 0] - rpx, cp[:, 1] - rpy)) < 15.0:
                il += 1
        if il > best:
            best = il
    return best / max(len(pp), 1), best

print(f'\n{"Angle":>6s} {"Orig":>8s} {"RotSize":>8s} {"EnhSz":>10s} {"Nod":>4s} {"Coarse":>7s} {"Verify":>7s} {"In/To":>7s}')
print('-' * 68)
for rot in [0, 30, 45, 60, 90, 120, 135, 150, 180]:
    img = cv2.imread(f'static/SOCOFing/Real/{target}.BMP', 0)
    rim = rot_img(img, rot)
    nodes, skel, orient, freq = pipeline(rim)
    descs = cylinders(nodes, skel, orient, freq)
    ranked = coarse(descs)
    cr = (ranked.index(target) + 1) if target in ranked else -1
    vr = -1
    il = 0
    for fid in ranked[:5]:
        cn = enrolled.get(fid, [])
        ratio, il2 = verify(nodes, cn)
        if fid == target:
            vr = len([1 for f2 in ranked[:5] if f2 != fid]) + 1
            il = il2
    if vr == -1: vr = 99
    print(f'  {rot:>5d}° {str(img.shape):>8s} {str(rim.shape):>8s} {str(skel.shape):>10s} {len(nodes):>4d} {cr:>7d} {vr:>7d} {il:>3d}/{len(nodes)}')

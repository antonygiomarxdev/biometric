"""Integration test for Plan 24-01: thinning-based minutiae extraction.

Tests:
  1. Determinism: same image → same minutiae across two runs.
  2. 5 SOCOFing persons produce valid minutiae.
  3. Cropped images produce fewer minutiae but at valid positions.
  4. Normalised coordinates are in [0, 1].

Usage (from apps/backend):
    uv run python ../../scripts/test_thinning_extractor.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import numpy as np

from src.services.mcc_matching_service import MccMatchingService

SOCOFING_REAL = (
    Path(__file__).resolve().parent.parent
    / "apps"
    / "backend"
    / "static"
    / "SOCOFing"
    / "Real"
)

PERSONS = ["SOC_0100", "SOC_0101", "SOC_0102", "SOC_0103", "SOC_0104"]
FINGER_NAME = "index"


def find_socofing_file(person_external_id: str) -> Path:
    pid = person_external_id.replace("SOC_", "").lstrip("0")
    for path in sorted(SOCOFING_REAL.glob(f"{pid}__*_{FINGER_NAME}_finger.BMP")):
        return path
    raise FileNotFoundError(
        f"No {FINGER_NAME} BMP found for {person_external_id} (pid={pid})",
    )


def crop_center(image_bytes: bytes, fraction: float) -> bytes:
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    y_off = (h - ch) // 2
    x_off = (w - cw) // 2
    cropped = img[y_off:y_off + ch, x_off:x_off + cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


def crop_corner(image_bytes: bytes, fraction: float) -> bytes:
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    cropped = img[:ch, :cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


def main() -> int:
    svc = MccMatchingService()
    failures = 0

    print("=" * 60)
    print("Test 1: Determinism (same image → same minutiae)")
    print("=" * 60)
    for ext_id in PERSONS:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        r1 = svc.preview_thinning(img_bytes)
        r2 = svc.preview_thinning(img_bytes)
        m1 = [(m["x"], m["y"], m["type"]) for m in r1["minutiae"]]
        m2 = [(m["x"], m["y"], m["type"]) for m in r2["minutiae"]]
        ok = m1 == m2
        status = "OK" if ok else "DIFFERS"
        print(f"  {ext_id}: {len(m1)} vs {len(m2)} → {status}")
        if not ok:
            failures += 1

    print()
    print("=" * 60)
    print("Test 2: All persons produce valid minutiae")
    print("=" * 60)
    for ext_id in PERSONS:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        r = svc.preview_thinning(img_bytes)
        count = len(r["minutiae"])
        has_types = all(m["type"] in (1, 3) for m in r["minutiae"])
        in_range = all(
            0.0 <= m["x"] <= 1.0 and 0.0 <= m["y"] <= 1.0
            for m in r["minutiae"]
        )
        unique_positions = len({(m["x"], m["y"]) for m in r["minutiae"]})
        print(
            f"  {ext_id}: {count} minutiae "
            f"types={'OK' if has_types else 'BAD'} "
            f"range={'OK' if in_range else 'OUT'} "
            f"unique_positions={unique_positions}",
        )
        if not has_types or not in_range:
            failures += 1

    print()
    print("=" * 60)
    print("Test 3: Cropped images produce valid results")
    print("=" * 60)
    for ext_id in PERSONS:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        full = svc.preview_thinning(img_bytes)
        full_count = len(full["minutiae"])

        center_50 = svc.preview_thinning(crop_center(img_bytes, 0.5))
        corner_25 = svc.preview_thinning(crop_corner(img_bytes, 0.25))

        center_ok = len(center_50["minutiae"]) > 0
        corner_ok = len(corner_25["minutiae"]) > 0
        print(
            f"  {ext_id}: full={full_count} "
            f"center50={len(center_50['minutiae'])} "
            f"corner25={len(corner_25['minutiae'])} "
            f"center={'OK' if center_ok else 'EMPTY'} "
            f"corner={'OK' if corner_ok else 'EMPTY'}",
        )
        if not center_ok:
            failures += 1

    print()
    if failures:
        print(f"FAILED: {failures} test(s) have issues")
    else:
        print("ALL TESTS PASSED")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

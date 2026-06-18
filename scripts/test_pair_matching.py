"""Integration test for Plan 24-02: pair-based matching.

Tests:
  1. Enroll 5 SOCOFing persons via pair pipeline.
  2. Self-match: each person should find themselves in top-1 with score >= 0.5.
  3. Crop match: cropped images should match their full enrollment.

Usage (from apps/backend):
    uv run python ../../scripts/test_pair_matching.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "backend"))

import cv2
import numpy as np

from src.db.qdrant_mcc_repository import QdrantMccRepository
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


def crop_center(img_bytes: bytes, fraction: float) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    y_off = (h - ch) // 2
    x_off = (w - cw) // 2
    cropped = img[y_off:y_off + ch, x_off:x_off + cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


def crop_corner(img_bytes: bytes, fraction: float) -> bytes:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    ch, cw = int(h * fraction), int(w * fraction)
    cropped = img[:ch, :cw]
    _, buf = cv2.imencode(".png", cropped)
    return buf.tobytes()


def main() -> int:
    svc = MccMatchingService()
    repo = QdrantMccRepository.from_host()

    # Ensure pair collection exists, delete old pairs
    repo.ensure_pair_collection()
    for pid in PERSONS:
        repo.delete_pairs_by_person(pid)

    # Generate unique IDs for enrollment
    import uuid
    enrollments: list[tuple[str, str]] = []  # (external_id, capture_id)

    print("=" * 60)
    print("Enrolling 5 SOCOFing persons via pair pipeline")
    print("=" * 60)
    for ext_id in PERSONS:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        cap_id = str(uuid.uuid4())
        fp_id = str(uuid.uuid4())
        num_min, num_pairs = svc.enroll_pairs(
            capture_id=cap_id,
            fingerprint_id=fp_id,
            person_id=ext_id,
            image_bytes=img_bytes,
        )
        enrollments.append((ext_id, cap_id))
        print(f"  {ext_id}: {num_min} minutiae, {num_pairs} pairs")

    print()
    print("=" * 60)
    print("Test 1: Self-match (same image)")
    print("=" * 60)
    self_match_ok = 0
    for ext_id, cap_id in enrollments:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        candidates = svc.search_by_pairs(img_bytes, top_k=5).get("candidates", [])
        if candidates and candidates[0]["person_id"] == ext_id:
            print(
                f"  {ext_id} -> {candidates[0]['person_id']} "
                f"(score={candidates[0]['score']:.3f}, "
                f"peak={candidates[0]['peak_votes']})  OK",
            )
            self_match_ok += 1
        else:
            top = candidates[0]["person_id"] if candidates else "NONE"
            print(f"  {ext_id} -> {top}  FAIL")
            if candidates:
                for c in candidates[:3]:
                    print(f"         {c['person_id']}: score={c['score']:.3f} peak={c['peak_votes']}")

    print()
    print(f"  Self-match: {self_match_ok}/{len(enrollments)}")

    print()
    print("=" * 60)
    print("Test 2: 50% center crop -> full enrollment")
    print("=" * 60)
    crop_50_ok = 0
    for ext_id, cap_id in enrollments:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        crop_bytes = crop_center(img_bytes, 0.5)
        candidates = svc.search_by_pairs(crop_bytes, top_k=5).get("candidates", [])
        if candidates and candidates[0]["person_id"] == ext_id:
            print(
                f"  crop50({ext_id}) -> {candidates[0]['person_id']} "
                f"(score={candidates[0]['score']:.3f}, "
                f"peak={candidates[0]['peak_votes']})  OK",
            )
            crop_50_ok += 1
        else:
            top = candidates[0]["person_id"] if candidates else "NONE"
            print(f"  crop50({ext_id}) -> {top}  FAIL")

    print(f"  50% crop match: {crop_50_ok}/{len(enrollments)}")

    print()
    print("=" * 60)
    print("Test 3: 25% corner crop -> full enrollment")
    print("=" * 60)
    crop_25_ok = 0
    for ext_id, cap_id in enrollments:
        img_bytes = find_socofing_file(ext_id).read_bytes()
        crop_bytes = crop_corner(img_bytes, 0.25)
        candidates = svc.search_by_pairs(crop_bytes, top_k=5).get("candidates", [])
        if candidates and candidates[0]["person_id"] == ext_id:
            print(
                f"  crop25({ext_id}) -> {candidates[0]['person_id']} "
                f"(score={candidates[0]['score']:.3f}, "
                f"peak={candidates[0]['peak_votes']})  OK",
            )
            crop_25_ok += 1
        else:
            top = candidates[0]["person_id"] if candidates else "NONE"
            print(f"  crop25({ext_id}) -> {top}  FAIL")

    print(f"  25% crop match: {crop_25_ok}/{len(enrollments)}")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Self-match:    {self_match_ok}/{len(enrollments)}")
    print(f"  50% crop:      {crop_50_ok}/{len(enrollments)}")
    print(f"  25% crop:      {crop_25_ok}/{len(enrollments)}")

    if self_match_ok == len(enrollments) and crop_50_ok > 0:
        print("\nOVERALL: PASS")
        return 0
    else:
        print("\nOVERALL: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())

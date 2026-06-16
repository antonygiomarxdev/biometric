# Phase 12: Forensic Standards (Orientation Field & MCC)

## Context
Phase 11 established a polyglot architecture (Qdrant coarse + NebulaGraph fine matching) to handle elastic skin deformation. However, Phase 11's fine matcher (using RANSAC on spatial coordinates) proved biologically flawed: it lacks orientation data ($\theta$) and relies on rigid affine transformations, failing under realistic non-linear viscoelastic skin stretching. To reach true forensic standards (ANSI/NIST), we must transition the fine matcher to Minutia Cylinder-Code (MCC).

## Objectives
1. Compute and persist Ridge Orientation ($\theta$) for every minutia.
2. Implement Minutia Cylinder-Code (MCC) local descriptors.
3. Replace the rigid RANSAC Fine Matcher with MCC Local Similarity + Global Relaxation.
4. Prove forensic viability through robust E2E testing against non-linear deformation.

---

## Tasks

### 12-01: Ridge Orientation Field Extraction
* **Description:** Update the extraction pipeline to calculate the angle of the ridge flow at each node.
* **Steps:**
  1. Add `angle: float` field (in radians) to `RidgeNode` dataclass.
  2. Implement a `compute_orientation(image, x, y)` utility using Sobel Structure Tensors.
  3. Update `RidgeGraphExtractor` to populate the `angle` for each node during graph building.
  4. Update `NebulaRepository` schema (`ensure_space`) and insertion logic to store `angle`.
* **Validation:** Unit tests verifying angle correctness on synthetic horizontal, vertical, and diagonal ridges.

### 12-02: Minutia Cylinder-Code (MCC) Generation
* **Description:** Implement the foundational MCC descriptor generation.
* **Steps:**
  1. Create `mcc_descriptor.py` in `processing/`.
  2. Implement cylinder creation: for a given center minutia, find neighbors within radius $R$ (e.g., 50px).
  3. Map neighbors into relative spatial bins $(x, y)$ and directional bins $(\Delta\theta)$, normalized by the center minutia's angle.
  4. Output a serialized bit-vector or float-vector `MccCylinder` per node.
* **Validation:** Tests proving rotation and translation invariance (a rotated fingerprint must yield identical cylinders).

### 12-03: Local Matching & Global Relaxation
* **Description:** Upgrade the `IFineMatcher` implementation in `NebulaRepository` to use MCC.
* **Steps:**
  1. Implement Local Similarity Score: compute similarities (e.g., cosine or bitwise Hamming) between all pairs of cylinders (latent vs. candidate).
  2. Implement Global Relaxation: an iterative voting process where strongly matched cylinder pairs boost the scores of their mutually consistent neighbors.
  3. Replace the `_compute_structural_score` (RANSAC) with the new `_compute_mcc_score`.
* **Validation:** Unit tests proving that partial overlaps score highly while entirely different topologies score near zero.

### 12-04: Forensic E2E Validation
* **Description:** Prove the system withstands real-world "day-to-day" forensics.
* **Steps:**
  1. Write `test_mcc_forensic_e2e.py`.
  2. **Test A (Viscoelasticity):** Apply a Thin-Plate Spline (TPS) non-linear deformation to a real SOCOFing print (simulating a "squished" finger). Assert match passes.
  3. **Test B (Ghost Minutiae):** Inject 30% random noise nodes with arbitrary angles. Assert match passes (MCC naturally ignores unstructured noise).
  4. **Test C (Strict Rejection):** Assert false positive gap is $> 0.5$.
* **Validation:** The full test suite passes with 0 Pyright errors and E2E proofs intact.

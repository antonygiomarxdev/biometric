---
phase: 02-ia-vision-computacional
plan: 04
subsystem: ai-processing
tags: [onnx, deep-learning, minutiae, extraction, heatmap-nms]
requires:
  - phase: 02-02
    provides: AiConfig, ModelManager, ONNX Runtime session lifecycle
provides:
  - ExtractionProcessor (pre/post processing for DL feature extraction)
  - AiFeatureExtractor (IFeatureExtractor implementation using ONNX inference)
affects: [02-05, 02-06]

tech-stack:
  added: [scipy.ndimage (maximum_filter, label)]
  patterns:
    - Heatmap-based minutiae decoding with non-maximum suppression
    - Output-space to canvas-space coordinate scaling for mismatched resolutions
    - Injection of ExtractionProcessor and ModelManager into IFeatureExtractor

key-files:
  created:
    - apps/backend/src/ai/extraction.py
    - apps/backend/tests/test_ai_extractor.py
  modified:
    - apps/backend/src/processing/extractor.py

key-decisions:
  - "Output-space coordinate scaling: model output may be at different resolution than input canvas; scaling factors computed from output spatial dimensions"
  - "Subtract-only padding offset (no additional scale): coordinate adjustment uses canvas offset minus padding, without additional scaling for the content region, since content placement within the canvas is identity"
  - "Simple offset subtraction for coordinate remap (fixing plan's combined scale+offset approach which produced incorrect results)"

patterns-established:
  - "Processor classes (SegmentationProcessor, EnhancementProcessor, ExtractionProcessor) follow a consistent preprocess → model inference → postprocess contract"
  - "Test MockModelManager pattern for all AI inference components"
  - "Empty-list-on-failure contract for IFeatureExtractor implementations"

requirements-completed: [AI-EXT]

duration: 8 min
completed: 2026-06-13
---

# Phase 02 Plan 04: DL Minutiae Extraction Summary

**ExtractionProcessor (pre/post heatmap processing) and AiFeatureExtractor implementing IFeatureExtractor via ONNX Runtime inference pipeline**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-13T20:48:00Z
- **Completed:** 2026-06-13T20:56:12Z
- **Tasks:** 2 (both TDD)
- **Files modified:** 3

## Accomplishments

- **ExtractionProcessor** — preprocess normalizes uint8 images to float32 tensors (1, 1, 512, 512), handles BGR/RGB/grayscale inputs, center-pads to canvas; postprocess decodes multi-channel heatmaps using NMS (3×3 neighbourhood) with connected-component labelling and weighted centre-of-mass for sub-pixel accuracy, maps coordinates back from model output space through canvas space to original image space.
- **AiFeatureExtractor** — Implements `IFeatureExtractor` with `ModelManager` injection for ONNX inference. Converts BGR → grayscale, guards blank/null images, catches model exceptions, returns `list[MinutiaCandidate]` with `AlgorithmOrigin.DEEP_LEARNING`.
- **Existing extractors untouched** — `SkeletonMinutiaeExtractor` and `GradientRidgeExtractor` remain fully functional as fallbacks.
- **20 new tests** covering preprocess shapes/normalization, postprocess decoding/empty/out-of-bounds/coordinate-adjustment/single-channel/confidence-threshold, and full AiFeatureExtractor pipeline with mock ModelManager.

## Task Commits

1. **Task 1: ExtractionProcessor (TDD)** — `12d753a` (test + feat)
2. **Task 2: AiFeatureExtractor (TDD)** — `eed70fe` (feat)

## Files Created/Modified

- `apps/backend/src/ai/extraction.py` — ExtractionProcessor with preprocess/postprocess/_decode_heatmap (272 lines)
- `apps/backend/tests/test_ai_extractor.py` — 20 tests for both processor and feature extractor
- `apps/backend/src/processing/extractor.py` — Added AiFeatureExtractor class (96 lines), imports from `src.ai.extraction` and `src.ai.model_manager`, updated module docstring

## Decisions Made

- **Output-space coordinate scaling**: When the model output is at a different spatial resolution than the 512×512 input canvas, the postprocess computes `scale_x = 512 / out_w` and `scale_y = 512 / out_h` to map output coordinates to canvas space before subtracting the padding offset. This differs from the plan's original `min(w_orig, size) / size` approach, which conflated scaling and offsetting incorrectly.
- **Simple offset subtraction**: Used `x_adj = x_canvas - x_offset` (no additional content-region scaling) since the image content is placed identity-mapped within the canvas.
- **Followed existing patterns**: ExtractionProcessor mirrors SegmentationProcessor/EnhancementProcessor interface, MockModelManager matches existing test fixture pattern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected coordinate remapping from model output space to original image space**
- **Found during:** Task 1 (postprocess coordinate tests)
- **Issue:** The plan's postprocess code used `x_adj = int(m.x / scale_x - x_offset)` with `scale_x = min(w_orig, size) / size`, which produced incorrect results when the output resolution differed from the input canvas resolution. Coordinates could end up outside the original image or at wrong positions.
- **Fix:** Added output-space to canvas-space scaling (`scale_x = size / out_w`, `scale_y = size / out_h`) before the padding offset subtraction. This correctly handles mismatched model output resolutions (e.g., 64×64 output mapping to 512×512 canvas).
- **Files modified:** `apps/backend/src/ai/extraction.py` (postprocess method)
- **Verification:** `test_postprocess_dual_channel`, `test_postprocess_coordinate_adjustment`, `test_postprocess_dual_channel_diff_output_resolution` all pass
- **Committed in:** `12d753a` (Task 1)

**2. [Rule 1 - Bug] Adjusted test peak positions for valid content region**
- **Found during:** Task 1 (test_postprocess_dual_channel failure)
- **Issue:** Test placed peaks at arbitrary output-space coordinates (e.g., y=20, x=10 on a 64×64 output), but after scaling to 512×512 canvas and subtracting offset (224), these mapped outside the 64×64 original image bounds.
- **Fix:** Moved peak positions to the valid content region in output space (e.g., y=32, x=30 for a 64×64 original on 512 canvas → maps to x=16, y=32 in original). Also added a dedicated test for different output resolutions.
- **Files modified:** `apps/backend/tests/test_ai_extractor.py` (peak positions, new test)
- **Verification:** All dual-channel and coordinate tests pass
- **Committed in:** `12d753a` (Task 1)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes were necessary for correct coordinate mapping between model output space, canvas space, and original image space. Without these fixes, extracted minutiae would appear at wrong image positions or be silently discarded.

## Issues Encountered

None — both TDD cycles completed without unexpected issues.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- DL extraction pipeline ready for integration into `FingerprintService`
- SkeletonMinutiaeExtractor remains as fallback — no breaking changes
- Ready for Plan 05 (editor integration / human correction UI)

---

*Phase: 02-ia-vision-computacional*
*Completed: 2026-06-13*

## Self-Check: PASSED

- [x] `apps/backend/src/ai/extraction.py` exists (272 lines)
- [x] `apps/backend/tests/test_ai_extractor.py` exists (371 lines)
- [x] `apps/backend/src/processing/extractor.py` modified (AiFeatureExtractor added)
- [x] Commit `12d753a` exists (test + feat: ExtractionProcessor)
- [x] Commit `eed70fe` exists (feat: AiFeatureExtractor)
- [x] All 25 tests pass (20 new + 5 existing backward compatibility)

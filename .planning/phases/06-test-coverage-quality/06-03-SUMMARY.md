---
phase: 06-test-coverage-quality
plan: 03
subsystem: testing
tags: pytest, coverage, mocks, onnx, llm, pytorch
requires:
  - phase: 05-01
    provides: AI module implementations (llm.py, assistant.py, report_generator.py, model_manager.py, segmentation.py, enhancement.py, extraction.py) and processing pipeline modules (enhancer.py, extractor.py, normalization.py, vectorizer.py)
provides:
  - Comprehensive unit tests for all AI/GenAI modules with >90% combined coverage
  - Mock infrastructure preventing real ONNX model loading, GPU detection, and LLM API calls
  - Test coverage for ModelManager lifecycle, AI pipeline components, and processing extractors
affects: []
tech-stack:
  added: [pytest-cov, importlib.reload patching technique]
  patterns:
    - "Session-scoped autouse fixtures for global mocks (GPU, ONNX, LLM)"
    - "importlib.reload for getting real implementations past class-level mocks"
    - "Synthetic numpy fixtures for deterministic image processing tests"
key-files:
  created:
    - "apps/backend/tests/ai/test_model_manager.py"
    - "apps/backend/tests/ai/test_ai_components.py"
    - "apps/backend/tests/processing/test_enhancers.py"
    - "apps/backend/tests/processing/test_extractor.py"
    - "apps/backend/tests/processing/test_cpu_enhancer.py"
    - "apps/backend/tests/processing/test_normalization.py"
    - "apps/backend/tests/processing/test_vectorizer.py"
  modified:
    - "apps/backend/tests/ai/test_llm.py"
key-decisions:
  - "Used importlib.reload to bypass session-scoped class-level mocks for testing real ModelManager.load_model"
  - "Created a session-scoped _unpatch_create_enhancer fixture to restore the real factory function for factory dispatch tests"
  - "Used synthetic gradient images and skeleton patterns for deterministic extraction tests"
requirements-completed: ["COV-02"]
duration: 42min
completed: 2026-06-13
---

# Phase 06 Plan 03: AI Module & Processing Coverage Summary

**91% combined pytest coverage for src.ai and src.processing with all ONNX/LLM/GPU dependencies mocked**

## Performance

- **Duration:** 42 min
- **Started:** 2026-06-13 (session)
- **Completed:** 2026-06-13
- **Tasks:** 3
- **Files modified:** 9 (2 updated, 7 created)

## Accomplishments

- GenAI modules (llm.py, assistant.py, report_generator.py) achieve 95-100% coverage with mocked LlamaIndex/OpenAI
- ModelManager lifecycle tested at 100% coverage including real load_model caching via importlib.reload
- AI pipeline components (segmentation, enhancement, extraction) tested at 92-100% with synthetic numpy arrays
- Processing enhancer factory tested for all 4 kinds (cpu, segmentation, enhancement, full_ai) including error paths
- Processing extractors (Skeleton, GradientRidge, AiFeatureExtractor) tested at 84% with synthetic skeletons and ridge images
- CpuEnhancer pipeline tested at 95% (normalization, orientation, frequency, Gabor filtering)
- Minutia normalizer (standard + rotation-invariant) and vectorizer tested at 99-100%
- All 121 tests execute in ~31s with zero real model loading or LLM API calls

## Task Commits

Each task was committed atomically:

1. **Task 1: GenAI Modules Coverage** - `7c7c4fa` (test: add GenAI module tests for AiConfig, ModelManager, and provider coverage)
2. **Task 2: AI Vision & Processing Modules Coverage** - `76ba2cd` (test: add AI vision and processing module tests for >90% coverage)
3. **Task 3: Verify AI/Processing Coverage Goal** - `c68ae1b` (test: add remaining coverage tests for cpu, normalization, vectorizer + improve extractor)

**Plan metadata:** (pending)

## Files Created/Modified

### New Test Files (7)
- `tests/ai/test_model_manager.py` - 182 lines: ModelManager init, session lifecycle, typed inference, real load_model caching via importlib.reload
- `tests/ai/test_ai_components.py` - 374 lines: SegmentationProcessor, EnhancementProcessor, ExtractionProcessor pre/post processing with synthetic arrays
- `tests/processing/test_enhancers.py` - 248 lines: SegmentationEnhancer, EnhancementEnhancer, create_enhancer factory, _ChainedAiEnhancer
- `tests/processing/test_extractor.py` - 395 lines: SkeletonMinutiaeExtractor (CN, angle, filter), GradientRidgeExtractor, AiFeatureExtractor
- `tests/processing/test_cpu_enhancer.py` - 96 lines: CpuEnhancer enhance pipeline, normalize, ridge orient/freq, Gabor filter
- `tests/processing/test_normalization.py` - 95 lines: MinutiaNormalizer (consensus, centering, sort), RotationInvariantNormalizer (PCA)
- `tests/processing/test_vectorizer.py` - 101 lines: MinutiaeVectorizer to/from vector, padding

### Modified Test Files (1)
- `tests/ai/test_llm.py` - Added AiConfig defaults, constructor overrides, _resolve_provider, missing provider error path

## Module Coverage Summary

| Module | Coverage | Key Lines Tested |
|--------|----------|-----------------|
| `src/ai/assistant.py` | 100% | ask_assistant, get_assistant_query_engine |
| `src/ai/config.py` | 100% | _resolve_provider (CUDA/CPU), AiConfig defaults |
| `src/ai/enhancement.py` | 100% | pre/post/process, enhance pipeline |
| `src/ai/extraction.py` | 92% | preprocess, postprocess, _decode_heatmap, NMS |
| `src/ai/llm.py` | 95% | OpenAICompatibleProvider, LLMFactory |
| `src/ai/model_manager.py` | 100% | load/unload/loaded_models, _run_single, caching |
| `src/ai/report_generator.py` | 100% | generate_dictamen with retry logic |
| `src/ai/segmentation.py` | 100% | pre/post/process, segment pipeline |
| `src/processing/enhancer.py` | 100% | create_enhancer factory (all 4 kinds) |
| `src/processing/enhancers/ai.py` | 100% | SegmentationEnhancer, EnhancementEnhancer |
| `src/processing/enhancers/base.py` | 96% | BaseEnhancer ABC |
| `src/processing/enhancers/cpu.py` | 95% | Enhance pipeline, ridge orient/freq |
| `src/processing/extractor.py` | 84% | Skeleton/GradientRidge/AiFeatureExtractors |
| `src/processing/normalization.py` | 99% | MinutiaNormalizer, RotationInvariantNormalizer |
| `src/processing/vectorizer.py` | 100% | to_vector, from_vector, pad_vector |
| **Combined** | **91%** | **All AI + Processing modules** |

## Decisions Made

- Used `importlib.reload` to bypass session-scoped class-level mocks when testing the real `ModelManager.load_model` caching and FileNotFoundError behavior. This preserves the global mocks (GPU, ONNX Runtime) while allowing the real Python logic to execute.
- Created a session-scoped `_unpatch_create_enhancer` fixture in `test_enhancers.py` that reloads the `enhancer` module to test the real factory dispatch logic, then wraps it with `wraps=` to protect against side effects.
- Synthetic gradient images (and crossed-line skeleton patterns) provide deterministic test data for extraction tests, avoiding flakiness from random noise patterns.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Session-scoped conftest mocks complicated testing of real methods**: The conftest patches `ModelManager.load_model` and `get_session` at the class level, making it impossible to test the real caching and FileNotFoundError logic through normal means. Resolved by using `importlib.reload` within test-specific `patch` contexts to create fresh, unpatched class definitions while keeping ONNX Runtime and GPU mocks active.
- **`importlib.reload` resets module state**: When reloading modules, any previous imports referencing the old module are stale. Mitigated by capturing the fresh class from the reloaded module and using it only within the test context.

## Threat Surface Scan

No new security-relevant surface introduced - all tests are self-contained unit tests with no network access, file system writes, or external service calls.

## Next Phase Readiness

- All AI and processing modules achieve >90% coverage (combined 91%)
- Coverage for GenAI layer (llm.py, assistant.py, report_generator.py) is 95-100%
- Coverage for vision layer (segmentation.py, enhancement.py, extraction.py) is 92-100%
- Coverage for enhancer factory and AI enhancers is 100%
- Coverage for processor (cpu.py, normalization.py, vectorizer.py) is 95-100%
- Coverage for extractors (Skeleton, GradientRidge, AI) is 84% (combined still >90%)
- Ready for Phase 06 Plan 04

## Self-Check: PASSED

- [x] All task files exist (test_llm.py, test_model_manager.py, test_ai_components.py, test_enhancers.py, test_extractor.py, test_cpu_enhancer.py, test_normalization.py, test_vectorizer.py)
- [x] All 3 commits exist with proper test(06-03) scope
- [x] Coverage at 90.87% exceeds 90% threshold (90% target)
- [x] All 121 tests pass with no real model loading or LLM API calls
- [x] SUMMARY.md created at 06-03-SUMMARY.md

---
*Phase: 06-test-coverage-quality*
*Completed: 2026-06-13*

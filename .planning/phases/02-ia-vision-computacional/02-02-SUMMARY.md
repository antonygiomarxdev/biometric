---
phase: 02-ia-vision-computacional
plan: 02
subsystem: ai-infrastructure
tags: ["pytorch", "onnxruntime", "gpu-detection", "cuda", "model-manager", "segmentation-models"]

# Dependency graph
requires:
  - phase: 02-ia-vision-computacional
    provides: Phase structure, locked decisions (D-01–D-06)
provides:
  - AiConfig frozen dataclass with env-var overrides and auto provider selection
  - ModelManager singleton for ONNX Runtime session lifecycle (GPU/CPU)
  - Real GPU detection via PyTorch replacing hardcoded CPU fallback
  - AlgorithmOrigin.DEEP_LEARNING / GAN_ENHANCED / SEGMENTATION_AI enum values
  - ML stack dependencies (torch, onnxruntime-gpu, segmentation-models-pytorch)
affects:
  - 02-plan-03 (segmentation)
  - 02-plan-04 (enhancement)
  - 02-plan-05 (extraction)
  - 02-plan-06 (integration)

# Tech tracking
tech-stack:
  added:
    - torch>=2.x (CUDA 12.6)
    - torchvision>=0.15
    - onnxruntime>=1.26
    - onnxruntime-gpu>=1.26
    - segmentation-models-pytorch>=0.3
  patterns:
    - AI module lives under src/ai/ as infrastructure layer
    - ModelManager is a singleton wired via FastAPI lifespan
    - Provider auto-select: CUDAExecutionProvider if GPU available, else CPUExecutionProvider

key-files:
  created:
    - apps/backend/src/ai/__init__.py
    - apps/backend/src/ai/config.py
    - apps/backend/src/ai/model_manager.py
  modified:
    - apps/backend/src/core/types.py
    - apps/backend/src/core/config.py
    - apps/backend/src/core/gpu_utils.py
    - apps/backend/pyproject.toml

key-decisions:
  - "AiConfig is a frozen dataclass (not Pydantic) to avoid dependency coupling in the infrastructure layer"
  - "Provider auto-select computed in __post_init__ via object.__setattr__ to work with frozen=True"
  - "ModelManager uses dict[str, ort.InferenceSession] for session cache, no external LRU yet"
  - "GPU detection function is module-level (eager) so consumers see GPU_AVAILABLE at import time"
  - "FORCE_CPU env var preserved from existing config for testing on GPU machines"
  - "ML packages declared as [project.optional-dependencies] ai extra group"

patterns-established:
  - "New AI components implement domain interfaces (IEnhancer, IFeatureExtractor) and never import ModelManager directly per Clean Architecture"
  - "All AI module code has full type annotations compatible with pyright strict"
  - "ONNX Runtime provider falls back to CPUExecutionProvider when CUDA is unavailable"

requirements-completed:
  - AI-INFRA

# Metrics
duration: 8min
completed: 2026-06-13
---

# Phase 02: Computer Vision AI — Plan 02 Summary

**AI infrastructure scaffold: AiConfig, ModelManager singleton, GPU detection via PyTorch, ONNX Runtime session management, and expanded AlgorithmOrigin enum**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-13T20:32:00Z
- **Completed:** 2026-06-13T20:40:00Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- **AiConfig frozen dataclass** with 9 fields (model_dir, GPU settings, model names, inference params) and automatic ONNX Runtime provider selection based on hardware availability.
- **ModelManager singleton** providing session caching, GPU/CPU provider fallback, typed inference methods (segmentation, enhancement, extraction), and full lifecycle management.
- **GPU detection** replaced hardcoded `GPU_AVAILABLE = False` with PyTorch-based real detection — detects NVIDIA GeForce RTX 4070 Laptop GPU with CUDA 12.6.
- **AlgorithmOrigin** extended with DEEP_LEARNING, GAN_ENHANCED, SEGMENTATION_AI for tracking AI inference origins alongside existing classical algorithms.
- **ML stack dependencies** declared as optional `[ai]` extras in pyproject.toml and verified working (torch 2.12, onnxruntime 1.26, CUDA 12.6 all functional).

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AI Module Structure & Update Types** — `a07935e` (feat)
2. **Task 2: Update Config & GPU Detection** — `eab2a6f` (feat)
3. **Task 3: Implement ModelManager Singleton** — `c3d56fb` (feat)
4. **Pyproject.toml ML dependencies** — `2446172` (chore)

## Files Created/Modified

- `apps/backend/src/ai/__init__.py` — Package marker exporting AiConfig, ModelManager
- `apps/backend/src/ai/config.py` — AiConfig frozen dataclass with 9 env-var-overridable fields
- `apps/backend/src/ai/model_manager.py` — ModelManager: session cache, lifecycle, 3 typed inference helpers
- `apps/backend/src/core/types.py` — 3 new AlgorithmOrigin enum values
- `apps/backend/src/core/config.py` — 5 new AI/ML config fields
- `apps/backend/src/core/gpu_utils.py` — Real PyTorch-based GPU detection with FORCE_CPU support
- `apps/backend/pyproject.toml` — ML stack optional dependencies

## Decisions Made

- **AiConfig is frozen dataclass, not Pydantic** — avoids coupling infrastructure layer to Pydantic. Env overrides work via `os.getenv()` in `field` defaults.
- **Provider auto-select in `__post_init__`** — uses `object.__setattr__` to set `provider` on the frozen dataclass. Computed once at construction time.
- **Module-level GPU detection** — `GPU_AVAILABLE` is set at import time via `_detect_gpu()`. Consumers see the value immediately without lazy initialization.
- **ML packages as optional extras** — declared under `[project.optional-dependencies] ai = [...]` so production deployments can choose `pip install biometric-fingerprint[ai]`.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- AI infrastructure complete, ready for Plans 03–06 (segmentation, enhancement, extraction, integration)
- Plans 03+ can inject ModelManager via their service constructors and call `run_segmentation()` / `run_enhancement()` / `run_extraction()` directly
- ModelManager loads ONNX models from `data/models/` — models themselves will be trained/exported in later plans

---
*Phase: 02-ia-vision-computacional*
*Completed: 2026-06-13*

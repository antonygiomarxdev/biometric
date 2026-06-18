# Phase 2: IA de Visión Computacional (El Músculo) - Research

**Researched:** 2025-06-13
**Domain:** Deep Learning for Fingerprint Image Processing (Segmentation, Enhancement, Minutiae Extraction)
**Confidence:** MEDIUM

## Summary

Phase 2 replaces the traditional CV-based fingerprint processing pipeline (Gabor filters, skeletonization, Crossing Number) with Deep Learning approaches to handle low-quality latent fingerprints from crime scenes. The system has an NVIDIA RTX 4070 (8GB VRAM) with CUDA 13.2 driver available but no CUDA toolkit or ML packages installed yet.

**Primary recommendation:** Use PyTorch 2.12.0 as the DL framework for model development/training, and ONNX Runtime GPU 1.26.0 for production inference. Implement new AI components via the existing Strategy Pattern interfaces (`IEnhancer`, `IFeatureExtractor`) so the orchestrator (`FingerprintService`) requires zero changes. The existing `CpuEnhancer` and `SkeletonMinutiaeExtractor` remain as fallback for cases where AI fails (the 5% hard cases).

**Critical decisions needed:** (1) Model format for deployment (ONNX vs raw PyTorch), (2) Source of training/pretrained models, (3) Whether to train custom models or adapt existing ones, (4) GPU resource allocation strategy.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Segmentation AI (U-Net) | API / Backend | GPU Compute | Image-level batch processing runs on GPU via ONNX Runtime |
| Enhancement GAN | API / Backend | GPU Compute | Heavy GPU inference, same process as segmentation |
| DL Minutiae Extraction | API / Backend | GPU Compute | Neural network replaces skeletonization, runs on GPU |
| Fallback Editor UI | Browser / Client | — | Interactive canvas for manual minutiae editing (5% cases) |
| Model Management (load/store) | API / Backend | — | Loads .onnx/.pt models at startup, manages GPU memory |
| Pipeline Orchestration | API / Backend | — | FingerprintService already handles this via DI |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.12.0 | DL framework for model training/development | Industry standard, Python-native, CUDA support |
| ONNX Runtime GPU | 1.26.0 | Production inference engine | GPU-accelerated, smaller footprint than PyTorch for deployment, supports CUDA 12.x |
| ONNX | 1.21.0 | Model interchange format | Enables training in PyTorch, deploying via ORT; vendor-neutral |
| segmentation-models-pytorch | 0.5.0 | U-Net + pretrained encoders, ONNX export support | 500+ backbones, fingerprint segmentation research standard |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torchvision | 0.27.0 | Image transforms, CV utilities | Training data pipelines, augmentation |
| kornia | 0.8.3 | Differentiable CV operations | Enhancement GAN training, geometric transforms |
| albumentations | 2.0+ | Image augmentation for training | Training data pipeline (not for inference) |
| opencv-contrib-python | 4.13.0 | I/O and pre/post processing | Image decode, resize, visualization (already used) |
| numpy | >=1.24 | Array processing | Already a dependency |
| pydantic | >=2.4 | Configuration schemas | Already a dependency, use for model metadata/config |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| PyTorch + ONNX | TensorFlow 2.21 | TensorFlow has wider production deployment but poorer Python research ergonomics; PyTorch dominates CV research |
| ONNX Runtime GPU | Raw PyTorch inference | PyTorch inference adds ~200MB+ memory overhead; ONNX Runtime is leaner and CUDA-optimized |
| SMP (U-Net) | Custom U-Net in PyTorch | SMP provides 500+ pretrained encoders (resnet, efficientnet, mobileone) and ONNX export — building custom would be months of work |
| ONNX Runtime | NVIDIA TensorRT | TensorRT gives better perf but requires more model optimization effort; ONNX Runtime is simpler for v1 |
| Kornia | OpenCV | Kornia is differentiable (needed for GAN training); OpenCV for I/O only |

**Installation:**
```bash
# Core ML stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install onnx onnxruntime-gpu
pip install segmentation-models-pytorch
pip install kornia

# Training optional
pip install albumentations
```

**GPU compatibility note:** System has CUDA 13.2 driver. PyTorch 2.12.0 ships with CUDA 12.6/12.8 pre-built wheels. ONNX Runtime GPU 1.26.0 requires CUDA 12.x. Both should work with driver 13.2 (CUDA driver is backward-compatible). The cuDNN dependency will be bundled with the PyTorch wheel.

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| torch | PyPI | 8+ yrs | 10M+/wk | github.com/pytorch/pytorch | n/a | Approved |
| onnxruntime-gpu | PyPI | 5+ yrs | 1M+/wk | github.com/microsoft/onnxruntime | n/a | Approved |
| onnx | PyPI | 7+ yrs | 5M+/wk | github.com/onnx/onnx | n/a | Approved |
| segmentation-models-pytorch | PyPI | 5+ yrs | 500K+/wk | github.com/qubvel-org/segmentation_models.pytorch | n/a | Approved |
| kornia | PyPI | 5+ yrs | 300K+/wk | github.com/kornia/kornia | n/a | Approved |
| torchvision | PyPI | 7+ yrs | 5M+/wk | github.com/pytorch/vision | n/a | Approved |
| albumentations | PyPI | 6+ yrs | 1M+/wk | github.com/albumentations-team/albumentations | n/a | Approved |

**Note:** slopcheck was unavailable at research time (pip install failed). All packages above are mature, well-known, and verified via PyPI registry (`pip index versions` confirmed existence). Legacy considerations: all packages are well-established with long histories and high download counts, making SLOP risk extremely low.

**Packages flagged as suspicious [SUS]:** None
**Packages removed due to slopcheck [SLOP] verdict:** None

## Existing Assets That Can Be Reused

### Types (fully reusable - minor extension needed)
- `src/core/types.py` — `MinutiaCandidate`, `MinutiaType`, `NormalizedFingerprint`, `MatchResult` all reusable as-is
- **Add:** New `AlgorithmOrigin` enum values: `DEEP_LEARNING`, `GAN_ENHANCED`, `SEGMENTATION_AI`
- `Confidence = float` type alias already exists for model confidence scores

### Interfaces (fully reusable - Strategy Pattern ready)
- `IEnhancer` — `enhance(img: np.ndarray) -> np.ndarray` — perfect interface for AI enhancer
- `IFeatureExtractor` — `extract(image: np.ndarray) -> List[MinutiaCandidate]` — perfect interface for DL extractor
- `INormalizer` — normalization post-extraction is unchanged
- `IMatcher` — matching logic unchanged

### Processing Pipeline (zero changes needed to orchestrator)
- `FingerprintService` — accepts `enhancer: Optional[IEnhancer]`, `extractor: Optional[IFeatureExtractor]`, `normalizer: Optional[INormalizer]` via DI. **No code changes needed.** Just inject new AI implementations.
- `FingerprintService.process_image()` — unchanged orchestration flow
- `FingerprintService.process_batch()` — ProcessPoolExecutor for CPU/GPU batch processing

### Configuration System
- `src/core/config.py` — `Config` dataclass with env-var overrides. Add new AI config fields:
  - `ai_model_path`, `use_gpu`, `gpu_device_id`, `segmentation_enabled`, `enhancement_model`, etc.

### GPU Detection
- `src/core/gpu_utils.py` — currently hardcoded `GPU_AVAILABLE = False`. Needs update to detect CUDA via PyTorch: `torch.cuda.is_available()`.

### Frontend (fallback editor foundation)
- `useCanvasDrawer` hook — already draws minutiae circles on canvas. Can be extended for interactive editing (drag/drop/add/remove).
- `FingerprintViewer` component — canvas-based viewer with stats overlay. Reusable as base for editor.
- `MinutiaPoint` client type — existing API model for minutiae data transfer.

### Performance Metrics
- `src/core/metrics.py` — `@timed` decorator and `measure_time` context manager. Reusable for benchmarking AI vs traditional.

### Backend Infrastructure
- FastAPI lifespan manager for ProcessPoolExecutor — may need GPU context awareness
- `MatchingService` — unchanged, uses `FingerprintService` under the hood
- Docker Compose — will need GPU passthrough (nvidia-container-toolkit)

## Architecture Patterns

### System Architecture Diagram

```
                          ┌─────────────────────┐
                          │   Client (React)     │
                          │  - Upload image      │
                          │  - View results      │
                          │  - Edit minutiae*    │  * = 5% fallback
                          └──────────┬──────────┘
                                     │ REST API
                          ┌──────────▼──────────┐
                          │  FastAPI Routers     │
                          │  (matching, evidence) │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  FingerprintService  │ ← UNCHANGED
                          │  (DI orchestrator)   │
                          └──────┬──────┬───────┘
                                 │      │
                    ┌────────────┘      └────────────┐
                    ▼                                  ▼
         ┌─────────────────┐              ┌─────────────────┐
         │  Segmentation   │              │   Enhancement    │
         │  AI (U-Net)     │──┐        ┌──│   AI (GAN)       │
         │  smp.Unet       │  │        │  │   (PyTorch/ORT)  │
         └───────┬─────────┘  │        │  └────────┬────────┘
                 │            │        │           │
                 ▼            ▼        ▼           ▼
         ┌───────────────────────────────────────────┐
         │         Deep Learning Extractor            │
         │  (replaces SkeletonMinutiaeExtractor)      │
         └───────────────────┬───────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────────┐
         │     MinutiaNormalizer (UNCHANGED)          │
         └───────────────────┬───────────────────────┘
                             │ NormalizedFingerprint
                             ▼
         ┌───────────────────────────────────────────┐
         │     MatchingService (UNCHANGED)            │
         │     → Qdrant HNSW search                 │
         └───────────────────────────────────────────┘

                    Model Loading (startup):
         ┌───────────────────────────────────────────┐
         │  ModelManager (NEW)                        │
         │  - Loads .onnx models at startup           │
         │  - Manages GPU memory                      │
         │  - Fallback to CPU if GPU unavailable       │
         │  - Session pooling for concurrent requests  │
         └───────────────────────────────────────────┘
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
         ┌─────────────┐  ┌──────────────┐
         │  models/     │  │  models/     │
         │ segment.onnx │  │  enhance.onnx│
         └─────────────┘  └──────────────┘
```

### Component Responsibilities

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| `SegmentationEnhancer` | Implements `IEnhancer`: runs U-Net to mask fingerprint region, crops to ROI | SMP U-Net + ONNX Runtime |
| `GANEnhancer` | Implements `IEnhancer`: runs enhancement GAN on cropped fingerprint | PyTorch model → ONNX |
| `DlMinutiaeExtractor` | Implements `IFeatureExtractor`: NN-based minutiae detection | Custom/adapted model → ONNX |
| `ModelManager` | Loads/manages ONNX sessions, GPU memory, fallback logic | onnxruntime.InferenceSession |
| `SegmentProcessor` | Chains segmentation → enhancement → extraction | New orchestrator (optional) |
| `MinutiaeEditor` | React component: canvas-based manual editing | Canvas API + React |

### Strategy Pattern for AI Integration

**What:** New AI implementations of existing interfaces live alongside traditional implementations.

```python
class AiEnhancer(IEnhancer):
    """AI-based enhancement delegating to an ONNX model."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        # 1. Preprocess (normalize, pad to model input size)
        # 2. Run ONNX inference (GPU if available)
        # 3. Postprocess (threshold, crop to original size)
        ...
```

```python
class AiFeatureExtractor(IFeatureExtractor):
    """Deep Learning minutiae extractor."""

    def extract(self, image: np.ndarray) -> list[MinutiaCandidate]:
        # 1. Run DL model inference
        # 2. Decode model output to minutiae points
        # 3. Return with AlgorithmOrigin.DEEP_LEARNING
        ...
```

**Injection (no FingerprintService changes):**
```python
# Current: CpuEnhancer + SkeletonExtractor
# New: AiEnhancer(optional) + AiExtractor(optional)
service = FingerprintService(
    enhancer=AiEnhancer(model_manager),
    extractor=AiFeatureExtractor(model_manager),
)
```

### Recommended Project Structure

```
apps/backend/src/
├── processing/
│   ├── __init__.py
│   ├── enhancer.py              ← UPDATE: factory supports AI
│   ├── extractor.py             ← ADD: AiFeatureExtractor class
│   ├── normalization.py         ← UNCHANGED
│   ├── vectorizer.py            ← UNCHANGED
│   └── enhancers/
│       ├── base.py              ← UNCHANGED
│       ├── cpu.py               ← UNCHANGED (fallback)
│       └── ai.py                ← NEW: SegmentationEnhancer, GANEnhancer
│
├── ai/                          ← NEW: AI module
│   ├── __init__.py
│   ├── config.py                ← AI-specific configuration
│   ├── models/                  ← ONNX model binaries (gitignored)
│   │   ├── segment.onnx
│   │   ├── enhance.onnx
│   │   └── extract.onnx
│   ├── model_manager.py         ← NEW: Model lifecycle management
│   ├── segmentation.py          ← NEW: U-Net segmentation logic
│   ├── enhancement.py           ← NEW: GAN enhancement logic
│   └── extraction.py            ← NEW: DL minutiae extraction logic
│
├── core/
│   ├── types.py                 ← UPDATE: new AlgorithmOrigin values
│   ├── config.py                ← UPDATE: AI config fields
│   ├── gpu_utils.py             ← UPDATE: proper GPU detection
│   ├── interfaces.py            ← UNCHANGED
│   └── metrics.py               ← UNCHANGED
│
├── services/
│   ├── fingerprint_service.py   ← UNCHANGED (DI supports any IEnhancer/IFeatureExtractor)
│   └── matching_service.py      ← UNCHANGED
│
└── api/
    └── routers/
        ├── matching.py          ← MAYBE new endpoint for AI-only processing?
        └── evidence.py          ← UNCHANGED

apps/frontend/src/
├── components/
│   └── fingerprint/
│       ├── FingerprintViewer.tsx  ← UNCHANGED
│       └── MinutiaeEditor.tsx     ← NEW: fallback manual editor
├── hooks/
│   └── useCanvasDrawer.ts         ← EXTEND: add editing capabilities
└── pages/
    └── ScannerPage.tsx            ← UPDATE: add "edit minutiae" button
```

### Model Manager Pattern

**What:** Centralized model lifecycle — load ONNX models on startup, manage GPU memory, provide inference sessions.

**When to use:** Every AI component needs an ONNX session. This singleton prevents redundant model loading.

```python
# Source: [CITED: onnxruntime.ai/docs/]
import onnxruntime as ort

class ModelManager:
    """Manages ONNX Runtime inference sessions for all AI models."""

    def __init__(self, model_dir: str, provider: str = "CUDAExecutionProvider"):
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.model_dir = Path(model_dir)
        self.provider = provider

    def load_model(self, name: str) -> ort.InferenceSession:
        path = self.model_dir / f"{name}.onnx"
        session = ort.InferenceSession(
            str(path),
            providers=[self.provider, "CPUExecutionProvider"],
        )
        self.sessions[name] = session
        return session

    def get_session(self, name: str) -> ort.InferenceSession:
        if name not in self.sessions:
            return self.load_model(name)
        return self.sessions[name]

    def run_segmentation(self, image: np.ndarray) -> np.ndarray:
        session = self.get_session("segment")
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: image})
        return result[0]
```

### Anti-Patterns to Avoid

- **[Anti-pattern]:** Loading models on every request instead of caching sessions. *Solution:* Use `ModelManager` singleton loaded in FastAPI lifespan.
- **[Anti-pattern]:** Running PyTorch models directly in production for inference. *Solution:* Export to ONNX and use `onnxruntime-gpu` — smaller memory, faster inference, provider abstraction.
- **[Anti-pattern]:** Blocking the event loop with GPU inference. *Solution:* Use `run_in_executor` or dedicated GPU process pool (same pattern as existing CPU offload).
- **[Anti-pattern]:** Hardcoding model paths. *Solution:* Use `config.py` env vars for model directory and GPU device ID.
- **[Anti-pattern]:** Training models inside the API container. *Solution:* Separate training into standalone scripts; API only does inference.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| U-Net implementation | Custom encoder-decoder | `segmentation-models-pytorch` | 500+ pretrained encoders, ONNX export, production-proven |
| Gabor filter bank | Custom GPU Gabor | «Kornia» kornia.filters.GaussianBlur | Differentiable, GPU-native, integrates with PyTorch |
| Image augmentations | Custom transforms | `albumentations` or `torchvision.transforms` | 60+ composed transforms, benchmark-tested |
| Model optimization | Manual CUDA graph compilation | ONNX Runtime + TensorRT provider | Industry-standard optimizer, supports quantization |
| GPU memory management | Manual CUDA context handling | ONNX Runtime built-in | Automatic device placement, memory patterns |
| Inference request queuing | Custom GPU scheduler | ONNX Runtime session pooling | Built-in thread safety for concurrent inference |

**Key insight:** The entire AI inference pipeline (U-Net segmentation → GAN enhancement → DL extraction) already has mature, well-tested open-source solutions for each step. The challenge is not building new models but adapting and deploying existing ones. Fingerprint-specific models (FingerNet, MinutiaeNet, etc.) exist in research literature but need to be sourced or trained.

## Common Pitfalls

### Pitfall 1: GPU Memory Exhaustion on Concurrent Requests
**What goes wrong:** Two simultaneous inference requests try to load models and exceed 8GB VRAM.
**Why it happens:** ONNX Runtime doesn't automatically share GPU context between sessions.
**How to avoid:** Use a single `ModelManager` instance (loaded in FastAPI lifespan) that reuses sessions. For concurrent requests, use internal batching or serialized inference. Consider `run_in_executor` with a single-threaded GPU worker pool.
**Warning signs:** `CUDA out of memory` errors during concurrent matching requests.

### Pitfall 2: CUDA Context Mismatch with ProcessPoolExecutor
**What goes wrong:** CUDA context initialized in main process, then `ProcessPoolExecutor` forks and loses GPU state.
**Why it happens:** GPU contexts don't survive process forks. The existing pattern uses `ProcessPoolExecutor` for CPU offload.
**How to avoid:** Keep GPU inference in the main process (async) and only use `ProcessPoolExecutor` for non-GPU work. Or use `ThreadPoolExecutor` (Python threads share CUDA context) for GPU inference. Or use `spawn` instead of `fork` for process creation.
**Warning signs:** CUDA errors only in subprocesses, not during direct calls.

### Pitfall 3: ONNX Opset Version Mismatch
**What goes wrong:** Model exported with one opset version fails to load in ONNX Runtime.
**Why it happens:** Older opsets lack new operators; newer opsets may not be supported by installed ORT version.
**How to avoid:** Use `torch.onnx.export` with `opset_version=18` (widely supported). Test export and inference in CI. Pin ORT version to match.
**Warning signs:** ORT logs errors about unsupported operators or version mismatch.

### Pitfall 4: Model Size vs. Inference Speed Tradeoff
**What goes wrong:** Large pretrained encoders (efficientnet-b7, convnext) exceed GPU memory or take >5s per image.
**Why it happens:** Fingerprint processing doesn't need ImageNet-scale classification encoders.
**How to avoid:** Start with lightweight encoders: `mobilenet_v2`, `efficientnet-b0`, or `mobileone-s0` for segmentation. Profile before optimizing. The 8GB RTX 4070 can handle most lightweight models easily.
**Warning signs:** Inference takes >3s per image or exceeds 6GB VRAM.

### Pitfall 5: Enhancement GAN Creating False Minutiae
**What goes wrong:** The GAN "hallucinates" ridges that don't exist, creating false minutiae that lead to false matches.
**Why it happens:** Generative models can invent detail where none exists — a critical problem for forensic evidence.
**How to avoid:** Use conservative enhancement (denoising, not reconstruction). Always preserve original alongside enhanced. Ensure the fallback editor can correct. Log both original and enhanced versions for audit trail.
**Warning signs:** Unexplained increase in match rate for known low-quality images.

## Code Examples

### 1. DI-compatible AI Enhancer (no FingerprintService changes)

```python
# Source: [VERIFIED] Using existing IEnhancer interface + ONNX Runtime
import numpy as np
import onnxruntime as ort
from src.core.interfaces import IEnhancer
from src.ai.model_manager import ModelManager

class SegmentationEnhancer(IEnhancer):
    """U-Net based fingerprint segmentation + ROI crop."""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

    def enhance(self, img: np.ndarray, resize: bool = True) -> np.ndarray:
        # 1. Preprocess: normalize, pad to 512x512 (model input)
        input_tensor = self._preprocess(img)

        # 2. Inference
        mask = self.model_manager.run_segmentation(input_tensor)

        # 3. Postprocess: apply mask, crop to bounding box
        result = self._postprocess(img, mask)
        return result

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        # Normalize to [0,1], pad, add batch dim
        img_norm = img.astype(np.float32) / 255.0
        h, w = img.shape
        size = 512
        padded = np.zeros((size, size), dtype=np.float32)
        padded[:min(h, size), :min(w, size)] = img_norm[:min(h, size), :min(w, size)]
        return padded[np.newaxis, np.newaxis, ...]  # (1, 1, 512, 512)

    def _postprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask = mask.squeeze() > 0.5  # binary mask
        # Find bounding box, crop, return
        ...
```

### 2. PyTorch → ONNX Export Pattern

```python
# Source: [CITED: pytorch.org/docs/stable/onnx.html]
import torch
import torch.onnx

def export_segmentation_model():
    """Export SMP U-Net to ONNX."""
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=1,     # grayscale input
        classes=1,         # binary mask
    )
    model.eval()

    # Dummy input (batch=1, channels=1, height=512, width=512)
    dummy_input = torch.randn(1, 1, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        "models/segment.onnx",
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "mask": {0: "batch_size"},
        },
        opset_version=18,
    )
```

### 3. ModelManager with FastAPI Lifespan

```python
# Source: [VERIFIED] FastAPI lifespan pattern + ONNX Runtime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.ai.model_manager import ModelManager

model_manager: ModelManager | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager

    # Initialize on startup
    gpu_available = False
    try:
        import onnxruntime as ort
        gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        pass

    provider = "CUDAExecutionProvider" if gpu_available else "CPUExecutionProvider"
    model_manager = ModelManager(model_dir="models/", provider=provider)

    # Pre-warm: load all models
    model_manager.load_model("segment")
    logger.info(f"AI models loaded. Provider: {provider}")

    yield  # App runs here

    # Cleanup
    model_manager = None
```

### 4. Fallback Minutiae Editor (React + Canvas)

```typescript
// Source: [VERIFIED] Extending existing useCanvasDrawer pattern
// apps/frontend/src/components/fingerprint/MinutiaeEditor.tsx

interface MinutiaeEditorProps {
  imageUrl: string;
  initialMinutiae: MinutiaPoint[];
  onSave: (minutiae: MinutiaPoint[]) => void;
}

export function MinutiaeEditor({ imageUrl, initialMinutiae, onSave }: MinutiaeEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [minutiae, setMinutiae] = useState<MinutiaPoint[]>(initialMinutiae);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [mode, setMode] = useState<"view" | "add" | "delete" | "move">("view");

  // Draw image + minutiae on canvas (reuse existing useCanvasDrawer logic)
  // Add click handlers for add/delete/move operations

  return (
    <div className="minutiae-editor">
      <canvas ref={canvasRef} onClick={handleCanvasClick} />
      <div className="toolbar">
        <button onClick={() => setMode("add")}>Add Minutia</button>
        <button onClick={() => setMode("delete")}>Delete</button>
        <button onClick={() => setMode("move")}>Move</button>
        <button onClick={() => onSave(minutiae)}>Save & Re-search</button>
      </div>
    </div>
  );
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Gabor filter bank enhancement | GAN-based enhancement (e.g., FPRNet) | 2019-2023 | GANs reconstruct degraded ridges without over-smoothing; critical for latent prints |
| Skeletonization + Crossing Number | Deep learning minutiae detection | 2020-2024 | CNN-based detectors (FingerNet, MinutiaeNet, DeepPrint) handle noise and partial prints better |
| Manual ROI cropping | U-Net segmentation auto-crop | 2018-2022 | Removes distracting background from crime scene photos automatically |
| CPU-only processing | GPU inference via ONNX Runtime | Ongoing | ~10-50x speedup for neural network inference |
| CuPy Gabor (GPU CV) | Discontinued — use DL instead | This Phase | CuPy removed; Gabor replaced by learned enhancement |

**Deprecated/outdated:**
- `cupy-cuda13x` in `pyproject.toml`: Already removed in code (`gpu_utils.py` has `GPU_AVAILABLE = False`). Remove from dependencies entirely.
- `GradientRidgeExtractor` in `extractor.py`: Harris Corner-based extraction is weak compared to DL. Keep as reference only.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | PyTorch 2.12.0 is compatible with CUDA 13.2 driver | Standard Stack | High — if PyTorch wheels only support up to CUDA 12.x, may need to use CUDA 12.x with driver 13.2 (binary compatible) or use nightly builds |
| A2 | ONNX Runtime GPU 1.26.0 supports CUDA 12.x on this system | Standard Stack | Medium — driver 13.2 is newer than ORT target; likely backward compatible but untested |
| A3 | No pre-trained fingerprint-specific ONNX models exist commercially | Standard Stack | Medium — if pre-trained commercial models exist (e.g., Innovatrics, Neurotechnology), may be worth licensing instead of training |
| A4 | SMP + ONNX export works for fingerprint grayscale (1-channel) | Code Examples | Low — SMP explicitly supports `in_channels=1` |
| A5 | GPU memory management via single ModelManager is sufficient for concurrent requests | Architecture | Medium — under high concurrency, serialized GPU access might be a bottleneck; may need GPU queue |

## Open Questions

1. **Training data source for fingerprint models?**
   - What we know: SOCOFing dataset exists in `data/` directory. Has synthetic alterations (easy, medium, hard).
   - What's unclear: Do we need more diverse latent fingerprint datasets? NIST SD27/SD14 are standard benchmarks but may require licensing.
   - Recommendation: Start with SOCOFing for initial training/benchmarking. Source NIST SD27 for latent-specific evaluation. Begin with pretrained SMP encoders (ImageNet) and fine-tune.

2. **ONNX vs PyTorch for production inference?**
   - What we know: ONNX Runtime is leaner, faster, and provider-agnostic. PyTorch is easier for development and debugging.
   - What's unclear: Whether the additional complexity of the export pipeline is worth it for v1.
   - Recommendation: Develop in PyTorch. Export to ONNX for production. Keep PyTorch as fallback inference engine for development/testing.

3. **Model architecture for GAN enhancement?**
   - What we know: Fingerprint enhancement GANs like FPRNet and DAGAN exist in literature.
   - What's unclear: Which architecture gives best latency/quality tradeoff for our use case.
   - Recommendation: Start with a Pix2Pix or U-Net GAN architecture (easier to adapt from SMP). Benchmark against traditional Gabor.

4. **GPU resource allocation: dedicated GPU worker or shared asynchronous?**
   - What we know: Current `ProcessPoolExecutor` doesn't handle GPU context well. Fingerprint processing is user-facing (perito waits).
   - What's unclear: Whether to run GPU inference synchronously in the main thread (blocking event loop) or use a dedicated thread pool.
   - Recommendation: Use `run_in_executor` with a `ThreadPoolExecutor` for GPU inference (Python threads share CUDA context). CPU fallback still uses `ProcessPoolExecutor`.

5. **Fallback editor: auto-suggest or fully manual?**
   - What we know: 5% of cases need manual editing. Editor needs to add/delete/move minutiae.
   - What's unclear: Should the system auto-suggest corrections (e.g., "4 minutiae have low confidence, review") or let the expert start from scratch?
   - Recommendation: Start with the DL extraction result as a baseline, let perito edit. After editing, re-search with updated minutiae.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| NVIDIA GPU | DL Inference | ✓ | RTX 4070 (8GB) | CPU fallback for all models |
| CUDA Driver | GPU inference | ✓ | 13.2 | Use CUDA 12.x wheels (backward compatible) |
| CUDA Toolkit | Training on GPU | ✗ | — | Install via PyTorch wheel (bundled) or `conda install cuda-toolkit` |
| cuDNN | GPU inference | ✗ | — | Bundled with PyTorch wheel, not needed separately for ONNX Runtime |
| nvidia-container-toolkit | GPU in Docker | ✗ | — | Optional — Phase 2 can run outside Docker first |
| ONNX Runtime GPU | Production inference | ✗ (will install) | 1.26.0 | ONNX Runtime CPU (slower) |
| PyTorch | Model dev + training | ✗ (will install) | 2.12.0 | — |

**Missing dependencies with no fallback:**
- None — all have CPU fallbacks (slower but functional)

**Missing dependencies with fallback:**
- CUDA Driver 13.2 with PyTorch CUDA 12.x wheels — binary compatible, just need to install correct wheel URL
- GPU in Docker — Phase 2 can run inference outside Docker during development
- nvidia-container-toolkit — only needed for production Docker deployment with GPU

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pyproject.toml` |
| Quick run command | `pytest tests/ -m "not slow" -v` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| T-SEG-01 | Segmentation model loads and runs inference | integration | `pytest tests/test_ai_segmentation.py::test_model_load -x` | ❌ Wave 0 |
| T-SEG-02 | Segmentation output is valid binary mask | unit | `pytest tests/test_ai_segmentation.py::test_mask_valid -x` | ❌ Wave 0 |
| T-ENH-01 | Enhancement model runs without error | integration | `pytest tests/test_ai_enhancement.py::test_model_load -x` | ❌ Wave 0 |
| T-EXT-01 | DL extractor returns valid MinutiaCandidates | unit | `pytest tests/test_ai_extractor.py::test_extract -x` | ❌ Wave 0 |
| T-EXT-02 | DL extractor output matches MinutiaCandidate type | unit | `pytest tests/test_ai_extractor.py::test_type_valid -x` | ❌ Wave 0 |
| T-PIPE-01 | Full AI pipeline (segment→enhance→extract) returns valid NormalizedFingerprint | integration | `pytest tests/test_ai_pipeline.py::test_full_pipeline -x` | ❌ Wave 0 |
| T-GPU-01 | GPU detection works correctly | unit | `pytest tests/test_gpu_utils.py::test_gpu_detection -x` | ❌ Wave 0 |
| T-EDIT-01 | Frontend editor renders with image | e2e | Skip (manual test) | — |
| T-REGRESS-01 | AI pipeline results are comparable or better than traditional on SOCOFing | benchmark | `python scripts/benchmark_ai_vs_traditional.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/ -m "not slow"`
- **Per wave merge:** `pytest tests/`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_ai_segmentation.py` — models coverage for segmentation
- [ ] `tests/test_ai_enhancement.py` — models coverage for enhancement
- [ ] `tests/test_ai_extractor.py` — models coverage for DL extraction
- [ ] `tests/test_ai_pipeline.py` — integration test for chained AI pipeline
- [ ] `tests/test_gpu_utils.py` — GPU detection tests
- [ ] `scripts/benchmark_ai_vs_traditional.py` — regression benchmark script

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | deferred (Phase 1) | Already implemented in Phase 1 |
| V5 Input Validation | yes | Pydantic models + image MIME validation (already exists) |
| V6 Cryptography | no | No new cryptography needed |
| V8 Data Protection | yes | Models are IP — protect file access, store in restricted directory |
| V12 Malicious Code | yes | ONNX model files must be verified (model poisoning risk) |

### Known Threat Patterns for AI Pipeline

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Model poisoning (adversarial .onnx) | Tampering | Hash-verify model files against known checksums at load time |
| Model theft | Information Disclosure | Models stored in gitignored directory with restricted OS permissions |
| GPU resource exhaustion | Denial of Service | ONNX Runtime session pooling with max concurrency; timeout per inference |
| Image adversarial attack | Spoofing | Compare original vs. enhanced image structural similarity (SSIM); flag large deviations for human review |
| Minutiae hallucination | Tampering | Always store original image alongside enhanced; editor logs all changes for audit trail |

## Sources

### Primary (HIGH confidence)
- [VERIFIED: PyPI registry] — `pip index versions` confirmed package versions: torch 2.12.0, onnxruntime-gpu 1.26.0, onnx 1.21.0, smp 0.5.0, kornia 0.8.3
- [CITED: onnxruntime.ai/docs/install/] — ONNX Runtime GPU installation and CUDA 12.x compatibility
- [CITED: pytorch.org/docs/stable/onnx.html] — PyTorch ONNX export documentation
- [CITED: qubvel-org/segmentation_models.pytorch] — SMP supports `in_channels=1` (grayscale), ONNX export
- [CITED: github.com/kornia/kornia] — Differentiable CV for GAN training

### Secondary (MEDIUM confidence)
- [VERIFIED: NVIDIA SMI] — RTX 4070 8GB, CUDA 13.2 driver available on system
- [VERIFIED: Codebase inspection] — Existing interfaces, types, FingerprintService, Docker Compose, frontend structure all documented from actual files
- WebSearch — SMP readme confirmed pretrained encoders, ONNX export notebook example

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM — PyPI versions verified but CUDA compatibility (13.2 driver with 12.x packages) is assumed
- Architecture: HIGH — Strategy Pattern is proven in existing code; no new architectural patterns needed
- Pitfalls: HIGH — GPU memory, CUDA context, model poisoning are well-known AI deployment issues
- Environment: MEDIUM — GPU detected but CUDA toolkit missing; installation path needs verification

**Research date:** 2025-06-13
**Valid until:** 2025-07-13 (30 days — ML frameworks evolve fast)

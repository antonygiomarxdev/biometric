---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 03-ia-generativa-burocracia
status: completed
last_updated: "2026-06-13T22:16:28.480Z"
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 19
  completed_plans: 17
  percent: 36
---

# State: Biometric

**Last updated:** 2026-06-13
**Current phase:** 03-ia-generativa-burocracia
**Status:** Plan 03 completed (Structured Dictamen Generation)

## Project Reference

See: `.planning/PROJECT.md`
**Core value:** Darle al perito forense una herramienta digital para identificar huellas latentes más rápido y con trazabilidad, manteniendo su criterio como autoridad final.
**Competitive Advantage:** Doble motor de Inteligencia Artificial (Visión Computacional para el procesamiento de la imagen + GenAI para la automatización de la burocracia judicial).

## Phase Progress

| Phase | Status | Progress |
|-------|--------|----------|
| 1. Flujo Core Forense | ✅ Completado | 100% |
| 2. IA Visión Computacional | 🏃‍♂️ En progreso | 67% (4/6 planes) |
| 3. IA Generativa (Dictámenes) | 🏃‍♂️ En progreso | 60% (3/5 planes) |
| 4. Despliegue On-Premise | ⏳ Pendiente | 0% |

## Current Work

Phase 3 (IA Generativa/Dictámenes) — Plan 03 completado. DictamenPericial Pydantic schema (Evidencia + DictamenPericial). Async report generator (generate_dictamen) con as_structured_llm, retry loop (max 3) en ValidationError, prompt en español legal (Perito informático Nicaragua). 14 tests, TDD. Next: Plan 04 (FastAPI router integration & genai endpoints).

## Completed Plans

| Phase | Plan | Summary |
|-------|------|---------|
| 02-ia-vision-computacional | 01 - Enhancement Spike | ✅ U-Net MobileNetV2 evaluado y recomendado. ONNX export validado. |
| 02-ia-vision-computacional | 02 - AI Infrastructure | ✅ AiConfig, ModelManager, GPU detection via PyTorch, AlgorithmOrigin AI values. |
| 02-ia-vision-computacional | 03 - AI Enhancement & Segmentation | ✅ SegmentationEnhancer, EnhancementEnhancer, factory AI-first con CPU fallback. |
| 02-ia-vision-computacional | 04 - DL Minutiae Extraction | ✅ ExtractionProcessor (pre/post), AiFeatureExtractor (IFeatureExtractor), 20 tests. |
| 03-ia-generativa-burocracia | 01 - LLM Factory | ✅ LLMFactory with ILLMProvider Protocol, Ollama/OpenAI providers, use_case profiles, 15 tests. |
| 03-ia-generativa-burocracia | 02 - Text-to-SQL | ✅ Read-only DB engine, NLP assistant with NLSQLTableQueryEngine, 5 tests (TDD). |
| 03-ia-generativa-burocracia | 03 - Structured Dictamen | ✅ DictamenPericial Pydantic schema, async report generator with as_structured_llm, retry logic, 14 tests (TDD). |

## Decisions Log

- **D-03 (Enhancement Architecture):** U-Net with MobileNetV2 encoder, L1+SSIM perceptual loss, 512×512 input, ImageNet pretrained weights. ONNX export opset 18.
- **D-04 (AI Infrastructure):** AiConfig frozen dataclass with env-var overrides; ModelManager singleton; ONNX Runtime provider auto-selects CUDAExecutionProvider; GPU detection via PyTorch at module level.
- **D-05 (Segmentation crop):** SegmentationEnhancer crops to mask bounding box instead of full-image — reduces downstream processing area.
- **D-06 (Enhancement letterbox):** EnhancementProcessor uses letterbox padding (aspect-ratio preserving) for enhancement model; SegmentationProcessor uses centre-pad for segmentation model.
- **D-07 (Extraction output scaling):** ExtractionProcessor computes scale factors from output spatial dims to canvas size before offset subtraction — handles model outputs at different resolutions than the 512×512 input canvas.
- **D-08 (LLM Factory Adapter Pattern):** ILLMProvider uses Protocol duck typing (not ABC) so providers are structurally typed. LLMFactory routes via config.llm_provider with use_case profiles (sql=120s timeout, default=60s). SecretStr for openai_api_key per T-03-01.
- **D-09 (Read-only DB enforcement):** SQLAlchemy engine uses `execution_options={"isolation_level": "AUTOCOMMIT", "postgresql_readonly": True}` to prevent writes at the driver level — enforces T-03-03.
- **D-10 (Double table scoping):** Both `SQLDatabase(include_tables=[...])` and `NLSQLTableQueryEngine(tables=[...])` specify `["peritajes", "evidencia"]`. This is intentional double-coverage: `include_tables` limits schema exposure, `tables` limits LLM prompt context (T-03-04).
- **D-11 (Dictamen schema & generator):** Prompt in Spanish legal language (not English) — LLM output follows prompt language. Retry only on `ValidationError` (other exceptions propagate). `PromptTemplate` not needed with `as_structured_llm.acomplete`. Mitigates T-03-06 (schema enforcement) and T-03-07 (exact ID/hash rule).

## Next Actions

1. ~~Plan 03-03: Dictamen pericial generation pipeline~~ ✅
2. Plan 03-04: Evaluate and refine generation quality + FastAPI router integration
3. Plan 03-05: Guardrails and production hardening

## Performance Metrics

| Phase | Plan | Duration | Notes |
|-------|------|----------|-------|
| Phase 02-ia-vision-computacional | P01 Enhancement Spike | 9 min | 3 tasks, 5 files |
| Phase 02-ia-vision-computacional | P02 AI Infrastructure | 5 commits | 5 files |
| Phase 02-ia-vision-computacional | P03 AI Segmentation & Enhancement | 4 min | 3 tasks, 7 files |
| Phase 02-ia-vision-computacional | P04 DL Minutiae Extraction | 8 min | 2 tasks (TDD), 3 files |
| Phase 03-ia-generativa-burocracia P01 | 3min | 2 tasks | 7 files |
| Phase 03-ia-generativa-burocracia P02 | 2min | 2 tasks (TDD) | 5 files |
| Phase 03-ia-generativa-burocracia P03 | 4min | 2 tasks (TDD) | 6 files |

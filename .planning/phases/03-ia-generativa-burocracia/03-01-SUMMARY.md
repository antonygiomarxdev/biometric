---
phase: 03-ia-generativa-burocracia
plan: 01
subsystem: ai
tags: [llm, ollama, openai, llama-index, adapter-pattern]
requires:
  - phase: 01-foundation
    provides: Core config dataclass and project structure
provides:
  - LLMFactory with Adapter/Strategy pattern for provider selection
  - OllamaProvider wrapping llama_index.llms.ollama
  - OpenAIProvider wrapping llama_index.llms.openai
  - Use-case profiles (sql=120s timeout, default=60s timeout)
  - LLM configuration with env-var overrides
affects: [03-02, 03-03]
tech-stack:
  added:
    - llama-index-core~=0.14
    - llama-index-llms-ollama~=0.10
    - llama-index-llms-openai~=0.7
  patterns:
    - Adapter/Strategy pattern via Protocol duck typing
    - Use-case profile configuration per provider
    - SecretStr for sensitive configuration values
key-files:
  created:
    - apps/backend/src/ai/llm.py (ILLMProvider, OllamaProvider, OpenAIProvider, LLMFactory)
    - apps/backend/tests/core/test_config.py
    - apps/backend/tests/ai/test_llm.py
  modified:
    - apps/backend/src/core/config.py
    - apps/backend/src/processing/enhancers/base.py
key-decisions:
  - "Use Protocol duck typing for ILLMProvider (structural subtyping, not ABC inheritance)"
  - "Top-level imports of Ollama/OpenAI for testability and clarity over inline imports"
  - "SecretStr for openai_api_key per threat model T-03-01"
  - "Ollama timeout: 120s for sql use_case, 60s for default"
requirements-completed: ["GENAI-01"]
duration: 3min
completed: 2026-06-13
---

# Phase 03: Generative AI for Bureaucracy Summary

**LLM Factory with Adapter/Strategy pattern supporting Ollama (local) and OpenAI (remote) providers, use-case profiles (sql vs default), and SecretStr-guarded API key configuration**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-13T21:59:05Z
- **Completed:** 2026-06-13T22:02:46Z
- **Tasks:** 2
- **Files modified:** 7 (5 created, 2 modified)

## Accomplishments

- Updated Config dataclass with LLM parameters: `llm_provider`, `local_model_name`, `remote_model_name`, `openai_api_key` (SecretStr)
- Implemented `ILLMProvider` Protocol for duck-typed provider interface
- Implemented `OllamaProvider` with use-case-based timeout (120s for SQL, 60s for default)
- Implemented `OpenAIProvider` with remote model and SecretStr API key extraction
- Implemented `LLMFactory` with provider registry, use_case routing, and ValueError on unknown provider
- All 15 tests pass (8 config tests, 7 LLM factory tests)

## Task Commits

Each task was committed atomically following TDD (RED → GREEN):

1. **Task 1: Update Configuration for LLM**
   - `feb7fcc` (test: RED — failing tests for LLM config)
   - `81e84da` (feat: GREEN — added LLM params to Config)
2. **Task 2: Implement LLM Factory and Providers**
   - `0aafe13` (test: RED — failing tests for LLM factory)
   - `a124c57` (feat: GREEN — implemented ILLMProvider, providers, factory)

## Files Created/Modified

- `apps/backend/src/core/config.py` - Added `llm_provider`, `local_model_name`, `remote_model_name`, `openai_api_key` (SecretStr) fields
- `apps/backend/src/ai/llm.py` - NEW: `ILLMProvider` protocol, `OllamaProvider`, `OpenAIProvider`, `LLMFactory`
- `apps/backend/tests/core/test_config.py` - NEW: 8 tests for LLM config defaults and env overrides
- `apps/backend/tests/ai/test_llm.py` - NEW: 7 tests for factory routing, use_case profiles, and error handling
- `apps/backend/src/processing/enhancers/base.py` - Fixed missing import of `IEnhancer` (Rule 3 blocking fix)

## Decisions Made

- **Protocol over ABC:** `ILLMProvider` uses Python Protocol for structural subtyping — implementations don't need to inherit, they just need the right method signature. Clean Architecture Dependency Inversion.
- **Top-level imports:** `Ollama` and `OpenAI` are imported at module level instead of inline. Improves testability (patch works at module level) and code clarity.
- **SecretStr for API key:** Follows threat model T-03-01 to prevent accidental logging of the OpenAI API key.
- **Timeout profiles:** SQL generation gets 120s timeout (schema-aware queries take longer), other tasks get 60s.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed missing IEnhancer import in base.py**
- **Found during:** Task 1 (test execution)
- **Issue:** `src/processing/enhancers/base.py` uses `IEnhancer` in class definition but never imports it, causing ImportError that blocked all test collection
- **Fix:** Added `from src.core.interfaces import IEnhancer` to `base.py`
- **Files modified:** `apps/backend/src/processing/enhancers/base.py`
- **Verification:** All 15 tests now collect and pass
- **Committed in:** `feb7fcc` (part of Task 1 RED commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary fix to enable test execution. No scope creep.

## Issues Encountered

- Pre-existing missing import in `src/processing/enhancers/base.py` blocked test collection — fixed via Rule 3.

## Threat Flags

None — all files align with the threat model. `openai_api_key` uses `SecretStr` per T-03-01 mitigation.

## Success Criteria Verification

- [x] LLMFactory with ILLMProvider support
- [x] OllamaProvider and OpenAIProvider implementations
- [x] use_case profiles for different tasks (sql / default)
- [x] Clean Architecture (domain code never imports providers directly)
- [x] Clean Code (single responsibility, meaningful names)
- [x] Full type annotations, no Any
- [x] SUMMARY.md created

## Next Phase Readiness

- LLM infrastructure complete. Ready for Phase 03-02 (Text-to-SQL query engine integration).
- Factory pattern allows adding new providers (Azure, Anthropic) by registering them in `_providers` dict.

---

*Phase: 03-ia-generativa-burocracia*
*Completed: 2026-06-13*

# Testing: BioSecure Gov

**Last updated:** 2025-06-12

## Test Framework
- **pytest** 7.4+ with pytest-asyncio, pytest-cov
- **Config:** pytest.ini with verbose, strict markers, -ra summary
- **Python:** 3.12+
- **Runner:** `uv run pytest` or `make test`

## Test Structure
```
apps/backend/tests/
├── conftest.py              # Shared fixtures
├── test_api_e2e.py          # E2E API tests (mocked services)
├── test_extractor.py        # Extractor unit tests
├── test_integration.py      # Integration tests (mocked DB)
├── test_models.py           # Model/schema tests
├── test_performance.py      # Performance benchmarks
└── test_vectorizer.py       # Vectorizer tests
```

## Testing Patterns

### E2E API Tests
- Uses TestClient from FastAPI
- Mocks fingerprint_service and comparison_service entirely
- Generates synthetic image bytes via cv2.imencode
- Tests: health check, extract, register, identify
- Limitation: Tests verify HTTP layer only

### Extractor Tests
- Tests SkeletonMinutiaeExtractor directly
- Uses synthetic skeleton images
- Tests crossing number detection, filtering

### Integration Tests
- Mocks database layer
- Tests end-to-end service flow

### Performance Tests
- Marked with @pytest.mark.performance
- Benchmarks for processing pipeline

## Coverage
- **Tool:** pytest-cov
- **Target:** src/ (excluding tests and cache)
- **HTML reports:** htmlcov/
- **Known gaps:**
  - No frontend tests
  - No real database integration tests
  - GPU paths not tested
  - Storage/object_storage.py not tested
  - CLI commands not tested

## Test Tags
- slow, integration, performance

## Running Tests
```bash
make test           # All tests
make test-cov       # With coverage
uv run pytest -m "not slow"  # Fast tests only
```

## Test Quality Observations
- No frontend testing infrastructure
- API tests mock entire service layer (low real-world confidence)
- No fixtures for real fingerprint images
- No database test containers

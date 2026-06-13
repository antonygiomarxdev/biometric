# Conventions: BioSecure Gov

**Last updated:** 2025-06-12

## Backend Conventions

### Python Style
- **Line length:** 100 chars (pyproject.toml)
- **Python target:** 3.12
- **Linter:** ruff (E, W, F, I, B, C4, UP)
- **Formatter:** ruff (not fully configured)
- **Type hints:** Used throughout, disallow_untyped_defs = false

### Code Patterns

**Frozen Dataclasses for Domain Types:**
```python
@dataclass(frozen=True)
class MinutiaCandidate:
    x: int; y: int; angle: float; type: MinutiaType; confidence: float; algorithm: AlgorithmOrigin
```

**Abstract Base Classes:**
```python
class IEnhancer(ABC):
    @abstractmethod
    def enhance(self, image: np.ndarray, resize: bool = True) -> np.ndarray: ...
```

**Global Singleton Instances:**
- config, fingerprint_service, db_manager, repository, storage, metrics, vector_index

**Dependency Injection in Constructors:**
```python
class FingerprintService:
    def __init__(self, enhancer=None, extractor=None, normalizer=None):
        self.enhancer = enhancer or create_enhancer()
```

**Structured Logging:**
- Named loggers per module (api.rest, processing, storage)
- Info/Warning/Error with context data

**Metrics Decorator:**
```python
@timed("index_add_vector")
def add(self, vector): ...
```

**Thread Pool for CPU-bound Work:**
```python
executor = ThreadPoolExecutor(max_workers=4)
fingerprint = await loop.run_in_executor(executor, service.process_image, image, ...)
```

### Error Handling
- API: Catch HTTPException, ValueError, then generic Exception
- Service: Raise ValueError for invalid input
- Storage: Catch DB errors with logging, non-blocking startup
- Startup: App starts even if DB unavailable

## Frontend Conventions

### TypeScript Style
- Strict mode, ESLint with typescript-eslint
- React 19 functional components + hooks

### UI Patterns
- shadcn/ui-style components with className forwarding
- Tailwind CSS utility classes
- lucide-react icons
- Dark mode default (class dark on root)
- Custom toast notification system

### API Client Pattern
```typescript
const res = await DefaultService.identifyFingerprintIdentifyPost({ file });
```

### Error Handling
- ApiError for HTTP errors with status code mapping
- Network error detection via message matching

## Project Conventions
- No conventional commits enforced
- Mix of Spanish and English in comments/docs
- Docker Compose for local development

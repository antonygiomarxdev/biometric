---
phase: 21-mcc-integration
reviewed: 2026-06-17T12:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - apps/backend/src/core/types.py
  - apps/backend/tests/domain/test_mcc_types.py
  - apps/backend/tests/domain/__init__.py
findings:
  critical: 0
  warning: 1
  info: 3
  total: 4
status: issues_found
---

# Phase 21: Code Review Report — Task 2

**Reviewed:** 2026-06-17T12:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Task 2 adds `MccCylinderHit` and `MccPersonHit` frozen dataclasses with slots to `core/types.py`, along with unit tests and an `__init__.py` package marker under `tests/domain/`. The core types correctly implement the plan requirements: frozen dataclasses with slots, correct field sets, and proper use of `field(default_factory=list)` for mutable defaults.

All 4 tests pass against Python 3.12.3.

One warning and three info items were found — no critical issues. The warning concerns a test reliability gap (broad exception catch), and the info items cover docstring inconsistencies and test coverage misses.

---

## Warnings

### WR-01: Frozen-property test catches overly broad `Exception`

**File:** `apps/backend/tests/domain/test_mcc_types.py:30`
**Issue:** `test_mcc_cylinder_hit_is_frozen` uses `pytest.raises(Exception):` to verify the frozen dataclass rejects attribute assignment. Catching the root `Exception` class is too broad — a `NameError`, `TypeError`, or any other unrelated error inside the `with` block would also be silently caught, making the test pass for the wrong reason. Frozen dataclass writes raise `dataclasses.FrozenInstanceError` (a subclass of `AttributeError`).

This affects test reliability: if refactoring accidentally breaks the test code itself (e.g., renames the variable `hit`), the test would still pass instead of failing immediately.

**Fix:** Narrow the exception to `AttributeError` or specifically to `dataclasses.FrozenInstanceError`:

```python
import dataclasses

def test_mcc_cylinder_hit_is_frozen() -> None:
    hit = MccCylinderHit(
        person_id="p1",
        fingerprint_id="f1",
        capture_id="c1",
        similarity=0.9,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        hit.person_id = "p2"  # type: ignore[misc]
```

---

## Info

### IN-01: `MccPersonHit` docstring says "per-fingerprint" but type is per-person

**File:** `apps/backend/src/core/types.py:287`
**Issue:** The `MccPersonHit` docstring begins with "Aggregated **per-fingerprint** match result" but the type carries `person_id` (not `fingerprint_id`) and `contributing_fingerprints` (plural), indicating it aggregates across fingerprints at the person level. Compare with `PersonHit` (line 266) which correctly says "Aggregated **person-level** result from chunk search." The mismatch could mislead future developers about the aggregation granularity.

**Fix:** Align the docstring with the type's actual semantics:

```python
@dataclass(frozen=True, slots=True)
class MccPersonHit:
    """Aggregated person-level result from MCC cylinder search.
    ...
    """
```

### IN-02: Docstring references `score_normalization` parameter that doesn't exist

**File:** `apps/backend/src/core/types.py:289-291`
**Issue:** The `MccPersonHit` docstring says "When ``score_normalization == 'fingerprint'`` (default), the caller divides by the number of enrolled cylinders to remove population bias." This references a `score_normalization` parameter that doesn't exist on the class. While this describes intended usage for downstream callers of the *repository* method (defined in `config.py` as `MccMatchingConfig.score_normalization`), putting it here creates a dangling reference. A reader scanning only this file won't find the parameter.

**Fix:** Either remove the `score_normalization` reference from the docstring (keeping only the `total_score` description), or rephrase it as a cross-reference: "Callers should divide by the number of enrolled cylinders according to their normalization strategy (see `MccMatchingConfig.score_normalization`)."

### IN-03: Test `test_mcc_person_hit_score_sums` doesn't test summation

**File:** `apps/backend/tests/domain/test_mcc_types.py:34-37`
**Issue:** The test is named `test_mcc_person_hit_score_sums` implying it validates some summation or aggregation behavior, but it only tests that field assignment (`total_score=2.5`, `hits=5`) returns the expected values. No summing logic is exercised. The test also doesn't cover the `contributing_fingerprints` field at all, leaving the default-factory behavior and list mutation untested.

**Fix:** Rename the test to reflect what it actually tests (e.g., `test_mcc_person_hit_assigns_fields`) and either:

- Add a second test that exercises real aggregation logic (when the service/repository layer that creates `MccPersonHit` exits), or
- At minimum, add assertions for `contributing_fingerprints`:

```python
def test_mcc_person_hit_defaults_contributing_fingerprints(self) -> None:
    hit = MccPersonHit(person_id="p1", total_score=2.5, hits=5)
    assert hit.contributing_fingerprints == []
```

---

_Reviewed: 2026-06-17T12:00:00Z_
_Reviewer: gsd-code-reviewer_
_Depth: standard_

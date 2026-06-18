"""
Development-only debug logger.

Opt-in via env var ``DEBUG_LOG=1``. Emits a single, structured line per
event to stderr. No state, no buffering, no dependencies.

Usage::

    from src.dev.logger import dev_log
    dev_log("enroll", person_id=str(pid), slot_id=str(sid), cylinders=42)

Output is gated by env var so production runs are unaffected.

The dev_log call is a no-op when DEBUG_LOG is unset, so calling it from
hot paths is safe. When unset, the function returns in <100ns.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any


_ENABLED: bool = os.getenv("DEBUG_LOG", "").lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """Return whether dev_log will emit anything. Read once at import time."""
    return _ENABLED


def _now_ms() -> int:
    return int(time.time() * 1000)


def dev_log(event: str, **fields: Any) -> None:
    """Emit a single debug line tagged with event name and structured fields.

    Format: ``[dev] {iso_ms} event={name} k1=v1 k2=v2 ...``

    The event name is required; fields are arbitrary string-coercible
    values. Booleans, ints, floats, and short strings render verbatim;
    anything else falls back to ``str(value)`` truncated to 200 chars.
    """
    if not _ENABLED:
        return
    parts: list[str] = [f"[dev] {_now_ms()} event={event}"]
    for k, v in fields.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        else:
            s = str(v)
            if len(s) > 200:
                s = s[:197] + "..."
            parts.append(f"{k}={s}")
    sys.stderr.write(" ".join(parts) + "\n")
    sys.stderr.flush()

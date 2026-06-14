"""
Test configuration for service-layer unit tests.

These tests are fully mocked and don't need numpy, cv2, or any
GPU/ML infrastructure.  This conftest keeps the test environment
light and avoids the pre-existing numpy/Coverage conflict.
"""

from __future__ import annotations

import os

os.environ.setdefault("FORCE_CPU", "1")
os.environ.setdefault("AI_USE_GPU", "false")
os.environ.setdefault("ENABLE_AI_TRACING", "false")

import os
from unittest import mock

import pytest

from src.gpu_utils import (
    GPU_AVAILABLE,
    _detect_gpu,
    get_device_info,
    is_gpu_enabled,
)

def test_gpu_available_flag():
    """GPU_AVAILABLE is a bool (import-level check)."""
    assert isinstance(GPU_AVAILABLE, bool)

def test_gpu_detection_returns_bool():
    """is_gpu_enabled() returns a boolean."""
    assert isinstance(is_gpu_enabled(), bool)

def test_get_device_info_returns_dict():
    """get_device_info() returns dict with keys backend, device, available."""
    info = get_device_info()
    assert isinstance(info, dict)
    assert "backend" in info
    assert "device" in info
    assert "available" in info

@mock.patch.dict(os.environ, {"FORCE_CPU": "1"})
def test_force_cpu_disables_gpu():
    """When FORCE_CPU=1, is_gpu_enabled() returns False."""
    assert is_gpu_enabled() is False

def test_gpu_detection_without_torch():
    """GPU detection returns bool without raising ImportError when torch is absent."""
    import sys
    
    # Store torch if present
    torch_mod = sys.modules.get('torch')
    
    # Force ImportError for torch
    sys.modules['torch'] = None
    
    try:
        assert isinstance(_detect_gpu(), bool)
    finally:
        # Restore torch
        if torch_mod is not None:
            sys.modules['torch'] = torch_mod
        else:
            del sys.modules['torch']

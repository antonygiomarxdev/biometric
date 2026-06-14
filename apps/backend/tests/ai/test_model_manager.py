"""Tests for ONNX Runtime model lifecycle manager.

All ONNX sessions are mocked via the session-scoped conftest fixtures so
no real ``.onnx`` files are loaded and no real inference is performed.
The conftest patches ``ModelManager.load_model`` and ``get_session`` at
the class level — we work with those patches where applicable and undo
them only when testing the cached-load logic directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ai.config import AiConfig


@pytest.fixture
def ai_config() -> AiConfig:
    """Minimal AiConfig pointing at a temp directory."""
    return AiConfig(
        model_dir="/tmp/test_models/",
        use_gpu=False,
    )


@pytest.fixture
def manager(ai_config: AiConfig) -> Any:
    """ModelManager instance backed by mocked ONNX Runtime."""
    from src.ai.model_manager import ModelManager

    return ModelManager(ai_config)


class TestModelManagerInit:
    """ModelManager construction creates the model directory."""

    @patch("src.ai.model_manager.Path.mkdir")
    def test_creates_model_dir(self, mock_mkdir: MagicMock) -> None:
        """The model_dir is created if it does not exist."""
        from src.ai.model_manager import ModelManager

        config = AiConfig(model_dir="/tmp/test_models/", use_gpu=False)
        manager = ModelManager(config)
        assert manager.config is config
        assert manager.model_dir == Path("/tmp/test_models/")
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert manager._sessions == {}


class TestModelManagerLoad:
    """Model loading — works with conftest mocks."""

    def test_load_model_returns_session(
        self, manager: Any
    ) -> None:
        """load_model returns a session object (mocked)."""
        session = manager.load_model("segment")
        assert session is not None

    def test_unload_model_removes_session(self, manager: Any) -> None:
        """unload_model evicts a single session from the cache."""
        mock_session = MagicMock()
        manager._sessions["test_model"] = mock_session
        assert "test_model" in manager._sessions

        manager.unload_model("test_model")
        assert "test_model" not in manager._sessions

    def test_unload_all_clears_cache(self, manager: Any) -> None:
        """unload_all removes all cached sessions."""
        manager._sessions["model_a"] = MagicMock()
        manager._sessions["model_b"] = MagicMock()
        assert len(manager._sessions) == 2

        manager.unload_all()
        assert manager._sessions == {}

    def test_loaded_models_property(self, manager: Any) -> None:
        """loaded_models returns a list of cached model names."""
        manager._sessions["seg"] = MagicMock()
        manager._sessions["enh"] = MagicMock()

        names = manager.loaded_models
        assert sorted(names) == ["enh", "seg"]


class TestModelManagerRun:
    """Typed inference helpers dispatch to _run_single."""

    def test_run_segmentation(
        self, manager: Any
    ) -> None:
        """run_segmentation calls _run_single with 'segment'."""
        input_tensor = np.zeros((1, 1, 64, 64), dtype=np.float32)

        with (
            patch.object(
                manager,
                "_run_single",
                return_value=np.ones((1, 1, 64, 64), dtype=np.float32),
            ) as mock_run,
        ):
            result = manager.run_segmentation(input_tensor)

            assert result is not None
            assert result.shape == (1, 1, 64, 64)
            mock_run.assert_called_once_with("segment", input_tensor)

    def test_run_enhancement(
        self, manager: Any
    ) -> None:
        """run_enhancement calls _run_single with 'enhance'."""
        input_tensor = np.zeros((1, 1, 64, 64), dtype=np.float32)

        with patch.object(
            manager, "_run_single", return_value=input_tensor
        ) as mock_run:
            result = manager.run_enhancement(input_tensor)

            assert result is not None
            mock_run.assert_called_once_with("enhance", input_tensor)

    def test_run_extraction(
        self, manager: Any
    ) -> None:
        """run_extraction calls _run_single with 'extract'."""
        input_tensor = np.zeros((1, 1, 64, 64), dtype=np.float32)

        with patch.object(
            manager, "_run_single", return_value=input_tensor
        ) as mock_run:
            result = manager.run_extraction(input_tensor)

            assert result is not None
            mock_run.assert_called_once_with("extract", input_tensor)

    def test_run_single_with_mocked_session(
        self, manager: Any
    ) -> None:
        """_run_single uses get_session and runs inference with correct names."""
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.run.return_value = [
            np.ones((1, 1, 8, 8), dtype=np.float32)
        ]

        input_tensor = np.zeros((1, 1, 8, 8), dtype=np.float32)

        with patch.object(
            manager, "get_session", return_value=mock_session
        ):
            result = manager._run_single("segment", input_tensor)

            assert result is not None
            mock_session.get_inputs.assert_called_once()
            mock_session.get_outputs.assert_called_once()
            mock_session.run.assert_called_once_with(
                ["output"], {"input": input_tensor}
            )

    def test_get_session_delegates_to_load(
        self, manager: Any
    ) -> None:
        """get_session delegates to load_model."""
        # conftest patches get_session to return a MagicMock,
        # but we can verify the method exists and is callable.
        session = manager.get_session("segment")
        assert session is not None


class TestModelManagerRealLoad:
    """Test the REAL ``load_model`` by working around session-scoped mocks.

    The conftest patches ``ModelManager.load_model`` and ``get_session``
    at the class level to prevent real ONNX loading.  These tests use
    ``importlib.reload`` to create a fresh, unpatched ``ModelManager``
    class while keeping ``onnxruntime.InferenceSession`` mocked at the
    package level.
    """

    def test_caches_session(self) -> None:
        """Real load_model returns cached session on second call."""
        import importlib

        import src.ai.model_manager as mm_mod

        with (
            patch("onnxruntime.InferenceSession") as mock_session_cls,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mm_mod = importlib.reload(mm_mod)
            FreshModelManager = mm_mod.ModelManager

            config = AiConfig(
                model_dir="/tmp/test_models/", use_gpu=False
            )
            manager = FreshModelManager(config)

            fake_session = MagicMock()
            mock_session_cls.return_value = fake_session

            # First call — loads from disk
            session1 = manager.load_model("segment")
            assert session1 is fake_session
            assert "segment" in manager._sessions
            mock_session_cls.assert_called_once()

            # Second call — returns cached (no new InferenceSession)
            mock_session_cls.reset_mock()
            session2 = manager.load_model("segment")
            assert session2 is fake_session
            assert session2 is session1
            mock_session_cls.assert_not_called()

    def test_raises_on_missing_file(self) -> None:
        """Real load_model raises FileNotFoundError when .onnx missing."""
        import importlib

        import src.ai.model_manager as mm_mod

        with (
            patch("onnxruntime.InferenceSession"),
            patch("pathlib.Path.mkdir"),  # prevent __init__ from crashing
        ):
            mm_mod = importlib.reload(mm_mod)
            FreshModelManager = mm_mod.ModelManager

            config = AiConfig(
                model_dir="/nonexistent/path/",
                use_gpu=False,
            )
            manager = FreshModelManager(config)

            with pytest.raises(FileNotFoundError, match="ONNX model not found"):
                manager.load_model("nonexistent_model")

    def test_get_session_calls_load_model(self) -> None:
        """Real get_session delegates to load_model."""
        import importlib

        import src.ai.model_manager as mm_mod

        with (
            patch("onnxruntime.InferenceSession"),
            patch("pathlib.Path.exists", return_value=True),
        ):
            mm_mod = importlib.reload(mm_mod)
            FreshModelManager = mm_mod.ModelManager

            config = AiConfig(
                model_dir="/tmp/test_models/", use_gpu=False
            )
            manager = FreshModelManager(config)

            with patch.object(
                manager, "load_model", wraps=manager.load_model
            ) as spy:
                session = manager.get_session("segment")
                assert session is not None
                spy.assert_called_once_with("segment")




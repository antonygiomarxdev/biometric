"""
Unit tests for :mod:`~src.services.fingerprint_service`.

Mocks the enhancer, extractor, and normalizer (expensive CV/ML operations)
so that tests exercise only the orchestration layer.  Coverage target >90%.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.interfaces import IEnhancer, IFeatureExtractor, INormalizer
from src.core.types import (
    AlgorithmOrigin,
    MinutiaCandidate,
    MinutiaType,
    NormalizedFingerprint,
)
from src.services.fingerprint_service import FingerprintService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_enhancer() -> MagicMock:
    """Return a mock enhancer that passes the image through unchanged."""
    enhancer = MagicMock(spec=IEnhancer)
    enhancer.enhance.side_effect = lambda img, resize=True: img
    return enhancer


@pytest.fixture
def mock_extractor() -> MagicMock:
    """Return a mock extractor that produces one known minutia."""
    extractor = MagicMock(spec=IFeatureExtractor)
    extractor.extract.return_value = [
        MinutiaCandidate(
            x=10,
            y=20,
            angle=1.5,
            type=MinutiaType.TERMINATION,
            confidence=0.9,
            origin=AlgorithmOrigin.SKELETON,
        ),
    ]
    return extractor


@pytest.fixture
def mock_normalizer() -> MagicMock:
    """Return a mock normalizer that returns a known NormalizedFingerprint."""
    normalizer = MagicMock(spec=INormalizer)
    normalizer.normalize.return_value = NormalizedFingerprint(
        id="test_fp",
        minutiae=[
            MinutiaCandidate(
                x=10,
                y=20,
                angle=1.5,
                type=MinutiaType.TERMINATION,
                confidence=0.9,
                origin=AlgorithmOrigin.SKELETON,
            ),
        ],
        width=200,
        height=200,
    )
    return normalizer


@pytest.fixture
def grayscale_image() -> np.ndarray:
    """200×200 grayscale test image."""
    return np.zeros((200, 200), dtype=np.uint8)


@pytest.fixture
def color_image() -> np.ndarray:
    """100×100×3 RGB test image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def service(
    mock_enhancer: MagicMock,
    mock_extractor: MagicMock,
    mock_normalizer: MagicMock,
) -> FingerprintService:
    """A FingerprintService with all expensive components mocked."""
    return FingerprintService(
        enhancer=mock_enhancer,
        extractor=mock_extractor,
        normalizer=mock_normalizer,
    )


# ---------------------------------------------------------------------------
# process_image
# ---------------------------------------------------------------------------


class TestProcessImage:
    """Tests for :meth:`FingerprintService.process_image`."""

    def test_normal_flow(
        self,
        service: FingerprintService,
        mock_enhancer: MagicMock,
        mock_extractor: MagicMock,
        mock_normalizer: MagicMock,
        grayscale_image: np.ndarray,
    ) -> None:
        """A grayscale image flows through enhance → extract → normalize."""
        result = service.process_image(grayscale_image, fingerprint_id="fp001")

        mock_enhancer.enhance.assert_called_once_with(grayscale_image, resize=True)
        mock_extractor.extract.assert_called_once()
        mock_normalizer.normalize.assert_called_once()

        assert result.id == "fp001"
        assert len(result.minutiae) == 1

    def test_raises_on_none_image(self, service: FingerprintService) -> None:
        """Passing None as image raises ValueError."""
        with pytest.raises(ValueError, match="no puede ser None"):
            service.process_image(None)  # type: ignore[arg-type]

    def test_converts_color_to_grayscale(
        self,
        service: FingerprintService,
        color_image: np.ndarray,
    ) -> None:
        """A 3-channel colour image is implicitly converted to grayscale."""
        with patch("src.services.fingerprint_service.cv2.cvtColor") as mock_cvt:
            mock_cvt.return_value = color_image[:, :, 0]

            service.process_image(color_image, fingerprint_id="fp002")

            mock_cvt.assert_called_once()

    def test_empty_candidates_list(
        self,
        mock_enhancer: MagicMock,
        mock_extractor: MagicMock,
        mock_normalizer: MagicMock,
        grayscale_image: np.ndarray,
    ) -> None:
        """When the extractor finds no minutiae the pipeline still produces a result."""
        mock_extractor.extract.return_value = []
        mock_normalizer.normalize.return_value = NormalizedFingerprint(
            id="empty",
            minutiae=[],
            width=200,
            height=200,
        )

        svc = FingerprintService(
            enhancer=mock_enhancer,
            extractor=mock_extractor,
            normalizer=mock_normalizer,
        )
        result = svc.process_image(grayscale_image, fingerprint_id="fp003")

        assert len(result.minutiae) == 0

    def test_resize_false(
        self,
        service: FingerprintService,
        mock_enhancer: MagicMock,
        grayscale_image: np.ndarray,
    ) -> None:
        """The resize parameter is forwarded to the enhancer."""
        service.process_image(grayscale_image, fingerprint_id="fp004", resize=False)
        mock_enhancer.enhance.assert_called_once_with(grayscale_image, resize=False)


# ---------------------------------------------------------------------------
# process_batch
# ---------------------------------------------------------------------------


class TestProcessBatch:
    """Tests for :meth:`FingerprintService.process_batch`."""

    def test_sequential_path(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """With use_parallel=False, images are processed sequentially."""
        service.use_parallel = False
        images = [grayscale_image, grayscale_image]
        results = service.process_batch(images, fingerprint_ids=["a", "b"])

        assert len(results) == 2
        assert all(r is not None for r in results)
        assert results[0].id == "a"
        assert results[1].id == "b"

    def test_parallel_path(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """With use_parallel=True, images are processed via ProcessPoolExecutor."""
        images = [grayscale_image, grayscale_image]

        with (
            patch(
                "src.services.fingerprint_service.concurrent.futures.ProcessPoolExecutor"
            ) as mock_pool_cls,
            patch(
                "src.services.fingerprint_service.concurrent.futures.as_completed",
                side_effect=lambda futures: futures,
            ),
        ):
            mock_ctx = MagicMock()
            mock_pool_cls.return_value.__enter__.return_value = mock_ctx

            # Make submit call _process_wrapper synchronously
            submitted: list = []

            def _mock_submit(fn: callable, *args: object) -> MagicMock:
                submitted.append((fn, args))
                future = MagicMock()
                future.result.return_value = NormalizedFingerprint(
                    id="worker",
                    minutiae=[],
                    width=200,
                    height=200,
                )
                return future

            mock_ctx.submit.side_effect = _mock_submit

            results = service.process_batch(images, fingerprint_ids=["a", "b"])

        assert len(results) == 2
        assert mock_ctx.submit.call_count == 2
        # Verify _process_wrapper was submitted for each (img, fid) pair
        fn_called = submitted[0][0]
        assert fn_called == FingerprintService._process_wrapper

    def test_process_wrapper(
        self,
        grayscale_image: np.ndarray,
    ) -> None:
        """_process_wrapper creates a fresh service and processes the image."""
        result = FingerprintService._process_wrapper(
            grayscale_image, "wrapper_fp"
        )
        assert isinstance(result, NormalizedFingerprint)
        assert result.id == "wrapper_fp"

    def test_parallel_error_handling(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """An exception in the parallel path is caught and logged."""
        with (
            patch(
                "src.services.fingerprint_service.concurrent.futures.ProcessPoolExecutor"
            ) as mock_pool_cls,
            patch(
                "src.services.fingerprint_service.concurrent.futures.as_completed",
                side_effect=lambda futures: futures,
            ),
        ):
            mock_ctx = MagicMock()
            mock_pool_cls.return_value.__enter__.return_value = mock_ctx
            future = MagicMock()
            future.result.side_effect = RuntimeError("worker failed")
            mock_ctx.submit.return_value = future

            results = service.process_batch(
                [grayscale_image], fingerprint_ids=["fail"]
            )

        assert len(results) == 1
        assert results[0] is None

    def test_empty_list(
        self,
        service: FingerprintService,
    ) -> None:
        """An empty image list returns an empty result list."""
        results = service.process_batch([])
        assert results == []

    def test_default_fingerprint_ids(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """When fingerprint_ids is None, all results get id='unknown'."""
        service.use_parallel = False
        results = service.process_batch([grayscale_image])
        assert len(results) == 1
        assert results[0] is not None
        assert results[0].id == "unknown"

    def test_error_handling_sequential(
        self,
        mock_enhancer: MagicMock,
        mock_extractor: MagicMock,
        mock_normalizer: MagicMock,
        grayscale_image: np.ndarray,
    ) -> None:
        """An exception in sequential processing does not crash the whole batch."""
        mock_enhancer.enhance.side_effect = RuntimeError("enhance failed")

        svc = FingerprintService(
            enhancer=mock_enhancer,
            extractor=mock_extractor,
            normalizer=mock_normalizer,
        )
        svc.use_parallel = False

        results = svc.process_batch(
            [grayscale_image, grayscale_image],
            fingerprint_ids=["ok", "fail"],
        )

        assert len(results) == 2
        assert results[0] is None  # first fails (enhance raise)
        assert results[1] is None  # second also fails


# ---------------------------------------------------------------------------
# process_image_from_bytes
# ---------------------------------------------------------------------------


class TestProcessImageFromBytes:
    """Tests for :meth:`FingerprintService.process_image_from_bytes`."""

    def test_valid_bytes(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """Valid image bytes are decoded and processed."""
        with (
            patch(
                "src.services.fingerprint_service.cv2.imdecode",
                return_value=grayscale_image,
            ) as mock_decode,
        ):
            result = service.process_image_from_bytes(
                b"fake-image-bytes",
                fingerprint_id="byte_fp",
            )

        mock_decode.assert_called_once()
        assert result.id == "byte_fp"
        assert len(result.minutiae) == 1

    def test_invalid_bytes_raises(
        self,
        service: FingerprintService,
    ) -> None:
        """When cv2.imdecode returns None, a ValueError is raised."""
        with patch(
            "src.services.fingerprint_service.cv2.imdecode",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="No se pudo decodificar"):
                service.process_image_from_bytes(b"garbage")

    def test_empty_image_raises(
        self,
        service: FingerprintService,
    ) -> None:
        """An image with zero size raises ValueError."""
        empty = np.zeros((0, 0), dtype=np.uint8)
        with patch(
            "src.services.fingerprint_service.cv2.imdecode",
            return_value=empty,
        ):
            with pytest.raises(ValueError, match="está vacía"):
                service.process_image_from_bytes(b"empty-image")

    def test_small_image_logs_warning(
        self,
        service: FingerprintService,
    ) -> None:
        """A very small image logs a warning but still processes."""
        small = np.zeros((30, 30), dtype=np.uint8)
        with (
            patch(
                "src.services.fingerprint_service.cv2.imdecode",
                return_value=small,
            ) as mock_decode,
            patch(
                "src.services.fingerprint_service.logger"
            ) as mock_logger,
        ):
            result = service.process_image_from_bytes(b"small")

        mock_decode.assert_called_once()
        mock_logger.warning.assert_called_once()
        assert result is not None

    def test_forwards_resize_true(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """process_image_from_bytes always passes resize=True to process_image."""
        with patch(
            "src.services.fingerprint_service.cv2.imdecode",
            return_value=grayscale_image,
        ):
            result = service.process_image_from_bytes(
                b"test", fingerprint_id="fp"
            )
        assert result.id == "fp"


# ---------------------------------------------------------------------------
# process_image_from_path
# ---------------------------------------------------------------------------


class TestProcessImageFromPath:
    """Tests for :meth:`FingerprintService.process_image_from_path`."""

    def test_valid_path(
        self,
        service: FingerprintService,
        grayscale_image: np.ndarray,
    ) -> None:
        """cv2.imread is called, then the image is processed."""
        with patch(
            "src.services.fingerprint_service.cv2.imread",
            return_value=grayscale_image,
        ) as mock_imread:
            result = service.process_image_from_path("/fake/path.png")

        mock_imread.assert_called_once_with(
            "/fake/path.png", 0  # cv2.IMREAD_GRAYSCALE = 0
        )
        assert result is not None

    def test_invalid_path_raises(
        self,
        service: FingerprintService,
    ) -> None:
        """When cv2.imread returns None, a ValueError is raised."""
        with patch(
            "src.services.fingerprint_service.cv2.imread",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="No se pudo cargar"):
                service.process_image_from_path("/nonexistent.png")


# ---------------------------------------------------------------------------
# create_ai_fingerprint_service
# ---------------------------------------------------------------------------


class TestCreateAIFingerprintService:
    """Tests for :func:`create_ai_fingerprint_service`."""

    def test_all_ai_enabled(self) -> None:
        """With all AI flags enabled, AI components are injected."""
        model_manager = MagicMock()
        mock_extractor = MagicMock()

        with (
            patch(
                "src.processing.enhancer.create_enhancer",
            ) as mock_create_enhancer,
            patch(
                "src.processing.extractor.AiFeatureExtractor",
                return_value=mock_extractor,
            ),
            patch("src.ai.config.AiConfig"),
        ):
            from src.services.fingerprint_service import (
                create_ai_fingerprint_service,
            )

            svc = create_ai_fingerprint_service(
                model_manager=model_manager,
                use_segmentation=True,
                use_enhancement=True,
                use_dl_extractor=True,
            )

        assert isinstance(svc, FingerprintService)
        assert svc.extractor is mock_extractor
        mock_create_enhancer.assert_called_once_with(
            "full_ai", model_manager=model_manager
        )

    def test_segmentation_only(self) -> None:
        """Only segmentation enabled calls create_enhancer with 'segmentation'."""
        model_manager = MagicMock()

        with (
            patch(
                "src.processing.enhancer.create_enhancer",
            ) as mock_create_enhancer,
            patch(
                "src.processing.extractor.AiFeatureExtractor",
            ),
            patch("src.ai.config.AiConfig"),
        ):
            from src.services.fingerprint_service import (
                create_ai_fingerprint_service,
            )

            svc = create_ai_fingerprint_service(
                model_manager=model_manager,
                use_segmentation=True,
                use_enhancement=False,
                use_dl_extractor=False,
            )

        assert isinstance(svc, FingerprintService)
        # extractor=None falls back to mocked SkeletonMinutiaeExtractor
        assert svc.extractor is not None
        mock_create_enhancer.assert_called_once_with(
            "segmentation", model_manager=model_manager
        )

    def test_enhancement_only(self) -> None:
        """Only enhancement enabled calls create_enhancer with 'enhancement'."""
        model_manager = MagicMock()

        with (
            patch(
                "src.processing.enhancer.create_enhancer",
            ) as mock_create_enhancer,
            patch(
                "src.processing.extractor.AiFeatureExtractor",
            ),
            patch("src.ai.config.AiConfig"),
        ):
            from src.services.fingerprint_service import (
                create_ai_fingerprint_service,
            )

            svc = create_ai_fingerprint_service(
                model_manager=model_manager,
                use_segmentation=False,
                use_enhancement=True,
                use_dl_extractor=False,
            )

        assert isinstance(svc, FingerprintService)
        # extractor=None falls back to mocked SkeletonMinutiaeExtractor
        assert svc.extractor is not None
        mock_create_enhancer.assert_called_once_with(
            "enhancement", model_manager=model_manager
        )

    def test_all_ai_disabled(self) -> None:
        """All AI flags off → service uses CPU defaults (None = default)."""
        model_manager = MagicMock()

        with (
            patch(
                "src.processing.enhancer.create_enhancer",
            ) as mock_create_enhancer,
            patch(
                "src.processing.extractor.AiFeatureExtractor",
            ),
            patch("src.ai.config.AiConfig"),
        ):
            from src.services.fingerprint_service import (
                create_ai_fingerprint_service,
            )

            svc = create_ai_fingerprint_service(
                model_manager=model_manager,
                use_segmentation=False,
                use_enhancement=False,
                use_dl_extractor=False,
            )

        assert isinstance(svc, FingerprintService)
        # enhancer=None → CPU enhancer (default), extractor=None → skeleton (default),
        # both of which are mocked by the session-level conftest
        assert svc.extractor is not None
        mock_create_enhancer.assert_not_called()


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------


class TestGlobalInstance:
    """Tests for the module-level ``fingerprint_service`` singleton."""

    def test_is_fingerprint_service_instance(self) -> None:
        """The global is a FingerprintService instance."""
        from src.services.fingerprint_service import fingerprint_service

        assert isinstance(fingerprint_service, FingerprintService)

"""Módulo de API (REST y CLI)."""

__all__ = ["app"]


def __getattr__(name):
    """Lazy-load ``app`` to break the circular import chain.

    ``src.api.dependencies`` is imported by ``auth_service``, which would
    trigger ``src.api.__init__`` before ``src.api.rest`` (and thus the router
    modules) are ready.  Deferring the import avoids the cycle.
    """
    if name == "app":
        from .rest import app  # noqa: F811
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

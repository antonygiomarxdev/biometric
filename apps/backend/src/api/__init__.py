"""API package — REST routers and CLI."""

__all__ = []


def __getattr__(name: str):
    """Lazy-load ``app`` to break the circular import chain.

    ``src.api.dependencies`` is imported by ``auth_service``, which would
    trigger ``src.api.__init__`` before ``main`` (and the router modules)
    are ready.  Deferring the import avoids the cycle.
    """
    if name == "app":
        from src.main import app  # noqa: F811
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

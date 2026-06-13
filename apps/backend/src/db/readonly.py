"""Read-only database engine factory for safe NLP-to-SQL queries.

Creates a SQLAlchemy engine configured to prevent accidental write
operations (INSERT, UPDATE, DELETE) through PostgreSQL read-only
transaction execution options.  Exposes ``get_readonly_engine`` as the
sole public interface — consumers should never import ``create_engine``
directly.
"""

from sqlalchemy import Engine, create_engine

from src.core.config import config


def get_readonly_engine() -> Engine:
    """Create a SQLAlchemy engine with read-only transaction semantics.

    Uses ``AUTOCOMMIT`` isolation level combined with
    ``postgresql_readonly`` execution option to enforce that generated
    SQL queries can never mutate the database.  ``pool_pre_ping`` is
    enabled to detect stale connections before handing them out.

    Returns:
        A SQLAlchemy ``Engine`` configured for read-only access.
    """
    return create_engine(
        config.database_url,
        execution_options={
            "isolation_level": "AUTOCOMMIT",
            "postgresql_readonly": True,
        },
        pool_pre_ping=True,
    )

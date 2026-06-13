"""Alembic migrations environment configuration.

Loads the ORM models' metadata so that ``alembic autogenerate`` can
detect schema changes and produce migration scripts automatically.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Alembic Config object
config = context.config

# Set up Python logging from the ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import the declarative Base so that Alembic can see all model metadata.
# This is the critical link: ``target_metadata = Base.metadata`` tells
# autogenerate which tables/columns to diff against the database.
from src.db.models import Base

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL, without an Engine.  By
    skipping the Engine creation we can issue ``CREATE TABLE`` /
    ``ALTER TABLE`` statements to any number of databases without
    having an active connection (handy for CI / review).
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode — requires a live database.

    Creates an Engine from the config URL and runs the migration
    against the connected database.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

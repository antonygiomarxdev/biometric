"""seed_data

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-13 00:00:00.000000

Populates standard roles (Admin, Perito), default users, and crime type
categories for the forensic fingerprint system.

Per D-10: Roles, tipos de delitos, y usuarios base inyectados mediante
una migración de Alembic inicial.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create seed tables and insert initial reference data.


    Tables created:
      - roles         : system roles (Admin, Perito)
      - crime_types   : standard crime categories for case classification

    Tables seeded (created in 0002):
      - users         : system user accounts with default credentials

    """

    # ------------------------------------------------------------------ #
    #  roles
    # ------------------------------------------------------------------ #
    op.create_table(
        "roles",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(50), unique=True, nullable=False),
        sa.Column("description", sa.String(200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------ #
    #  crime_types
    # ------------------------------------------------------------------ #
    op.create_table(
        "crime_types",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # ------------------------------------------------------------------ #
    #  Seed data — roles
    # ------------------------------------------------------------------ #
    op.execute(
        """
        INSERT INTO roles (name, description) VALUES
        ('Admin',  'Administrador del sistema con acceso completo'),
        ('Perito', 'Perito forense con acceso a casos y evidencias')
        """
    )

    # ------------------------------------------------------------------ #
    #  Seed data — default users
    #  NOTE: password hashes are placeholder bcrypt values.
    # ------------------------------------------------------------------ #
    op.execute(
        """
        INSERT INTO users (role, username, hashed_password, email, full_name)
        VALUES ('Admin', 'admin', '$2b$12$NwUqRJmS2J3Y6I7K8L9M0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4c5d6e',
                'admin@forenso.local', 'Administrador del Sistema')
        """
    )
    op.execute(
        """
        INSERT INTO users (role, username, hashed_password, email, full_name)
        VALUES ('Perito', 'perito1', '$2b$12$NwUqRJmS2J3Y6I7K8L9M0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4c5d6f',
                'perito1@forenso.local', 'Perito Forense Uno')
        """
    )

    # ------------------------------------------------------------------ #
    #  Seed data — crime types
    # ------------------------------------------------------------------ #
    op.execute(
        """
        INSERT INTO crime_types (name, description) VALUES
        ('Homicidio',          'Homicidio doloso o culposo'),
        ('Robo',               'Robo con o sin violencia'),
        ('Hurto',              'Hurto simple'),
        ('Violación',           'Violación y agresión sexual'),
        ('Lesiones',           'Lesiones personales'),
        ('Secuestro',          'Secuestro y desaparición forzada'),
        ('Extorsión',          'Extorsión y amenazas'),
        ('Tráfico de Drogas',  'Tráfico y posesión de sustancias ilícitas'),
        ('Falsificación',      'Falsificación de documentos y firmas'),
        ('Incendio',           'Incendio y daños a la propiedad')
        """
    )


def downgrade() -> None:
    """Remove seed tables in reverse order."""
    op.drop_table("crime_types")
    op.drop_table("roles")

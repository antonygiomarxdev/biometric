"""seed_data

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-13 00:00:00.000000

Populates standard roles (Admin, Perito), default users, and crime type
categories for the forensic fingerprint system.

Per D-10: Roles, tipos de delitos, y usuarios base inyectados mediante
una migración de Alembic inicial.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create seed tables and insert initial reference data.

    Tables created:
      - roles         : system roles (Admin, Perito)
      - users         : system user accounts linked to roles
      - crime_types   : standard crime categories for case classification
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
    #  users
    # ------------------------------------------------------------------ #
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "role_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("roles.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("username", sa.String(100), unique=True, nullable=False, index=True),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("email", sa.String(200), nullable=True),
        sa.Column("full_name", sa.String(200), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_index("idx_users_role_id", "users", ["role_id"])

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
    #  NOTE: password hashes are placeholder bcrypt values.  The auth
    #  module (planned in 01-06) will provide proper password management.
    # ------------------------------------------------------------------ #
    op.execute(
        """
        INSERT INTO users (role_id, username, password_hash, email, full_name)
        SELECT r.id, 'admin',   '$2b$12$NwUqRJmS2J3Y6I7K8L9M0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4c5d6e',
               'admin@forenso.local', 'Administrador del Sistema'
        FROM roles r WHERE r.name = 'Admin'
        """
    )
    op.execute(
        """
        INSERT INTO users (role_id, username, password_hash, email, full_name)
        SELECT r.id, 'perito1', '$2b$12$NwUqRJmS2J3Y6I7K8L9M0O1P2Q3R4S5T6U7V8W9X0Y1Z2a3b4c5d6f',
               'perito1@forenso.local', 'Perito Forense Uno'
        FROM roles r WHERE r.name = 'Perito'
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
    """Remove seed tables and all seed data in reverse order."""
    op.drop_table("crime_types")
    op.drop_table("users")
    op.drop_table("roles")

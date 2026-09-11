"""add importador to quotations

Revision ID: 003
Revises: 002
Create Date: 2026-09-06

Agrega la columna `importador` (String, nullable) a quotations: nombre de la
empresa importadora real de la operación, usada como feature del modelo
(importador_freq) en lugar del valor por defecto fijo.

Se usa `batch_alter_table` por portabilidad, no por SQLite.

CORRECCIÓN DE LA TERCERA AUDITORÍA: el docstring anterior justificaba
`batch_alter_table` por el DROP COLUMN de SQLite < 3.35, pero el motor real de
este proyecto es PostgreSQL (`postgresql+asyncpg`, ver app/config.py y .env).
Sobre PostgreSQL, `batch_alter_table` sin `recreate` emite ALTER TABLE normales:
la migración es reversible y no destructiva, y `downgrade()` con `drop_column`
es correcto. Se conserva la construcción porque los tests unitarios sí usan
SQLite en memoria (ver app/main.py, ENVIRONMENT == "test"), que es la
justificación verdadera.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.add_column(sa.Column("importador", sa.String(200), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.drop_column("importador")

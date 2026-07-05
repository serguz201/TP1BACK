"""make tipo_contenedor nullable in quotations

Revision ID: 002
Revises: 001
Create Date: 2026-06-28

El tipo de contenedor dejó de ser un campo del formulario de cotización:
no es feature del modelo XGBoost. Se conserva la columna como metadato
opcional, por lo que pasa a ser nullable para no bloquear las inserciones.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.alter_column(
            "tipo_contenedor",
            existing_type=sa.String(50),
            nullable=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.alter_column(
            "tipo_contenedor",
            existing_type=sa.String(50),
            nullable=False,
        )

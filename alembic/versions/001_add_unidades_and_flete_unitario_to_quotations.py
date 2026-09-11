"""add unidades and flete_unitario_usd to quotations

Revision ID: 001
Revises:
Create Date: 2026-06-19

Adds two missing columns to the quotations table:
  - unidades (Integer, nullable): quantity of units shipped (input from form)
  - flete_unitario_usd (Float, nullable): unit freight price in USD/kg (computed server-side)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = "000"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("quotations", sa.Column("unidades", sa.Integer(), nullable=True))
    op.add_column("quotations", sa.Column("flete_unitario_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("quotations", "flete_unitario_usd")
    op.drop_column("quotations", "unidades")

"""add periodo to quotations

Revision ID: 006
Revises: 005
Create Date: 2026-10-21

Agrega la columna `periodo` (String(10), nullable) a quotations.

MOTIVO (diagnóstico QA v1.8). El formulario envía el periodo de embarque
(semanal, mensual o anual) y el modelo lo usa para estimar, pero la cotización
solo guardaba `fecha_embarque`, que es una fecha REPRESENTATIVA del periodo
(lunes de la semana ISO, día 1 del mes o 1 de enero). Una cotización Anual 2027
aparecía en el PDF y en el detalle como "Fecha de Embarque: 2027-01-01", como si
fuera un embarque puntual de ese día, aunque el importe es el promedio del año.

Nullable a propósito: las cotizaciones anteriores a esta migración no tienen el
dato, y una cotización sin periodo sigue siendo válida (se cotiza a la fecha de
hoy). La UI y el PDF tratan `null` como "sin periodo": muestran la fecha tal cual.

`batch_alter_table` por portabilidad con los tests que usan SQLite en memoria.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.add_column(sa.Column("periodo", sa.String(10), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.drop_column("periodo")

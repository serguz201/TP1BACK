"""add mape_regimen to quotations

Revision ID: 004
Revises: 003
Create Date: 2026-09-06

Agrega la columna `mape_regimen` (String(20), nullable) a quotations.

MOTIVO (tercera auditoría). La cotización guardaba `mape_modelo`, que vale ~22.2%
si la fecha cae dentro del histórico de mercado y ~27.5% si el mercado estaba
congelado, pero no guardaba cuál de los dos regímenes era. Una cotización
recuperada de la base, o su PDF, mostraba un error esperado sin decir sobre qué
supuesto se calculó — que es justo la distinción que el resto del sistema se
esfuerza en declarar en la respuesta de la API y en la UI.

Nullable a propósito: las cotizaciones anteriores a esta migración no tienen el
dato y no puede inventarse retroactivamente. La UI debe tratar `null` como
"régimen no registrado", no como "histórico".

`batch_alter_table` por portabilidad con los tests que usan SQLite en memoria;
sobre PostgreSQL emite ALTER TABLE normales y el downgrade es no destructivo
salvo por la propia columna que se elimina.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.add_column(sa.Column("mape_regimen", sa.String(20), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("quotations") as batch_op:
        batch_op.drop_column("mape_regimen")

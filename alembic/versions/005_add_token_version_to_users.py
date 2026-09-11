"""add token_version to users

Revision ID: 005
Revises: 004
Create Date: 2026-09-10

Agrega la columna `token_version` (Integer, NOT NULL, default 0) a users.

MOTIVO (H-21, cuarta auditoría). `POST /api/auth/logout` era un no-op: devolvía
"Sesión cerrada exitosamente" y el mismo access token seguía funcionando, y el
refresh token —que vive 7 días— seguía emitiendo tokens nuevos. Se verificó:
tras el logout, `GET /api/quotations` con el token viejo devolvía 200 y
`POST /api/auth/refresh` devolvía 200 con un access token nuevo.

Con `token_version` el servidor gana una revocación real y barata: el número se
firma dentro del JWT y se compara contra el de la fila del usuario en cada
petición. Cerrar sesión lo incrementa, y con ello todos los tokens emitidos
antes dejan de validar de inmediato. Es también el interruptor de emergencia
para una cuenta comprometida.

NOT NULL con server_default='0' para que las filas existentes queden en la
versión 0 sin necesidad de un backfill: los tokens ya emitidos que no llevan el
campo se tratan como versión 0 y siguen siendo válidos hasta el próximo logout,
de modo que la migración no desloguea a nadie.

`batch_alter_table` por portabilidad con SQLite; sobre PostgreSQL emite ALTER
TABLE normales.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(
            sa.Column(
                "token_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_column("token_version")

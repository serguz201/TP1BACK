"""create initial schema

Revision ID: 000
Revises:
Create Date: 2026-06-19

Creates the complete initial database schema with all 6 tables:
  users, password_reset_tokens, audit_log, ports, container_types, quotations.
The quotations table here reflects the state BEFORE migration 001
(i.e. without the 'unidades' and 'flete_unitario_usd' columns).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "000"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
        sa.Column("failed_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("token", sa.String(255), nullable=False),
        sa.Column("used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.UUID(), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity", sa.String(50), nullable=True),
        sa.Column("entity_id", sa.String(100), nullable=True),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audit_log_user_id"), "audit_log", ["user_id"], unique=False)
    op.create_index(op.f("ix_audit_log_created_at"), "audit_log", ["created_at"], unique=False)

    op.create_table(
        "ports",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("freq_encoding", sa.Float(), nullable=False, server_default="0.05"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_index(op.f("ix_ports_code"), "ports", ["code"], unique=True)

    op.create_table(
        "container_types",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("code", sa.String(20), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("volume_cbm", sa.Float(), nullable=False),
        sa.Column("max_weight_kg", sa.Float(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )

    op.create_table(
        "quotations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("code", sa.String(40), nullable=False),
        sa.Column("puerto_origen", sa.String(100), nullable=False),
        sa.Column("tipo_contenedor", sa.String(50), nullable=False),
        sa.Column("peso_kg", sa.Float(), nullable=False),
        sa.Column("volumen_cbm", sa.Float(), nullable=True),
        sa.Column("fecha_embarque", sa.String(10), nullable=True),
        sa.Column("flete_estimado_usd", sa.Float(), nullable=False),
        sa.Column("ic95_min", sa.Float(), nullable=False),
        sa.Column("ic95_max", sa.Float(), nullable=False),
        sa.Column("mape_modelo", sa.Float(), nullable=False),
        sa.Column("tiempo_ms", sa.Integer(), nullable=False),
        sa.Column("shap_contribuciones", sa.JSON(), nullable=True),
        sa.Column("estado", sa.String(30), nullable=False, server_default="Pendiente"),
        sa.Column("costo_real_usd", sa.Float(), nullable=True),
        sa.Column("comentario", sa.Text(), nullable=True),
        sa.Column("usuario_id", sa.UUID(), nullable=True),
        sa.Column("usuario_nombre", sa.String(150), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("code"),
    )
    op.create_index(op.f("ix_quotations_code"), "quotations", ["code"], unique=True)


def downgrade() -> None:
    op.drop_index(op.f("ix_quotations_code"), table_name="quotations")
    op.drop_table("quotations")
    op.drop_table("container_types")
    op.drop_index(op.f("ix_ports_code"), table_name="ports")
    op.drop_table("ports")
    op.drop_index(op.f("ix_audit_log_created_at"), table_name="audit_log")
    op.drop_index(op.f("ix_audit_log_user_id"), table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_table("password_reset_tokens")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")

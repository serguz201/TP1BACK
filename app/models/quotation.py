import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Float, Integer, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Quotation(Base):
    __tablename__ = "quotations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    code: Mapped[str] = mapped_column(String(40), unique=True, nullable=False, index=True)

    # Input del formulario
    puerto_origen: Mapped[str] = mapped_column(String(100), nullable=False)
    tipo_contenedor: Mapped[str] = mapped_column(String(50), nullable=False)
    peso_kg: Mapped[float] = mapped_column(Float, nullable=False)
    unidades: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    volumen_cbm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fecha_embarque: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)

    # Resultado del modelo
    flete_estimado_usd: Mapped[float] = mapped_column(Float, nullable=False)
    flete_unitario_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ic95_min: Mapped[float] = mapped_column(Float, nullable=False)
    ic95_max: Mapped[float] = mapped_column(Float, nullable=False)
    mape_modelo: Mapped[float] = mapped_column(Float, nullable=False)
    tiempo_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    # SHAP top 3 variables (JSON)
    shap_contribuciones: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Estado y seguimiento
    estado: Mapped[str] = mapped_column(String(30), default="Pendiente", nullable=False)
    costo_real_usd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    comentario: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Auditoría
    usuario_id: Mapped[Optional[uuid.UUID]] = mapped_column(nullable=True)
    usuario_nombre: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

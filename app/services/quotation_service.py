"""CRUD de cotizaciones en base de datos."""

import math
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog
from app.models.quotation import Quotation
from app.models.user import User
from app.schemas.quotation import QuotationCreate


def _generate_code() -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = str(uuid.uuid4())[:4].upper()
    return f"JPS-{today}-{suffix}"


async def create_quotation(
    db: AsyncSession,
    data: QuotationCreate,
    usuario_id: Optional[uuid.UUID],
    usuario_nombre: Optional[str],
) -> Quotation:
    shap_list = (
        [c.model_dump() for c in data.shap_contribuciones]
        if data.shap_contribuciones
        else None
    )
    flete_unitario = (
        round(data.flete_estimado_usd / data.peso_kg, 6)
        if data.peso_kg > 0
        else None
    )
    q = Quotation(
        code=_generate_code(),
        puerto_origen=data.puerto_origen,
        tipo_contenedor=data.tipo_contenedor,
        peso_kg=data.peso_kg,
        unidades=data.unidades,
        volumen_cbm=data.volumen_cbm,
        fecha_embarque=data.fecha_embarque,
        flete_estimado_usd=data.flete_estimado_usd,
        flete_unitario_usd=flete_unitario,
        ic95_min=data.ic95_min,
        ic95_max=data.ic95_max,
        mape_modelo=data.mape_modelo,
        tiempo_ms=data.tiempo_ms,
        shap_contribuciones=shap_list,
        comentario=data.comentario,
        usuario_id=usuario_id,
        usuario_nombre=usuario_nombre,
    )
    db.add(q)
    await db.flush()  # obtiene el id antes del commit
    db.add(AuditLog(
        user_id=usuario_id,
        action="cotizacion_creada",
        entity="quotation",
        entity_id=str(q.id),
        details={"code": q.code, "flete_estimado_usd": data.flete_estimado_usd},
    ))
    await db.commit()
    await db.refresh(q)
    return q


async def list_quotations(
    db: AsyncSession,
    current_user: User,
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None,
    origen: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    estado: Optional[str] = None,
    usuario_id: Optional[uuid.UUID] = None,
) -> tuple[list[Quotation], int]:
    stmt = select(Quotation).order_by(desc(Quotation.created_at))

    # Control de acceso: operativo ve solo las suyas; admin/analista ven todas.
    # El filtro proviene siempre del token, nunca de un parámetro del cliente.
    if current_user.role == "operativo":
        stmt = stmt.where(Quotation.usuario_id == current_user.id)
    elif usuario_id:
        # Admin/analista pueden filtrar por operador específico.
        stmt = stmt.where(Quotation.usuario_id == usuario_id)

    if search:
        stmt = stmt.where(Quotation.code.ilike(f"%{search}%"))
    if origen and origen != "All":
        stmt = stmt.where(Quotation.puerto_origen == origen)
    if date_from:
        stmt = stmt.where(Quotation.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        stmt = stmt.where(Quotation.created_at <= datetime.fromisoformat(date_to + "T23:59:59"))
    if estado:
        stmt = stmt.where(Quotation.estado == estado)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    return list(rows), total


async def get_quotation(db: AsyncSession, quotation_id: uuid.UUID) -> Optional[Quotation]:
    result = await db.execute(select(Quotation).where(Quotation.id == quotation_id))
    return result.scalar_one_or_none()


async def update_actual_cost(
    db: AsyncSession,
    quotation_id: uuid.UUID,
    costo_real_usd: float,
    current_user_id: Optional[uuid.UUID] = None,
) -> Optional[Quotation]:
    q = await get_quotation(db, quotation_id)
    if not q:
        return None
    q.costo_real_usd = costo_real_usd
    q.estado = "Cerrada con Costo Real"
    db.add(AuditLog(
        user_id=current_user_id,
        action="costo_real_registrado",
        entity="quotation",
        entity_id=str(quotation_id),
        details={
            "costo_real_usd": costo_real_usd,
            "flete_estimado_usd": q.flete_estimado_usd,
        },
    ))
    await db.commit()
    await db.refresh(q)
    return q

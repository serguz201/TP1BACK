import math
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, require_roles
from app.models.audit_log import AuditLog

router = APIRouter(prefix="/api/audit", tags=["Auditoría"])


class AuditLogItem(BaseModel):
    id: int
    user_id: Optional[uuid.UUID]
    action: str
    entity: Optional[str]
    entity_id: Optional[str]
    details: Optional[dict]
    ip_address: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class AuditLogListResponse(BaseModel):
    items: list[AuditLogItem]
    total: int
    page: int
    page_size: int
    total_pages: int


@router.get("", response_model=AuditLogListResponse)
async def list_audit_log(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    action: Optional[str] = None,
    user_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("admin", "analista")),
):
    stmt = select(AuditLog).order_by(desc(AuditLog.created_at))

    if action:
        stmt = stmt.where(AuditLog.action == action)
    if user_id:
        stmt = stmt.where(AuditLog.user_id == user_id)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    items = [
        AuditLogItem(
            id=r.id,
            user_id=r.user_id,
            action=r.action,
            entity=r.entity,
            entity_id=r.entity_id,
            details=r.details,
            ip_address=r.ip_address,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]

    return AuditLogListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 0,
    )

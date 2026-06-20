import math
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user, get_db, require_roles
from app.models.user import User
from app.schemas.quotation import (
    QuotationActualCost,
    QuotationCreate,
    QuotationListResponse,
    QuotationResponse,
)
from app.services import quotation_service

router = APIRouter(prefix="/api/quotations", tags=["Cotizaciones"])


@router.post("", response_model=QuotationResponse, status_code=201)
async def create_quotation(
    body: QuotationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = await quotation_service.create_quotation(
        db, body, current_user.id, current_user.name
    )
    return q


@router.get("", response_model=QuotationListResponse)
async def list_quotations(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: Optional[str] = None,
    origen: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    estado: Optional[str] = None,
    usuario_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items, total = await quotation_service.list_quotations(
        db, current_user, page, page_size, search, origen, date_from, date_to, estado, usuario_id
    )
    return QuotationListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/{quotation_id}", response_model=QuotationResponse)
async def get_quotation(
    quotation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = await quotation_service.get_quotation(db, quotation_id)
    if not q:
        raise HTTPException(status_code=404, detail="Cotización no encontrada.")
    return q


@router.patch("/{quotation_id}/actual-cost", response_model=QuotationResponse)
async def update_actual_cost(
    quotation_id: uuid.UUID,
    body: QuotationActualCost,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_roles("admin", "analista")),
):
    q = await quotation_service.update_actual_cost(db, quotation_id, body.costo_real_usd, current_user.id)
    if not q:
        raise HTTPException(status_code=404, detail="Cotización no encontrada.")
    return q


@router.get("/{quotation_id}/pdf")
async def download_pdf(
    quotation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from app.services.pdf_service import generate_quotation_pdf

    q = await quotation_service.get_quotation(db, quotation_id)
    if not q:
        raise HTTPException(status_code=404, detail="Cotización no encontrada.")

    if current_user.role == "operativo" and q.usuario_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tiene permiso para descargar esta cotización.")

    pdf_bytes = generate_quotation_pdf(q)
    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=cotizacion_{q.code}.pdf"},
    )

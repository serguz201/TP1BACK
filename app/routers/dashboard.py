from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_db, require_roles
from app.schemas.dashboard import DashboardResponse, PrecisionMetricsResponse
from app.services import dashboard_service

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("", response_model=DashboardResponse)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("analista", "admin")),
):
    data = await dashboard_service.get_dashboard_data(db)
    return data


@router.get("/precision", response_model=PrecisionMetricsResponse)
async def get_precision_metrics(
    db: AsyncSession = Depends(get_db),
    _=Depends(require_roles("analista", "admin")),
):
    data = await dashboard_service.get_precision_metrics(db)
    return data

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.dependencies import get_current_user, get_db
from app.models.container_type import ContainerType
from app.schemas.catalog import ContainerTypeResponse, PortCatalogResponse
from ml.predictor import get_catalog_ports

router = APIRouter(prefix="/api/catalogs", tags=["Catálogos"])


@router.get("/ports", response_model=list[PortCatalogResponse])
async def list_ports(_=Depends(get_current_user)):
    """
    Retorna los puertos curados para el dropdown: asiáticos con ≥50 registros
    en el dataset de entrenamiento (≈99.8% del volumen real).
    Fuente: modelo_meta.json → puertos_dropdown.
    Cada objeto incluye key (MAYÚSCULAS, se envía al backend) y name (Title Case, se muestra).
    """
    return get_catalog_ports()


@router.get("/container-types", response_model=list[ContainerTypeResponse])
async def list_container_types(
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    result = await db.execute(
        select(ContainerType).where(ContainerType.is_active == True).order_by(ContainerType.name)
    )
    return result.scalars().all()


@router.get("/app-config")
async def get_app_config(_=Depends(get_current_user)):
    """Retorna configuración de la aplicación para el frontend."""
    return {"destination_port": settings.DESTINATION_PORT}

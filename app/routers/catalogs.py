from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.schemas.prediction import (
    FECHA_MAX,
    FECHA_MIN,
    HORIZONTE_IC_CALIBRADO,
)
from app.core.dependencies import get_current_user, get_db
from app.models.container_type import ContainerType
from app.schemas.catalog import ContainerTypeResponse, ImportadorCatalogResponse, PortCatalogResponse
from ml.predictor import get_catalog_importadores, get_catalog_ports

router = APIRouter(prefix="/api/catalogs", tags=["Catálogos"])


@router.get("/ports", response_model=list[PortCatalogResponse])
async def list_ports(_=Depends(get_current_user)):
    """
    Retorna los puertos de embarque del dropdown: los de ≥50 registros
    en el tramo de entrenamiento. Se curan con las mismas filas que ajustaron
    los encoders, así que toda opción tiene una entrada real en puerto_freq.
    Fuente: modelo_meta.json → puertos_dropdown.
    Cada objeto incluye key (MAYÚSCULAS, se envía al backend) y name (Title Case, se muestra).
    """
    return get_catalog_ports()


@router.get("/importadores", response_model=list[ImportadorCatalogResponse])
async def list_importadores(_=Depends(get_current_user)):
    """
    Retorna los importadores curados para el dropdown: ≥50 registros en el
    tramo de ENTRENAMIENTO, excluyendo el bucket anonimizado por Ley 29733.

    Cobertura real, medida y guardada en modelo_meta.json → cobertura_catalogos:
    las 56 opciones cubren el 90.5% de las filas del histórico (96.9% si se
    excluyen las anonimizadas, que ninguna empresa puede seleccionar). La cifra
    de "≈98.5% del volumen real" que figuraba aquí no correspondía a ninguna
    medición y se corrigió en la segunda auditoría.

    Fuente: modelo_meta.json → importadores_dropdown.
    """
    return get_catalog_importadores()


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
    """Retorna configuración de la aplicación para el frontend.

    Incluye el rango de fechas cotizable. TERCERA AUDITORÍA: ese rango estaba
    duplicado a mano en `app/schemas/prediction.py` y en `NewQuote.tsx`, y ambos
    hardcodeados. Ahora el schema lo deriva del artifact y el frontend lo lee de
    aquí, de modo que un reentrenamiento con otro periodo de datos no puede
    dejar la UI ofreciendo fechas que el backend rechaza (ni al revés).
    """
    return {
        "destination_port": settings.DESTINATION_PORT,
        "fecha_min": FECHA_MIN.isoformat(),
        "fecha_max": FECHA_MAX.isoformat(),
        "ic95_horizonte_calibrado_meses": HORIZONTE_IC_CALIBRADO,
    }

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services import prediction_service

router = APIRouter(prefix="/api/predictions", tags=["Predicciones"])


@router.post("/estimate", response_model=PredictionResponse)
async def estimate_freight(
    body: PredictionRequest,
    current_user: User = Depends(get_current_user),
):
    try:
        result = await prediction_service.estimate(
            puerto_origen=body.puerto_origen,
            tipo_contenedor=body.tipo_contenedor,
            peso_kg=body.peso_kg,
            unidades=body.unidades,
            volumen_cbm=body.volumen_cbm,
            fecha_embarque=body.fecha_embarque,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="El modelo tardó demasiado. Intente de nuevo.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en el modelo predictivo: {str(exc)}",
        )

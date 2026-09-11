import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.dependencies import get_current_user
from app.models.user import User
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services import prediction_service

logger = logging.getLogger(__name__)

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
            periodo=body.periodo,
            importador=body.importador,
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="El modelo tardó demasiado. Intente de nuevo.",
        )
    except Exception:
        # H-19: antes se devolvia `str(exc)` al cliente, exponiendo claves de
        # diccionario, rutas del servidor y errores internos de joblib/XGBoost
        # —que el frontend ademas pinta tal cual en pantalla—. La traza queda en
        # el log del servidor; el usuario recibe un mensaje accionable.
        logger.exception("Fallo la prediccion.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "El servicio de pronóstico no está disponible en este momento. "
                "Intente de nuevo; si persiste, avise al administrador."
            ),
        )

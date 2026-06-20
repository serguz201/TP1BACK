"""Servicio de predicción: delega en ml/predictor.py y maneja errores."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.config import settings
from ml.predictor import predict

_executor = ThreadPoolExecutor(max_workers=2)


async def estimate(
    puerto_origen: str,
    tipo_contenedor: str,
    peso_kg: float,
    unidades: Optional[int],
    volumen_cbm: Optional[float],
    fecha_embarque: Optional[str],
) -> dict:
    """Ejecuta la predicción ML en un thread pool para no bloquear el event loop."""
    loop = asyncio.get_running_loop()

    result = await asyncio.wait_for(
        loop.run_in_executor(
            _executor,
            _run_predict,
            puerto_origen,
            tipo_contenedor,
            peso_kg,
            unidades,
            volumen_cbm,
            fecha_embarque,
        ),
        timeout=settings.PREDICTION_TIMEOUT_SECONDS,
    )
    return result


def _run_predict(
    puerto_origen: str,
    tipo_contenedor: str,
    peso_kg: float,
    unidades: Optional[int],
    volumen_cbm: Optional[float],
    fecha_embarque: Optional[str],
) -> dict:
    return predict(puerto_origen, tipo_contenedor, peso_kg, unidades, volumen_cbm, fecha_embarque)

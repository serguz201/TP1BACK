"""
Planificador semanal de la ingesta de Aduanet.

Es una tarea asyncio arrancada en el lifespan de FastAPI, no APScheduler ni
Celery: el sistema ya declara despliegue de UN worker (ver ml/market_state.py) y
la unica necesidad es "un barrido a la semana". Meter un planificador externo
anadiria una dependencia y un proceso mas para resolver un `while True`.

DOS PROPIEDADES QUE HAY QUE CONOCER ANTES DE OPERARLO:

  1. NO DISPARA DOS VECES EL MISMO DIA. La fecha del ultimo disparo se persiste
     en ml/datos_aduanet/estado_ingesta.json ANTES de lanzar el barrido, asi que
     reiniciar el servidor un domingo por la tarde no relanza el barrido de ese
     domingo. Se persiste antes y no despues a proposito: si el barrido revienta
     a la mitad, es preferible saltarse la semana a entrar en un bucle de
     reintentos contra un servicio publico. Queda el boton manual.

  2. NO RECUPERA SEMANAS PERDIDAS. Si el servidor estuvo caido todo el domingo,
     el lunes NO barre. La ventana por defecto es de 120 dias, asi que la
     siguiente ejecucion recupera igualmente los datos; perder un disparo no
     pierde informacion.

Con varios workers, cada uno arrancaria su propio planificador y todos barrerian
a la vez. Para ese caso se apaga con INGESTA_SCHEDULER_ENABLED=false en todos
menos uno.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

from ml import ingesta_config, ingesta_mercado

logger = logging.getLogger(__name__)

# Cada cuanto se comprueba si toca barrer. 15 minutos es suficiente para una
# tarea semanal y mantiene el hilo ocioso practicamente todo el tiempo.
INTERVALO_CHEQUEO_S = 900


def _toca_ejecutar(ahora: datetime, prog: dict, ultima: str | None) -> bool:
    if not prog.get("activa", False):
        return False
    if ahora.weekday() != int(prog.get("dia_semana", 6)):
        return False
    if ahora.hour < int(prog.get("hora", 3)):
        return False
    return ultima != ahora.date().isoformat()


async def bucle_planificador() -> None:
    logger.info("Planificador de ingesta Aduanet iniciado.")
    while True:
        try:
            cfg = ingesta_config.cargar()
            estado = ingesta_mercado.cargar_estado()
            ahora = datetime.now()
            if _toca_ejecutar(ahora, cfg["programacion"], estado.get("ultima_ejecucion_programada")):
                hoy: date = ahora.date()
                # Se marca ANTES de barrer: ver propiedad 1 del docstring.
                ingesta_mercado.marcar_ejecucion_programada(hoy)
                aplicar = bool(cfg["programacion"].get("aplicar_automaticamente", True))
                logger.info("Ingesta programada de Aduanet: iniciando (aplicar=%s).", aplicar)
                resultado = await asyncio.to_thread(
                    ingesta_mercado.ejecutar_ingesta, None, None, aplicar
                )
                logger.info(
                    "Ingesta programada terminada: %d declaraciones nuevas, "
                    "%d errores, aplicado=%s (%s).",
                    resultado["declaraciones_nuevas"], resultado["n_errores"],
                    resultado["aplicado"], resultado["motivo_no_aplicado"] or "sin incidencias",
                )
        except asyncio.CancelledError:
            logger.info("Planificador de ingesta Aduanet detenido.")
            raise
        except Exception:
            # Un fallo aqui no puede tumbar el bucle: la semana siguiente debe
            # volver a intentarlo.
            logger.exception("Fallo en el planificador de ingesta Aduanet.")

        await asyncio.sleep(INTERVALO_CHEQUEO_S)

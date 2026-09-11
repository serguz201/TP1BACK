"""
Estado de las tasas de mercado de referencia (USD/kg) que alimentan
`mercado_lag1/2/3` y `mercado_ma3`.

Estas tres variables concentran ~89% del gain del modelo, asi que su vigencia
determina la calidad real de las predicciones. El modulo cumple dos funciones:

  1. Es la fuente unica de los rezagos vigentes. Se inicializa desde la serie
     mensual real del artifact (`modelo_meta.json` -> `serie_mercado`), que es
     lo ultimo que el modelo observo al entrenarse.
  2. Permite actualizarlos en caliente, sin reentrenar ni reiniciar, via
     PATCH /api/maintenance/market-rates cuando se cierra un mes nuevo.

LIMITACION CONOCIDA: EL ESTADO ES POR PROCESO. `_state` es una variable de
modulo, asi que vive en la memoria del worker que atendio la peticion. Dos
consecuencias que hay que conocer antes de operar el sistema:

  1. Tras reentrenar, un proceso ya levantado sigue sirviendo los rezagos
     viejos hasta que se reinicie (o hasta que un admin llame a
     POST /api/maintenance/market-rates/reset, que si relee el artifact).
  2. Con varios workers de uvicorn/gunicorn, un PATCH solo alcanza al worker
     que lo atendio. Los demas siguen con el estado anterior y las cotizaciones
     se vuelven no deterministas segun a quien caiga la peticion.

Mientras el despliegue sea de un unico worker esto no se manifiesta. Para
despliegue multi-worker hay que persistir el estado (tabla, Redis o fichero) en
vez de mantenerlo en memoria. Se declara aqui para que la limitacion no se
descubra en produccion.

NOTA DE AUDITORIA. Hasta la revision de 2026-09, este modulo existia pero
`predictor.py` no lo importaba: leia siempre los ultimos tres meses del
artifact y el endpoint de actualizacion era un no-op. La documentacion afirmaba
lo contrario. Ahora `predictor.build_features()` consulta este estado de
verdad, y solo para fechas posteriores al ultimo mes historico observado — para
fechas dentro del historico usa el mes realmente anterior a la fecha de
embarque, igual que en entrenamiento.
"""

import json
import threading
from pathlib import Path

META_PATH = Path(__file__).parent / "modelo_meta.json"

_lock = threading.Lock()

# Limites de plausibilidad para los rezagos, en USD/kg. La serie real 2021-2025
# se mueve entre ~0.12 y ~1.16; el rango aceptado es deliberadamente amplio para
# no bloquear un regimen de mercado nuevo, pero rechaza valores que solo pueden
# venir de un error de carga (un flete unitario de 500 USD/kg no existe).
LAG_MIN, LAG_MAX = 0.001, 10.0

# Cuantos meses hacia adelante del ultimo mes historico se acepta declarar como
# "vigente". Mas alla de eso no es una actualizacion de mercado, es un error.
MAX_MESES_ADELANTE = 24


def _ultimo_mes_artifact() -> str:
    """Ultimo mes de la serie real del artifact. Referencia para validar."""
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return max(meta["serie_mercado"])


def _mes_a_indice(mes: str) -> int:
    """'YYYY-MM' -> indice absoluto de mes. ValueError si no es un mes real."""
    anio, m = int(mes[:4]), int(mes[5:7])
    if not 1 <= m <= 12:
        raise ValueError(
            f"'{mes}' no es un mes valido: el componente de mes debe estar entre "
            "01 y 12. El patron del schema solo comprobaba la FORMA, asi que "
            "'9999-99' lo pasaba y hacia que el calculo de meses extrapolados "
            "diera 0 para cualquier fecha, silenciando todas las advertencias."
        )
    return anio * 12 + m


def validar_actualizacion(lag1: float, lag2: float, lag3: float, vigente_hasta: str) -> None:
    """Valida CONTENIDO, no solo forma. Lanza ValueError con un mensaje accionable.

    NOTA DE AUDITORIA (segunda revision). Hasta esta version no habia ninguna
    validacion de contenido. Un PATCH con `vigente_hasta = "9999-99"` era
    aceptado, multiplicaba la prediccion por ~2.8 y apagaba por completo el
    sistema de advertencias de extrapolacion.
    """
    for nombre, v in (("lag1", lag1), ("lag2", lag2), ("lag3", lag3)):
        if not LAG_MIN <= float(v) <= LAG_MAX:
            raise ValueError(
                f"{nombre}={v} esta fuera del rango plausible "
                f"[{LAG_MIN}, {LAG_MAX}] USD/kg. La serie historica real se mueve "
                "entre 0.12 y 1.16 USD/kg; revise las unidades del valor cargado."
            )
    idx = _mes_a_indice(vigente_hasta)
    ultimo = _ultimo_mes_artifact()
    idx_ultimo = _mes_a_indice(ultimo)
    if idx < idx_ultimo:
        raise ValueError(
            f"vigente_hasta='{vigente_hasta}' es anterior al ultimo mes ya "
            f"observado en el entrenamiento ({ultimo}). Retroceder la vigencia "
            "haria que cotizaciones dentro del historico se marcaran como "
            "extrapoladas. Use POST /api/maintenance/market-rates/reset para "
            "volver a la serie real."
        )
    if idx - idx_ultimo > MAX_MESES_ADELANTE:
        raise ValueError(
            f"vigente_hasta='{vigente_hasta}' esta {idx - idx_ultimo} meses por "
            f"delante del ultimo mes observado ({ultimo}), mas de los "
            f"{MAX_MESES_ADELANTE} aceptados. Si la fecha es correcta, el modelo "
            "necesita reentrenarse, no un override de mercado."
        )


def _cargar_desde_artifact() -> dict:
    """Rezagos iniciales: los ultimos 3 meses reales de la serie del artifact."""
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    serie = meta["serie_mercado"]
    meses = sorted(serie)  # 'YYYY-MM' -> lexicografico = cronologico
    ultimos = meses[-3:]
    return {
        "lag1": float(serie[ultimos[-1]]),   # mes mas reciente
        "lag2": float(serie[ultimos[-2]]),
        "lag3": float(serie[ultimos[-3]]),
        "vigente_hasta": ultimos[-1],
        "origen": "artifact",
    }


_state: dict = _cargar_desde_artifact()


def get_market_rates() -> dict:
    """Rezagos vigentes + hasta que mes son validos y de donde salieron."""
    with _lock:
        return dict(_state)


ORIGENES = {"artifact", "manual", "ingesta_aduanet"}


def set_market_rates(lag1: float, lag2: float, lag3: float, vigente_hasta: str,
                     origen: str = "manual") -> dict:
    """Actualiza en caliente los rezagos y el mes hasta el que son vigentes.

    `vigente_hasta` ('YYYY-MM') es el mes al que corresponde `lag1`. Sin el, el
    sistema no puede saber cuantos meses esta extrapolando una cotizacion.

    `origen` distingue quien escribio el valor: 'manual' (un administrador lo
    tecleo en /mantenimiento) o 'ingesta_aduanet' (lo calculo el barrido
    automatico de ml/ingesta_mercado.py). Sin esa distincion, la pantalla de
    mantenimiento no puede decir si el numero que muestra lo puso una persona o
    el robot, que es justo lo que hay que saber para auditar una cotizacion.
    """
    if origen not in ORIGENES:
        raise ValueError(f"origen '{origen}' desconocido; use uno de {sorted(ORIGENES)}.")
    validar_actualizacion(lag1, lag2, lag3, vigente_hasta)
    with _lock:
        _state.update({
            "lag1": float(lag1),
            "lag2": float(lag2),
            "lag3": float(lag3),
            "vigente_hasta": vigente_hasta,
            "origen": origen,
        })
        return dict(_state)


def reset_market_rates() -> dict:
    """Vuelve a los ultimos meses reales del artifact (deshace un override)."""
    global _state
    with _lock:
        _state = _cargar_desde_artifact()
        return dict(_state)

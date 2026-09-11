"""
Configuracion persistente de la ingesta automatica desde Aduanet.

Vive en un JSON al lado del artifact (`ml/ingesta_config.json`) y no en
`app/config.py` porque son datos que un administrador edita desde la pantalla
de Mantenimiento en caliente —el padron de importadores cambia cuando JPS
empieza o deja de seguir a un cliente—, no variables de despliegue.

El padron es necesario: Aduanet NO permite consultar una subpartida sin RUC
(probado con `cagente` vacio, `0` y `118`: cero filas en los tres casos), asi
que la unica forma de barrer el mercado es iterar sobre los importadores
conocidos. El padron inicial lo genera `scripts/generar_padron_importadores.py`
a partir del historico real (166 RUCs, 100% del peso neto importado del
alcance del modelo).
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "ingesta_config.json"

_lock = threading.Lock()

# Subpartida del alcance del modelo. El historico de entrenamiento contiene
# EXCLUSIVAMENTE 4011101000 (neumaticos radiales de automovil), y la
# calibracion del estimador de flete se midio sobre esa subpartida: anadir
# otras partidas de la 4011 cambiaria el nivel de la serie respecto al que el
# modelo vio en entrenamiento. Se deja configurable, pero cambiarlo obliga a
# recalibrar (ver ml/ingesta_mercado.FACTOR_FLETE).
PARTIDAS_POR_DEFECTO = ["4011101000"]

# Aduana Maritima del Callao. El modelo solo cubre ese destino.
ADUANA_POR_DEFECTO = "118"

DEFAULTS: dict = {
    "partidas": PARTIDAS_POR_DEFECTO,
    "aduana": ADUANA_POR_DEFECTO,
    # Dias hacia atras que barre cada ejecucion. 120 dias garantizan 3 meses
    # CERRADOS completos aun ejecutando el dia 1 de un mes, que es lo que
    # necesitan mercado_lag1/2/3. Tambien recoge declaraciones que SUNAT
    # publica con retraso dentro de esa ventana.
    "ventana_dias": 120,
    # Pausa entre peticiones. Aduanet responde en ~2 s; 1 s de cortesia entre
    # llamadas mantiene el barrido completo por debajo de 10 minutos sin
    # castigar un servicio publico.
    "pausa_segundos": 1.0,
    # Un mes solo entra en la serie si acumula al menos estas declaraciones.
    # Evita que un mes a medio publicar (o una ejecucion que fallo a la mitad)
    # se tome por un mes cerrado. El historico real promedia ~109
    # declaraciones/mes en el alcance del modelo; 50 es la mitad de eso.
    "min_declaraciones_mes": 50,
    "programacion": {
        "activa": True,
        # 6 = domingo (lunes=0, como datetime.weekday()).
        "dia_semana": 6,
        "hora": 3,
        # Aplica automaticamente los rezagos calculados a market_state.
        # En False la ejecucion programada solo acumula datos y deja la
        # decision de aplicar a un administrador.
        "aplicar_automaticamente": True,
    },
    "importadores": [],
}

_RE_RUC = re.compile(r"^\d{11}$")


def _fusionar(base: dict, guardado: dict) -> dict:
    """Defaults + fichero, para que una clave nueva no rompa un config viejo."""
    out = dict(base)
    for k, v in guardado.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def cargar() -> dict:
    with _lock:
        if not CONFIG_PATH.exists():
            return json.loads(json.dumps(DEFAULTS))
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return _fusionar(DEFAULTS, json.load(f))


def guardar(config: dict) -> dict:
    """Escritura atomica: fichero temporal + replace, para que una caida a
    mitad de escritura no deje el padron truncado."""
    with _lock:
        tmp = CONFIG_PATH.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        tmp.replace(CONFIG_PATH)
        return config


def importadores_activos() -> list[dict]:
    return [i for i in cargar().get("importadores", []) if i.get("activo", True)]


def validar_ruc(ruc: str) -> str:
    ruc = str(ruc).strip()
    if not _RE_RUC.match(ruc):
        raise ValueError(f"RUC invalido: '{ruc}'. Debe tener exactamente 11 digitos.")
    return ruc


def agregar_importador(ruc: str, nombre: str) -> dict:
    ruc = validar_ruc(ruc)
    cfg = cargar()
    for imp in cfg["importadores"]:
        if imp["ruc"] == ruc:
            imp["nombre"] = nombre or imp.get("nombre", "")
            imp["activo"] = True
            return guardar(cfg)
    cfg["importadores"].append({"ruc": ruc, "nombre": nombre or ruc, "activo": True})
    cfg["importadores"].sort(key=lambda i: i.get("nombre", ""))
    return guardar(cfg)


def set_importador_activo(ruc: str, activo: bool) -> dict:
    ruc = validar_ruc(ruc)
    cfg = cargar()
    for imp in cfg["importadores"]:
        if imp["ruc"] == ruc:
            imp["activo"] = bool(activo)
            return guardar(cfg)
    raise ValueError(f"El RUC {ruc} no esta en el padron.")


def eliminar_importador(ruc: str) -> dict:
    ruc = validar_ruc(ruc)
    cfg = cargar()
    antes = len(cfg["importadores"])
    cfg["importadores"] = [i for i in cfg["importadores"] if i["ruc"] != ruc]
    if len(cfg["importadores"]) == antes:
        raise ValueError(f"El RUC {ruc} no esta en el padron.")
    return guardar(cfg)


def actualizar_programacion(
    activa: bool | None = None,
    dia_semana: int | None = None,
    hora: int | None = None,
    aplicar_automaticamente: bool | None = None,
) -> dict:
    cfg = cargar()
    prog = cfg["programacion"]
    if activa is not None:
        prog["activa"] = bool(activa)
    if dia_semana is not None:
        if not 0 <= int(dia_semana) <= 6:
            raise ValueError("dia_semana debe estar entre 0 (lunes) y 6 (domingo).")
        prog["dia_semana"] = int(dia_semana)
    if hora is not None:
        if not 0 <= int(hora) <= 23:
            raise ValueError("hora debe estar entre 0 y 23.")
        prog["hora"] = int(hora)
    if aplicar_automaticamente is not None:
        prog["aplicar_automaticamente"] = bool(aplicar_automaticamente)
    return guardar(cfg)

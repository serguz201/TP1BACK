"""
Ingesta automatica de la serie de mercado desde Aduanet.

Cierra el ultimo agujero operativo del sistema: hasta ahora `mercado_lag1/2/3`
—que concentran ~89% del gain del modelo— solo se podian actualizar tecleando
tres numeros a mano en /mantenimiento. Este modulo los obtiene del origen
oficial (SUNAT/Aduanet) y los aplica solo.

FLUJO DE UNA EJECUCION

  1. Barre el padron (`ml/ingesta_config.json`) consultando, por importador y
     subpartida, las declaraciones de los ultimos `ventana_dias` dias.
  2. Filtra al alcance del modelo: Aduana Maritima del Callao, peso neto > 0 y
     CIF > FOB.
  3. Acumula las filas en `ml/datos_aduanet/observaciones_aduanet.csv`,
     deduplicando por DUA + subpartida. El fichero es acumulativo: una segunda
     ejecucion sobre la misma ventana no duplica nada, y una declaracion que
     SUNAT publique con retraso entra en cuanto aparezca.
  4. Recalcula la serie mensual y, si hay tres meses CERRADOS y consecutivos con
     datos suficientes, escribe mercado_lag1/2/3 via ml.market_state.

COMO SE ESTIMA EL FLETE (y cuanto se equivoca)

La consulta publica de Aduanet no expone FLE_DOLAR —el detalle por serie que si
lo tiene esta detras de un CAPTCHA— pero si FOB y CIF. Por identidad aduanera:

    CIF = FOB + FLETE + SEGURO   =>   FLETE = (CIF - FOB) - SEGURO

Se verifico la identidad sobre las 86.977 filas del historico: residuo maximo
1.8e-11 USD. El seguro no es publico y NO es un porcentaje fijo del FOB (mediana
0.196%, p95 1.50%), asi que se absorbe en un factor multiplicativo unico:

    FLETE_estimado = (CIF - FOB) * FACTOR_FLETE

FACTOR_FLETE se calibro por busqueda en rejilla [0.90, 1.20] comparando la serie
mensual reconstruida a nivel DECLARACION contra la serie real del artifact
(`modelo_meta.json -> serie_mercado`, 60 meses, 2021-01 a 2025-12):

    factor 0.96 -> MAPE 6.31%   sesgo -4.22%
    factor 1.00 -> MAPE 4.90%   sesgo -0.23%   <-- elegido
    factor 1.01 -> MAPE 4.83%   sesgo +0.77%   (optimo, a 0.07 pp)
    factor 1.04 -> MAPE 5.47%   sesgo +3.77%

Se elige 1.00 y no el optimo numerico 1.01 porque 1.00 es el unico valor con
sentido fisico (el flete no puede superar a flete+seguro), su sesgo es
practicamente nulo y la diferencia de MAPE es ruido. Repitiendo la calibracion
sin los importadores anonimizados por la Ley 29733 —que es lo que la ingesta
puede ver realmente, ver mas abajo— 1.00 da MAPE 4.97% y sesgo -0.99%.

Que el sesgo sea casi cero no es casualidad: el seguro infla la estimacion, pero
agregar por declaracion (en vez de por serie, como hace el entrenamiento)
la desinfla en una magnitud parecida. Los dos errores se cancelan. Si alguna vez
se cambia la subpartida o el nivel de agregacion, HAY QUE RECALIBRAR: la
cancelacion no sobrevive a ese cambio.

LO QUE ESTA INGESTA NO HACE

No sustituye a un reentrenamiento. Aduanet no publica el puerto de embarque sin
CAPTCHA, asi que `puerto_freq`, `importador_freq` y `ruta_directa_por_puerto`
siguen exigiendo el CSV completo + ml/train_model.py + ml/train_quantile_models.py
(en ese orden) y despues POST /api/maintenance/model/reload.

Tampoco alcanza a los importadores anonimizados por la Ley 29733 (~6.8% de las
filas historicas): sin RUC no hay consulta. El padron cubre el 95.7% del peso
neto importado del alcance del modelo.

DESPLIEGUE EN CONTENEDOR. `ml/datos_aduanet/` vive en el sistema de ficheros del
contenedor, asi que un redespliegue lo borra. NO es perdida de informacion: la
ventana por defecto (120 dias) vuelve a descargar los tres meses que necesitan
los rezagos en el primer barrido. Lo que si se pierde es el historial de
ejecuciones; montar un volumen en ese directorio lo conserva.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ml import ingesta_config
from ml.aduanet_scraper import AduanetError, Declaracion, consultar
from ml.market_state import get_market_rates, set_market_rates

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "datos_aduanet"
OBS_CSV = DATA_DIR / "observaciones_aduanet.csv"
ESTADO_JSON = DATA_DIR / "estado_ingesta.json"

# Ver el bloque "COMO SE ESTIMA EL FLETE" del docstring del modulo.
FACTOR_FLETE = 1.00

# Recorte de outliers del flete unitario, en los mismos percentiles que usa
# ml/data_pipeline.py para entrenar. Se aplica sobre el acumulado, no sobre la
# ventana de la ejecucion, para que el umbral no dependa de cuando se corra.
P_LOW, P_HIGH = 0.005, 0.995

# Banda de imposibilidad fisica, aplicada ANTES de los percentiles: un flete
# unitario fuera de aqui solo puede venir de un dato corrupto y no debe influir
# ni siquiera en el calculo de los percentiles.
FLETE_UNIT_MIN, FLETE_UNIT_MAX = 0.001, 10.0

COLUMNAS_OBS = [
    "clave", "dua", "aduana", "anio", "correlativo", "fecha", "ruc", "agente",
    "partida", "canal", "n_series", "fob", "cif", "peso_neto", "peso_bruto",
]

MAX_HISTORIAL = 20

_ejecucion_lock = threading.Lock()
_progreso: dict = {"en_curso": False}


class IngestaEnCursoError(RuntimeError):
    """Ya hay una ingesta corriendo en este proceso."""


# ──────────────────────────────────────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────────────────────────────────────

def _asegurar_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def cargar_observaciones() -> pd.DataFrame:
    if not OBS_CSV.exists():
        return pd.DataFrame(columns=COLUMNAS_OBS)
    df = pd.read_csv(OBS_CSV, dtype={"ruc": str, "partida": str, "aduana": str,
                                     "correlativo": str, "agente": str})
    for col in COLUMNAS_OBS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[COLUMNAS_OBS]


def _guardar_observaciones(df: pd.DataFrame) -> None:
    _asegurar_dir()
    tmp = OBS_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(OBS_CSV)


def cargar_estado() -> dict:
    if not ESTADO_JSON.exists():
        return {"ultima_ejecucion": None, "ultimo_resultado": None,
                "historial": [], "ultima_ejecucion_programada": None}
    with open(ESTADO_JSON, encoding="utf-8") as f:
        return json.load(f)


def _guardar_estado(estado: dict) -> None:
    _asegurar_dir()
    tmp = ESTADO_JSON.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(estado, f, ensure_ascii=False, indent=2)
    tmp.replace(ESTADO_JSON)


def marcar_ejecucion_programada(dia: date) -> None:
    """Deja constancia de que el planificador ya disparo la ejecucion de `dia`.

    Se persiste para que reiniciar el servidor un domingo por la tarde no vuelva
    a lanzar el barrido de ese mismo domingo.
    """
    estado = cargar_estado()
    estado["ultima_ejecucion_programada"] = dia.isoformat()
    _guardar_estado(estado)


# ──────────────────────────────────────────────────────────────────────────────
# Serie mensual y rezagos
# ──────────────────────────────────────────────────────────────────────────────

def _con_flete_unitario(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for col in ("fob", "cif", "peso_neto"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["fecha"] = pd.to_datetime(d["fecha"], errors="coerce")
    d = d.dropna(subset=["fob", "cif", "peso_neto", "fecha"])
    d = d[(d["peso_neto"] > 0) & (d["cif"] > d["fob"])]
    d["flete_unit"] = (d["cif"] - d["fob"]) * FACTOR_FLETE / d["peso_neto"]
    d = d[d["flete_unit"].between(FLETE_UNIT_MIN, FLETE_UNIT_MAX)]
    if len(d) >= 100:
        lo, hi = d["flete_unit"].quantile([P_LOW, P_HIGH])
        d = d[d["flete_unit"].between(lo, hi)]
    d["mes"] = d["fecha"].dt.strftime("%Y-%m")
    return d


def serie_mensual(df: pd.DataFrame) -> pd.DataFrame:
    """Serie mensual observada: mes -> flete unitario medio y nº declaraciones."""
    d = _con_flete_unitario(df)
    if d.empty:
        return pd.DataFrame(columns=["mes", "flete_unit", "declaraciones"])
    g = (d.groupby("mes")
           .agg(flete_unit=("flete_unit", "mean"), declaraciones=("clave", "nunique"))
           .reset_index()
           .sort_values("mes"))
    return g


def _mes_anterior(mes: str) -> str:
    a, m = int(mes[:4]), int(mes[5:7])
    return f"{a - 1:04d}-12" if m == 1 else f"{a:04d}-{m - 1:02d}"


def calcular_rezagos(serie: pd.DataFrame, min_declaraciones: int,
                     hoy: Optional[date] = None) -> dict:
    """Deriva lag1/lag2/lag3 de la serie observada.

    Reglas, todas necesarias para que el resultado sea comparable con lo que el
    modelo vio en entrenamiento:

      · Solo meses CERRADOS. El mes en curso siempre esta a medias y arrastraria
        la media hacia las declaraciones de los primeros dias.
      · Solo meses con al menos `min_declaraciones` declaraciones. Un mes que
        SUNAT todavia esta publicando parece cerrado en el calendario pero no
        en los datos.
      · Los tres meses deben ser CONSECUTIVOS y terminar en el mas reciente que
        cumpla lo anterior. lag1/2/3 son rezagos, no "los tres ultimos meses con
        datos": un hueco intermedio los desalinearia.
    """
    hoy = hoy or date.today()
    mes_actual = hoy.strftime("%Y-%m")

    validos = serie[
        (serie["mes"] < mes_actual)
        & (serie["declaraciones"] >= min_declaraciones)
    ]
    if validos.empty:
        return {"ok": False, "motivo":
                "Ningun mes cerrado alcanza el minimo de declaraciones "
                f"({min_declaraciones}). Amplie la ventana de barrido o revise "
                "el padron de importadores."}

    valores = dict(zip(validos["mes"], validos["flete_unit"]))
    m1 = max(valores)
    m2, m3 = _mes_anterior(m1), _mes_anterior(_mes_anterior(m1))
    faltan = [m for m in (m2, m3) if m not in valores]
    if faltan:
        return {"ok": False, "motivo":
                f"El mes mas reciente con datos suficientes es {m1}, pero "
                f"faltan {', '.join(faltan)}. mercado_lag1/2/3 exige tres meses "
                "consecutivos; amplie la ventana de barrido."}

    return {
        "ok": True,
        "lag1": round(float(valores[m1]), 6),
        "lag2": round(float(valores[m2]), 6),
        "lag3": round(float(valores[m3]), 6),
        "vigente_hasta": m1,
        "meses": [m1, m2, m3],
        "declaraciones_por_mes": {
            m: int(validos.loc[validos["mes"] == m, "declaraciones"].iloc[0])
            for m in (m1, m2, m3)
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Ejecucion
# ──────────────────────────────────────────────────────────────────────────────

def progreso() -> dict:
    return dict(_progreso)


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ejecutar_ingesta(
    desde: Optional[date] = None,
    hasta: Optional[date] = None,
    aplicar: bool = True,
    on_progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Barre Aduanet, acumula observaciones y (opcionalmente) aplica los rezagos.

    Es SINCRONA y puede tardar varios minutos (166 importadores x ~2 s). El
    router la ejecuta en un hilo aparte. Solo se admite una a la vez por
    proceso: dos barridos concurrentes escribirian el mismo CSV.
    """
    if not _ejecucion_lock.acquire(blocking=False):
        raise IngestaEnCursoError(
            "Ya hay una ingesta en curso. Espere a que termine antes de lanzar otra."
        )
    inicio = time.monotonic()
    try:
        cfg = ingesta_config.cargar()
        hasta = hasta or date.today()
        desde = desde or (hasta - timedelta(days=int(cfg["ventana_dias"])))
        if desde > hasta:
            raise ValueError(f"Rango invalido: {desde} > {hasta}")

        importadores = [i for i in cfg.get("importadores", []) if i.get("activo", True)]
        partidas = cfg["partidas"]
        aduana = str(cfg["aduana"])
        pausa = float(cfg["pausa_segundos"])
        total_consultas = len(importadores) * len(partidas)

        if not importadores:
            raise ValueError(
                "El padron de importadores esta vacio. Aduanet no permite "
                "consultar una subpartida sin RUC, asi que sin padron no hay "
                "nada que barrer. Genere uno con "
                "'python -m scripts.generar_padron_importadores'."
            )

        _progreso.clear()
        _progreso.update({
            "en_curso": True, "inicio": _ahora(),
            "desde": desde.isoformat(), "hasta": hasta.isoformat(),
            "consultas_totales": total_consultas, "consultas_hechas": 0,
            "declaraciones": 0, "importador_actual": None, "errores": 0,
        })

        nuevas: list[Declaracion] = []
        errores: list[dict] = []
        hechas = 0

        for imp in importadores:
            for partida in partidas:
                _progreso["importador_actual"] = imp.get("nombre") or imp["ruc"]
                try:
                    filas = consultar(imp["ruc"], partida, desde, hasta)
                    nuevas.extend(f for f in filas if f.aduana == aduana)
                except (AduanetError, ValueError) as exc:
                    errores.append({"ruc": imp["ruc"],
                                    "nombre": imp.get("nombre"),
                                    "partida": partida,
                                    "error": str(exc)})
                    logger.warning("Ingesta: fallo %s/%s: %s", imp["ruc"], partida, exc)
                hechas += 1
                _progreso.update({"consultas_hechas": hechas,
                                  "declaraciones": len(nuevas),
                                  "errores": len(errores)})
                if on_progress:
                    on_progress(dict(_progreso))
                if pausa and hechas < total_consultas:
                    time.sleep(pausa)

        # Un barrido que fallo en mas de la mitad de las consultas no es una
        # foto del mercado: aplicar sus rezagos seria peor que no actualizar.
        barrido_fiable = not errores or len(errores) < total_consultas / 2

        previas = cargar_observaciones()
        n_previas = len(previas)
        if nuevas:
            df_nuevas = pd.DataFrame([{**d.as_dict(), "clave": d.clave} for d in nuevas])
            df_nuevas = df_nuevas[COLUMNAS_OBS]
            combinado = pd.concat([previas, df_nuevas], ignore_index=True)
            # Se conserva la ULTIMA aparicion: si SUNAT rectifica una DUA, la
            # lectura mas reciente manda sobre la almacenada.
            combinado = combinado.drop_duplicates(subset="clave", keep="last")
            combinado = combinado.sort_values(["fecha", "clave"]).reset_index(drop=True)
            _guardar_observaciones(combinado)
        else:
            combinado = previas

        serie = serie_mensual(combinado)
        rezagos = calcular_rezagos(serie, int(cfg["min_declaraciones_mes"]))

        aplicado = False
        motivo_no_aplicado: Optional[str] = None
        if not aplicar:
            motivo_no_aplicado = "Ejecucion en modo solo-acumular: no se pidio aplicar."
        elif not rezagos["ok"]:
            motivo_no_aplicado = rezagos["motivo"]
        elif not barrido_fiable:
            motivo_no_aplicado = (
                f"{len(errores)} de {total_consultas} consultas fallaron: el "
                "barrido no es representativo y no se aplican los rezagos."
            )
        else:
            try:
                set_market_rates(rezagos["lag1"], rezagos["lag2"],
                                 rezagos["lag3"], rezagos["vigente_hasta"],
                                 origen="ingesta_aduanet")
                aplicado = True
            except ValueError as exc:
                # ml.market_state rechaza rezagos implausibles o una vigencia
                # incoherente con el artifact. Se reporta, no se fuerza.
                motivo_no_aplicado = str(exc)

        resultado = {
            "inicio": _progreso.get("inicio"),
            "fin": _ahora(),
            "duracion_s": round(time.monotonic() - inicio, 1),
            "desde": desde.isoformat(),
            "hasta": hasta.isoformat(),
            "importadores_consultados": len(importadores),
            "partidas": partidas,
            "declaraciones_descargadas": len(nuevas),
            "declaraciones_nuevas": max(0, len(combinado) - n_previas),
            "declaraciones_acumuladas": len(combinado),
            "errores": errores[:20],
            "n_errores": len(errores),
            "serie_mensual": serie.tail(12).to_dict("records"),
            "rezagos": rezagos,
            "aplicado": aplicado,
            "motivo_no_aplicado": motivo_no_aplicado,
            "estado_mercado": get_market_rates(),
        }

        estado = cargar_estado()
        estado["ultima_ejecucion"] = resultado["fin"]
        estado["ultimo_resultado"] = resultado
        estado.setdefault("historial", []).insert(0, {
            k: resultado[k] for k in
            ("fin", "desde", "hasta", "declaraciones_nuevas",
             "declaraciones_acumuladas", "n_errores", "aplicado",
             "motivo_no_aplicado", "duracion_s")
        })
        estado["historial"] = estado["historial"][:MAX_HISTORIAL]
        _guardar_estado(estado)

        return resultado
    finally:
        _progreso.update({"en_curso": False, "importador_actual": None})
        _ejecucion_lock.release()


def estado_completo() -> dict:
    """Todo lo que la pantalla de Mantenimiento necesita mostrar de la ingesta."""
    cfg = ingesta_config.cargar()
    estado = cargar_estado()
    obs = cargar_observaciones()
    serie = serie_mensual(obs)
    rezagos = calcular_rezagos(serie, int(cfg["min_declaraciones_mes"]))
    return {
        "programacion": cfg["programacion"],
        "ventana_dias": cfg["ventana_dias"],
        "partidas": cfg["partidas"],
        "aduana": cfg["aduana"],
        "min_declaraciones_mes": cfg["min_declaraciones_mes"],
        "n_importadores": len(cfg.get("importadores", [])),
        "n_importadores_activos": len([i for i in cfg.get("importadores", [])
                                       if i.get("activo", True)]),
        "declaraciones_acumuladas": len(obs),
        "serie_mensual": serie.tail(12).to_dict("records"),
        "rezagos_calculados": rezagos,
        "ultima_ejecucion": estado.get("ultima_ejecucion"),
        "ultimo_resultado": estado.get("ultimo_resultado"),
        "historial": estado.get("historial", []),
        "progreso": progreso(),
        "estado_mercado": get_market_rates(),
        "factor_flete": FACTOR_FLETE,
    }

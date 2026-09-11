"""
Monitor de deriva: cuando hay que pedir un CSV nuevo y reentrenar.

EL PROBLEMA QUE RESUELVE. El modelo se degrada por dos vias distintas y con
ritmos distintos: la serie de mercado envejece (lo arregla la ingesta de
Aduanet, sola, cada semana) y los codificadores envejecen (`puerto_freq`,
`importador_freq`, `ruta_directa_por_puerto`), que solo se arreglan reentrenando
con el CSV detallado de SUNAT. Sin un indicador, la segunda erosion es
invisible: nadie tiene forma de saber si el modelo lleva tres meses o tres anos
sin ver un puerto nuevo.

LO QUE ESTE MODULO **NO** PUEDE MEDIR, Y POR QUE. Merece decirse antes que lo
que si mide, porque la tentacion de inventar una senal aqui es alta:

  · PUERTOS NUEVOS. La consulta publica de Aduanet no expone el puerto de
    embarque (esta en el detalle por serie, tras CAPTCHA). No hay ninguna
    fuente automatica de la que deducirlo.
  · IMPORTADORES NUEVOS. El scraper solo consulta los RUC del padron, asi que
    por construccion no puede descubrir un importador que no conociera ya.

  Se probo si "% de importadores nuevos" servia de PROXY de "puertos nuevos",
  midiendolo sobre los 48 meses utiles del historico: correlacion de Pearson
  0.142 y de Spearman 0.131. Los meses con MAS importadores nuevos trajeron 0.62
  puertos nuevos de media; los meses con MENOS, 0.58. No hay senal. Se descarta
  explicitamente para que nadie vuelva a proponerlo como atajo.

  La deriva de puertos e importadores solo se observa cuando llega un CSV, y
  entonces la mide `ml/corpus.validar()`, que reporta exactamente que puertos e
  importadores nuevos trae el fichero. Ese es su sitio, no este.

LO QUE SI MIDE. Tres senales reales, todas derivadas del propio artifact para
que no se queden obsoletas si se reentrena con otros datos:

  1. HORIZONTE DE EXTRAPOLACION. Meses entre el ultimo mes de mercado que el
     modelo observo al entrenarse y el mes de mercado vigente. Es la senal mas
     fuerte porque el artifact trae medido su propio coste: MAPE dentro del
     historico frente a MAPE en regimen extrapolado.
  2. ANTIGUEDAD DEL CORPUS. Dias desde la ultima declaracion del corpus de
     entrenamiento. Es el numero que responde literalmente a "¿hace cuanto que
     no le damos datos nuevos?".
  3. POSICION DEL MERCADO EN EL RANGO DE ENTRENAMIENTO. Si el flete unitario
     vigente cae fuera del rango que el modelo vio, la extrapolacion ya no es
     solo temporal: es del espacio de features, y los arboles de XGBoost no
     extrapolan fuera del rango con el que se ajustaron.

El veredicto agrega las tres con los umbrales del propio sistema
(`predictor.MAX_MESES_EXTRAPOLACION`), no con numeros inventados aqui.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from ml import corpus, predictor
from ml.market_state import get_market_rates

logger = logging.getLogger(__name__)

# Antiguedad del corpus, en meses, a partir de la cual conviene pedir datos.
# 6 meses es medio ciclo de la estacionalidad que el modelo codifica en
# mes_sin/mes_cos; 12 es un ciclo completo sin datos nuevos.
CORPUS_MESES_ATENCION = 6
CORPUS_MESES_CRITICO = 12

NIVELES = {"ok": 0, "atencion": 1, "critico": 2}


def _meses_entre(desde: str, hasta: str) -> int:
    """Distancia en meses entre dos 'YYYY-MM'."""
    a = int(desde[:4]) * 12 + int(desde[5:7])
    b = int(hasta[:4]) * 12 + int(hasta[5:7])
    return b - a


def _senal(nombre: str, nivel: str, titulo: str, detalle: str,
           valor=None, **extra) -> dict:
    return {"nombre": nombre, "nivel": nivel, "titulo": titulo,
            "detalle": detalle, "valor": valor, **extra}


def _senal_horizonte(hoy: date) -> dict:
    ultimo_train = predictor._ULTIMO_MES_SERIE
    mercado = get_market_rates()
    vigente = mercado["vigente_hasta"]
    meses = max(0, _meses_entre(ultimo_train, vigente))
    limite = predictor.MAX_MESES_EXTRAPOLACION

    if meses == 0:
        nivel, detalle = "ok", (
            f"El modelo se entreno con datos de mercado hasta {ultimo_train} y "
            "el mercado vigente es de ese mismo mes: no hay extrapolacion."
        )
    elif meses <= 3:
        nivel, detalle = "ok", (
            f"El mercado vigente ({vigente}) va {meses} mes(es) por delante del "
            f"ultimo que el modelo observo al entrenarse ({ultimo_train}). "
            "Dentro de lo normal."
        )
    elif meses <= limite:
        nivel, detalle = "atencion", (
            f"El mercado vigente ({vigente}) va {meses} meses por delante del "
            f"ultimo observado en entrenamiento ({ultimo_train}). Toda cotizacion "
            f"futura se sirve en regimen extrapolado, con un error esperado de "
            f"~{predictor.MAPE_EXTRAPOLADO}% en vez del {predictor.MODEL_MAPE}% "
            "medido dentro del historico."
        )
    else:
        nivel, detalle = "critico", (
            f"El mercado vigente ({vigente}) va {meses} meses por delante del "
            f"ultimo observado en entrenamiento ({ultimo_train}), por encima del "
            f"limite de {limite} meses que el propio sistema declara. Reentrenar "
            "con datos actuales es lo unico que devuelve el modelo a su regimen "
            "medido."
        )
    return _senal(
        "horizonte_mercado", nivel,
        "Distancia entre el mercado actual y el del entrenamiento",
        detalle, valor=meses,
        unidad="meses", ultimo_mes_entrenamiento=ultimo_train,
        mes_vigente=vigente, origen_mercado=mercado["origen"], limite=limite,
    )


def _senal_corpus(hoy: date, est: dict) -> dict:
    ultima = est.get("fecha_max_en_alcance")
    if not ultima:
        return _senal(
            "antiguedad_corpus", "critico", "Antiguedad del corpus",
            "El corpus no tiene ninguna fila en el alcance del modelo.", valor=None,
        )
    f = datetime.strptime(ultima, "%Y-%m-%d").date()
    dias = (hoy - f).days
    meses = dias // 30

    if meses < CORPUS_MESES_ATENCION:
        nivel = "ok"
        detalle = (
            f"La ultima declaracion del corpus es del {ultima} ({meses} meses). "
            "El corpus esta al dia."
        )
    elif meses < CORPUS_MESES_CRITICO:
        nivel = "atencion"
        detalle = (
            f"La ultima declaracion del corpus es del {ultima}: {meses} meses sin "
            "datos nuevos. Los puertos e importadores que hayan aparecido desde "
            "entonces caen al valor por defecto de sus codificadores."
        )
    else:
        nivel = "critico"
        detalle = (
            f"La ultima declaracion del corpus es del {ultima}: {meses} meses, mas "
            f"de un ciclo estacional completo sin datos nuevos. Conviene pedir a "
            "SUNAT el CSV detallado actualizado, incorporarlo y reentrenar."
        )
    return _senal(
        "antiguedad_corpus", nivel, "Antiguedad del corpus de entrenamiento",
        detalle, valor=meses, unidad="meses", ultima_fecha=ultima, dias=dias,
        filas=est.get("filas_en_alcance"), ruta=est.get("ruta"),
    )


def _senal_rango_mercado() -> dict:
    serie = predictor.SERIE_MERCADO
    lo, hi = min(serie.values()), max(serie.values())
    actual = get_market_rates()["lag1"]

    if lo <= actual <= hi:
        pos = (actual - lo) / (hi - lo) if hi > lo else 0.0
        nivel = "ok"
        detalle = (
            f"El flete unitario vigente ({actual:.4f} USD/kg) cae dentro del rango "
            f"que el modelo vio al entrenarse ({lo:.4f}-{hi:.4f} USD/kg), en el "
            f"percentil {pos:.0%} de ese rango."
        )
    else:
        exceso = (actual - hi) / hi if actual > hi else (lo - actual) / lo
        nivel = "atencion" if exceso < 0.25 else "critico"
        extremo = "por encima del maximo" if actual > hi else "por debajo del minimo"
        detalle = (
            f"El flete unitario vigente ({actual:.4f} USD/kg) esta {exceso:.0%} "
            f"{extremo} del rango visto en entrenamiento ({lo:.4f}-{hi:.4f} USD/kg). "
            "Los arboles de XGBoost no extrapolan fuera del rango con el que se "
            "ajustaron: la prediccion se satura en el extremo y el error real es "
            "mayor que el declarado."
        )
    return _senal(
        "rango_mercado", nivel, "El mercado actual frente al rango de entrenamiento",
        detalle, valor=round(actual, 6), unidad="USD/kg",
        rango_entrenamiento=[round(lo, 6), round(hi, 6)],
    )


def diagnosticar(hoy: Optional[date] = None) -> dict:
    """Diagnostico completo de deriva, con veredicto agregado y recomendacion."""
    hoy = hoy or date.today()
    est_corpus = corpus.estado()

    senales = [
        _senal_horizonte(hoy),
        _senal_corpus(hoy, est_corpus),
        _senal_rango_mercado(),
    ]
    peor = max(senales, key=lambda s: NIVELES[s["nivel"]])
    nivel = peor["nivel"]

    if nivel == "ok":
        veredicto = "El modelo esta en su regimen medido. No hace falta reentrenar."
    elif nivel == "atencion":
        veredicto = (
            "El modelo sigue sirviendo, pero fuera de las condiciones en las que "
            "se midio su error. Conviene planificar un reentrenamiento."
        )
    else:
        veredicto = (
            "El modelo esta operando lejos de las condiciones en las que se midio. "
            "Reentrenar con un CSV actualizado es la accion recomendada."
        )

    return {
        "fecha": hoy.isoformat(),
        "nivel": nivel,
        "veredicto": veredicto,
        "senal_dominante": peor["nombre"],
        "senales": senales,
        "no_observable": {
            "titulo": "Lo que este diagnostico no puede ver",
            "detalle": (
                "La aparicion de puertos de embarque e importadores nuevos no es "
                "observable sin el CSV detallado: Aduanet no publica el puerto sin "
                "CAPTCHA, y el barrido solo consulta los RUC que ya conoce. Se "
                "descarto usar '% de importadores nuevos' como proxy de 'puertos "
                "nuevos' tras medirlo sobre 48 meses del historico (correlacion de "
                "Pearson 0.142). Esa deriva se cuantifica al incorporar un CSV, no "
                "antes."
            ),
        },
        "corpus": est_corpus,
    }

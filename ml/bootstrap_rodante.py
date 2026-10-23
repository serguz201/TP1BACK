"""
Bootstrap pareado de los DOS efectos centrales del articulo, medidos sobre las
mismas filas del protocolo de origen rodante, y descomposicion de la ganancia
de reentrenar en recencia y masa de datos.

POR QUE EXISTE ESTE MODULO

`ml/walk_forward.py` publica dos cifras como escalares:

    resumen.ganancia_de_reentrenar_pp  = 3.73
    resumen.coste_del_rezago_sunat_pp  = 7.00

Son las dos cifras que el articulo declara como su aporte, y hasta ahora eran
las unicas comparaciones del trabajo sin intervalo de confianza: la comparacion
de algoritmos y la ablacion si lo tienen (`ml/estudios_paper.py`, B=2,000,
bootstrap pareado). Un revisor que lea que el 86% de los 3.73 pp proviene de un
solo mes de doce pedira el intervalo antes de aceptar la conclusion. Este modulo
lo calcula.

TRES BOOTSTRAP, PORQUE RESPONDEN TRES PREGUNTAS DISTINTAS

  1. `por_filas` — remuestreo de las 20,885 declaraciones evaluadas, pareado:
     el mismo indice se aplica a las dos series comparadas, de modo que la
     diferencia no depende de haber usado muestras distintas. Es exactamente el
     protocolo de las Tablas VII y VIII, y responde: dada esta trayectoria de
     mercado, cuanta incertidumbre hay en la diferencia por el muestreo de
     declaraciones.

  2. `por_mes` — remuestreo de los DOCE PLIEGUES con reemplazo (bootstrap de
     bloques). Trata el mes como la unidad de observacion, que es lo que es: las
     declaraciones de un mismo mes comparten regimen de mercado y no son
     independientes. Responde la pregunta que de verdad hace el revisor: el
     efecto sobrevive a que junio pudiera no haber ocurrido? Es el intervalo
     honesto para una afirmacion sobre el sistema, y es mucho mas ancho que (1).

  3. `por_filas_sin_junio` — el de (1) excluyendo junio de 2025. No sustituye a
     (2): sirve para publicar el efecto del mes extremo como dato y no como
     impresion.

Ninguno de los tres se elige segun el resultado: los tres se reportan.

DESCOMPOSICION RECENCIA / MASA

El pliegue de enero entrena con 2021-2024 y el de diciembre con eso mas once
meses de 2025, asi que la diferencia entre el modelo rodante y el congelado
mezcla dos cosas: que el modelo rodante ve datos MAS RECIENTES y que ve MAS
datos. Para separarlas se anade un tercer regimen por pliegue:

  `ventana_fija`: el mismo pliegue, con el mismo pipeline y los mismos
  encoders, pero ajustado unicamente sobre las N filas mas recientes de su
  bloque, con N = el tamano del bloque del primer pliegue. La masa de datos
  queda constante en los doce meses y solo varia la recencia.

De donde:

    estatico -> ventana_fija   = efecto de la RECENCIA (masa constante)
    ventana_fija -> rodante    = efecto de la MASA acumulada
    estatico -> rodante        = la ganancia total publicada (3.73 pp)

SALIDA

  ml/bootstrap_rodante.json

Reejecutar tras cualquier cambio en el pipeline o en la receta: el modulo lee
`n_estimators` del artifact, igual que `walk_forward.py`, de modo que no puede
describir una configuracion distinta de la desplegada.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from ml import data_pipeline as dp
from ml import walk_forward as wf

CSV_PATH = os.environ.get("JPS_CSV_PATH", "resultado_combinado.csv")
INFORME_PATH = "ml/bootstrap_rodante.json"

# Las mismas constantes que ml/estudios_paper.py, para que los intervalos de
# este modulo y los de las Tablas VII-VIII sean del mismo procedimiento.
N_BOOTSTRAP = 2000
RANDOM_STATE = 42

ANIO = 2025

# Series comparadas. La clave es el nombre en el informe; el valor, la etiqueta
# legible que se usa en los mensajes de consola y en el paper.
SERIES = {
    "rodante": "reentrenamiento mensual",
    "estatico": "modelo congelado (mismas filas)",
    "rezago_sunat": "reentrenado con la estadistica un mes tarde",
    "ventana_fija": "reentrenado con masa de datos constante",
    "persistencia_lag1": "linea base: persistencia de mercado (lag1)",
    "media_movil_ma3": "linea base: media movil de tres meses",
    "mediana_train": "linea base: mediana del bloque de entrenamiento",
}


def _ape(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Error porcentual absoluto por fila, en puntos porcentuales.

    El MAPE es la media de este vector, de modo que remuestrear filas equivale
    a remuestrear este vector: no hace falta recalcular predicciones dentro del
    bootstrap, y el resultado es identico al de `mean_absolute_percentage_error`
    sobre la submuestra.
    """
    return np.abs((y - yhat) / y) * 100.0


def _pliegue_filas(scope: pd.DataFrame, mes: pd.Period, modelo_base,
                   n_ventana: int | None) -> dict | None:
    """Un pliegue del origen rodante devolviendo las PREDICCIONES por fila.

    Replica `walk_forward._pliegue` —misma particion, mismas aserciones, mismo
    ajuste— pero conserva los vectores en vez de resumirlos en metricas, que es
    lo que el bootstrap pareado necesita. Las invariantes se repiten aqui a
    proposito: si este modulo divergiera del protocolo, debe fallar, no dar un
    intervalo de otra cosa.
    """
    inicio = pd.Timestamp(mes.start_time).normalize()
    fin = pd.Timestamp(mes.end_time).normalize() + pd.Timedelta(days=1)

    scope_pliegue = scope[scope["FECHA"] < fin]
    historia = scope_pliegue["FECHA"] < inicio
    if not historia.any():
        return None

    corte_interno = dp._corte_por_fecha(
        scope_pliegue.loc[historia, "FECHA"], dp.FRACCION_TRAIN_INTERNO
    )
    p = dp.particionar(scope_pliegue, corte_interno, inicio, esquema="80_20")
    if len(p.test) < wf.MIN_FILAS_MES:
        return None

    assert p.train["FECHA"].max() < inicio, "train del pliegue toca el mes evaluado"
    assert p.val["FECHA"].max() < inicio, "val del pliegue toca el mes evaluado"
    assert p.test["FECHA"].min() >= inicio and p.test["FECHA"].max() < fin
    assert p.serie_mercado.index.max() <= mes

    bloque = pd.concat([p.train, p.val]).sort_values("FECHA", kind="mergesort")
    modelo = wf._ajustar(bloque)

    X_test = p.test[dp.FEATURES]
    y_test = p.test[dp.TARGET].to_numpy(dtype=float)

    preds = {
        "rodante": modelo.predict(X_test),
        "estatico": modelo_base.predict(X_test),
        "persistencia_lag1": X_test["mercado_lag1"].to_numpy(dtype=float),
        "media_movil_ma3": X_test["mercado_ma3"].to_numpy(dtype=float),
        "mediana_train": np.full(len(y_test), float(p.train[dp.TARGET].median())),
    }

    # Cota pesimista: la estadistica aduanera del mes anterior aun no se
    # publico. Si la serie no llega tan atras, el mes queda fuera de ESTE
    # regimen (y solo de este), igual que en walk_forward.
    periodos = p.test["FECHA"].dt.to_period("M")
    X_tarde = wf._retrasar_lags(X_test, periodos, p.serie_mercado, 1)
    if X_tarde is not None:
        preds["rezago_sunat"] = modelo.predict(X_tarde)

    # Masa constante: las N filas mas recientes del bloque de este pliegue.
    # `n_ventana` lo fija el primer pliegue; en el primero, por tanto, este
    # regimen coincide exactamente con `rodante`.
    if n_ventana is not None:
        modelo_vf = wf._ajustar(bloque.iloc[-n_ventana:])
        preds["ventana_fija"] = modelo_vf.predict(X_test)

    return {
        "mes": str(mes),
        "n_bloque": int(len(bloque)),
        "n_evaluado": int(len(y_test)),
        "y": y_test,
        "preds": {k: np.asarray(v, dtype=float) for k, v in preds.items()},
    }


def recolectar(csv_path: str = CSV_PATH, anio: int = ANIO, verbose: bool = True) -> dict:
    """Los doce pliegues con sus predicciones por fila, listos para remuestrear."""
    scope = dp.cargar_scope(csv_path)
    p_base = dp.construir(csv_path, esquema="80_20")
    bloque_base = pd.concat([p_base.train, p_base.val])

    n_estimators, diag = wf._receta_del_artifact()
    if n_estimators is None:
        raise RuntimeError(
            "No hay receta en el artifact. Ejecutar antes ml/train_model.py: este "
            "modulo no selecciona n_estimators por su cuenta para no poder describir "
            "una configuracion distinta de la desplegada."
        )
    wf._N_ESTIMATORS = n_estimators

    modelo_base = wf._ajustar(bloque_base)
    if verbose:
        print(f"=== BOOTSTRAP PAREADO DEL ORIGEN RODANTE {anio} ===")
        print(f"  n_estimators = {n_estimators} (receta del artifact desplegado)")
        print(f"  bloque del modelo congelado: {len(bloque_base):,} filas")

    pliegues: list[dict] = []
    n_ventana: int | None = None
    for m in range(1, 13):
        mes = pd.Period(f"{anio}-{m:02d}", freq="M")
        f = _pliegue_filas(scope, mes, modelo_base, n_ventana)
        if f is None:
            if verbose:
                print(f"  {mes} omitido (sin datos suficientes)")
            continue
        if n_ventana is None:
            # El primer pliegue define la masa de referencia. Se recalcula su
            # regimen de ventana fija para que la serie este completa en los
            # doce meses; por construccion reproduce `rodante`.
            n_ventana = f["n_bloque"]
            f = _pliegue_filas(scope, mes, modelo_base, n_ventana)
        pliegues.append(f)
        if verbose:
            print(f"  {f['mes']}  bloque {f['n_bloque']:>7,}  evaluado "
                  f"{f['n_evaluado']:>6,}")

    if len(pliegues) != 12:
        print(f"  AVISO: se recolectaron {len(pliegues)} de 12 pliegues.")

    crecimiento = round(
        100 * (pliegues[-1]["n_bloque"] / pliegues[0]["n_bloque"] - 1), 1
    )
    if verbose:
        print(f"  El bloque de ajuste crece {crecimiento:+.1f}% entre el primer "
              "pliegue y el ultimo.")
    return {"pliegues": pliegues, "crecimiento_bloque_%": crecimiento,
            "n_estimators": n_estimators, "seleccion_n_estimators": diag}


def _apilar(pliegues: list[dict], series: list[str]) -> tuple[np.ndarray, dict]:
    """(etiqueta de mes por fila, {serie: vector de APE}) solo con las filas
    en las que TODAS las series pedidas existen.

    Pararse en esto importa: `rezago_sunat` puede faltar en un mes si la serie
    de mercado no llega tan atras, y comparar su agregado contra un `rodante`
    calculado sobre doce meses seria comparar dos conjuntos distintos. Al
    exigir que las series pedidas esten presentes, cada contraste se hace sobre
    sus propias filas comunes y el pareado es real.
    """
    usables = [p for p in pliegues if all(s in p["preds"] for s in series)]
    meses = np.concatenate([np.full(p["n_evaluado"], p["mes"]) for p in usables])
    apes = {
        s: np.concatenate([_ape(p["y"], p["preds"][s]) for p in usables])
        for s in series
    }
    return meses, apes


def _ic(muestras: np.ndarray) -> list[float]:
    return [round(float(np.percentile(muestras, 2.5)), 2),
            round(float(np.percentile(muestras, 97.5)), 2)]


def _contraste(ape_a: np.ndarray, ape_b: np.ndarray, meses: np.ndarray,
               rng: np.random.Generator) -> dict:
    """Contraste pareado de `a` (peor esperado) contra `b` (mejor esperado).

    Delta positivo = `b` tiene menos error que `a`, que es el sentido en que el
    articulo enuncia sus dos efectos (reentrenar REDUCE el MAPE; el rezago lo
    ELEVA). Se devuelven los tres remuestreos descritos en el encabezado del
    modulo. El p-valor es de una cola y conserva la definicion de
    estudios_paper.py: proporcion de remuestreos en los que el efecto no
    aparece (delta <= 0).
    """
    n = len(ape_a)
    delta_punto = float(ape_a.mean() - ape_b.mean())

    # (1) Remuestreo de filas, pareado: un solo indice para las dos series.
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    d_filas = ape_a[idx].mean(axis=1) - ape_b[idx].mean(axis=1)

    # (2) Remuestreo de los doce pliegues con reemplazo. Se agregan ponderando
    # por declaraciones, igual que el informe del walk-forward, de modo que el
    # estadistico remuestreado es el mismo MAPE ponderado que se publica.
    etiquetas = np.unique(meses)
    grupos = {m: np.flatnonzero(meses == m) for m in etiquetas}
    sel = rng.integers(0, len(etiquetas), size=(N_BOOTSTRAP, len(etiquetas)))
    d_mes = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        filas = np.concatenate([grupos[etiquetas[j]] for j in sel[b]])
        d_mes[b] = ape_a[filas].mean() - ape_b[filas].mean()

    return {
        "delta_pp": round(delta_punto, 2),
        "n_filas": int(n),
        "por_filas": {
            "delta_medio_pp": round(float(d_filas.mean()), 2),
            "IC95": _ic(d_filas),
            "p_valor_una_cola": round(float(np.mean(d_filas <= 0)), 4),
            "significativo_95": bool(np.percentile(d_filas, 2.5) > 0),
        },
        "por_mes": {
            "delta_medio_pp": round(float(d_mes.mean()), 2),
            "IC95": _ic(d_mes),
            "p_valor_una_cola": round(float(np.mean(d_mes <= 0)), 4),
            "significativo_95": bool(np.percentile(d_mes, 2.5) > 0),
        },
    }


def _sin_mes(ape_a: np.ndarray, ape_b: np.ndarray, meses: np.ndarray,
             mes_fuera: str, rng: np.random.Generator) -> dict:
    """El contraste por filas excluyendo un mes, para publicar su peso."""
    keep = meses != mes_fuera
    a, b = ape_a[keep], ape_b[keep]
    idx = rng.integers(0, len(a), size=(N_BOOTSTRAP, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {
        "mes_excluido": mes_fuera,
        "n_filas": int(keep.sum()),
        "delta_pp": round(float(a.mean() - b.mean()), 2),
        "delta_medio_pp": round(float(d.mean()), 2),
        "IC95": _ic(d),
        "p_valor_una_cola": round(float(np.mean(d <= 0)), 4),
        "significativo_95": bool(np.percentile(d, 2.5) > 0),
    }


def ejecutar(csv_path: str = CSV_PATH, verbose: bool = True) -> dict:
    datos = recolectar(csv_path, verbose=verbose)
    pliegues = datos["pliegues"]
    rng = np.random.default_rng(RANDOM_STATE)

    # MAPE ponderado de cada serie sobre sus propias filas comunes con
    # `rodante`, que es la referencia de todos los contrastes.
    mape = {}
    for s in SERIES:
        if not any(s in p["preds"] for p in pliegues):
            continue
        meses, apes = _apilar(pliegues, ["rodante", s])
        mape[s] = {
            "MAPE_ponderado_%": round(float(apes[s].mean()), 2),
            "n_filas": int(len(apes[s])),
            "meses": int(len(np.unique(meses))),
        }

    if verbose:
        print("\n  MAPE ponderado por regimen (sobre las filas comunes con rodante):")
        for s, v in mape.items():
            print(f"    {SERIES[s]:<48} {v['MAPE_ponderado_%']:>6.2f}%  "
                  f"n={v['n_filas']:,}  ({v['meses']} meses)")

    # Los dos efectos del articulo, mas las tres lineas base y el regimen de
    # masa constante. En cada par, el primero es el que debe tener MAS error si
    # el efecto que el articulo afirma existe.
    pares = [
        ("ganancia_de_reentrenar", "estatico", "rodante"),
        ("coste_del_rezago_sunat", "rezago_sunat", "rodante"),
        ("vs_persistencia_lag1", "persistencia_lag1", "rodante"),
        ("vs_media_movil_ma3", "media_movil_ma3", "rodante"),
        ("vs_mediana_train", "mediana_train", "rodante"),
        ("recencia_a_masa_constante", "estatico", "ventana_fija"),
        ("masa_acumulada", "ventana_fija", "rodante"),
    ]

    contrastes = {}
    for nombre, peor, mejor in pares:
        if peor not in mape or mejor not in mape:
            continue
        meses, apes = _apilar(pliegues, [peor, mejor])
        c = _contraste(apes[peor], apes[mejor], meses, rng)
        c["comparacion"] = f"{SERIES[peor]} vs {SERIES[mejor]}"
        contrastes[nombre] = c
        if verbose:
            f, m = c["por_filas"], c["por_mes"]
            print(f"\n  {nombre}: delta {c['delta_pp']:+.2f} pp")
            print(f"    por filas  IC95 [{f['IC95'][0]:+.2f}, {f['IC95'][1]:+.2f}]  "
                  f"p={f['p_valor_una_cola']:.4f}  sig={f['significativo_95']}")
            print(f"    por mes    IC95 [{m['IC95'][0]:+.2f}, {m['IC95'][1]:+.2f}]  "
                  f"p={m['p_valor_una_cola']:.4f}  sig={m['significativo_95']}")

    # Peso del mes extremo. Se calcula sobre el mes de mayor delta mensual, que
    # se identifica aqui y no se escribe a mano.
    meses, apes = _apilar(pliegues, ["estatico", "rodante"])
    por_mes_delta = {
        m: round(float(apes["estatico"][meses == m].mean()
                       - apes["rodante"][meses == m].mean()), 2)
        for m in np.unique(meses)
    }
    mes_extremo = max(por_mes_delta, key=lambda k: por_mes_delta[k])
    sin_extremo = _sin_mes(apes["estatico"], apes["rodante"], meses, mes_extremo, rng)
    if verbose:
        print(f"\n  Ganancia de reentrenar por mes: {por_mes_delta}")
        print(f"  Excluyendo {mes_extremo}: delta {sin_extremo['delta_pp']:+.2f} pp "
              f"IC95 [{sin_extremo['IC95'][0]:+.2f}, {sin_extremo['IC95'][1]:+.2f}] "
              f"sig={sin_extremo['significativo_95']}")

    informe = {
        "procedimiento": (
            "Bootstrap pareado con B=2,000 remuestreos y semilla 42, el mismo "
            "procedimiento que ml/estudios_paper.py aplica a la comparacion de "
            "algoritmos y a la ablacion. Cada contraste se calcula sobre las filas "
            "en las que ambas series existen, con el mismo indice de remuestreo para "
            "las dos, y se reporta en dos unidades de remuestreo: la declaracion "
            "(`por_filas`) y el pliegue mensual (`por_mes`, bootstrap de bloques). "
            "El segundo es el intervalo pertinente para una afirmacion sobre el "
            "sistema, porque las declaraciones de un mismo mes comparten regimen de "
            "mercado y no son independientes."
        ),
        "n_bootstrap": N_BOOTSTRAP,
        "semilla": RANDOM_STATE,
        "n_estimators": datos["n_estimators"],
        "anio_evaluado": ANIO,
        "MAPE_por_regimen": mape,
        "contrastes": contrastes,
        "ganancia_de_reentrenar_por_mes_pp": por_mes_delta,
        "ganancia_sin_el_mes_extremo": sin_extremo,
        "confound_masa_de_datos": {
            "crecimiento_del_bloque_%": datos["crecimiento_bloque_%"],
            "n_bloque_primer_pliegue": pliegues[0]["n_bloque"],
            "n_bloque_ultimo_pliegue": pliegues[-1]["n_bloque"],
            "nota": (
                "El bloque de ajuste crece a lo largo del anio, de modo que la "
                "ganancia de reentrenar recoge conjuntamente recencia y masa de "
                "datos. `recencia_a_masa_constante` y `masa_acumulada` las separan: "
                "el regimen `ventana_fija` ajusta cada pliegue sobre las N filas mas "
                "recientes de su bloque, con N igual al tamano del bloque del primer "
                "pliegue, de modo que solo varia la recencia."
            ),
        },
        "detalle_pliegues": [
            {"mes": p["mes"], "n_bloque": p["n_bloque"], "n_evaluado": p["n_evaluado"],
             "MAPE_%": {k: round(float(_ape(p["y"], v).mean()), 2)
                        for k, v in p["preds"].items()}}
            for p in pliegues
        ],
    }
    return informe


def main() -> None:
    informe = ejecutar()
    with open(INFORME_PATH, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado: {INFORME_PATH}")


if __name__ == "__main__":
    main()

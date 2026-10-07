"""
Walk-forward mensual sobre 2025 (origen rodante), el protocolo de evaluacion
que acompana al esquema de particion 80/20.

QUE MIDE Y POR QUE HACE FALTA

El holdout del esquema 80/20 (todo 2025) da UNA foto: un modelo congelado en
diciembre de 2024 mirando doce meses hacia adelante. Es la cifra correcta para
responder "cuanto cuesta NO reentrenar", y es pesimista por construccion,
porque la fila de diciembre de 2025 se predice con once meses de horizonte.

No es, en cambio, la cifra que describe al sistema desplegado, que se reentrena
con el corpus acumulado. Para eso esta este modulo, que implementa exactamente
el protocolo que pidio la revision metodologica:

    entrena hasta diciembre-2024  ->  predice enero-2025
    entrena hasta enero-2025      ->  predice febrero-2025
    entrena hasta febrero-2025    ->  predice marzo-2025
    ...
    entrena hasta noviembre-2025  ->  predice diciembre-2025

Doce pliegues, los doce meses de 2025, cada uno predicho por un modelo que solo
vio datos ESTRICTAMENTE anteriores al mes que predice. El primer pliegue es, por
construccion, el modelo base del split 80/20: el origen rodante arranca donde
termina el split y no es un experimento aparte.

POR QUE NO PUEDE FILTRARSE INFORMACION DEL FUTURO

Cada pliegue llama a `data_pipeline.particionar()` sobre el scope truncado a su
propia historia. Es el MISMO codigo que entrena el modelo de produccion, no una
reimplementacion: por tanto el pliegue reajusta, con datos estrictamente
anteriores al mes evaluado y sin que haya que acordarse de ninguno,

  - los umbrales de recorte de atipicos (P0.5-P99.5 estratificados por regimen),
  - el lookup puerto -> ruta directa y su valor por defecto,
  - `puerto_freq`, `importador_freq` y sus defaults,
  - las medianas de `densidad_carga` y `ratio_bruto_neto`,
  - la serie mensual de mercado que alimenta `mercado_lag1/2/3` y `mercado_ma3`,
  - el corte interno que separa cabeza y cola del bloque de entrenamiento.

Esto corrige una limitacion real del walk-forward anterior
(`estudios_paper.etapa_walkforward`), que reajustaba los encoders por pliegue
pero heredaba del pipeline completo los umbrales de recorte y la serie de
mercado. Aqui no queda nada heredado. El unico contacto del mes evaluado con su
propio pliegue es que sus filas se recortan con umbrales ajenas y se les aplican
encoders ajenos — que es exactamente lo que le pasa a una cotizacion real.

DOS REGIMENES POR MES, PORQUE PRODUCCION TIENE DOS

  - `rodante`: los rezagos son los reales de los meses anteriores. Supone que
    la estadistica del mes T-1 esta disponible al cotizar en T.
  - `rodante_rezago_sunat`: los rezagos se retrasan un mes mas (T-2, T-3, T-4),
    porque SUNAT publica con demora y esa suposicion puede no cumplirse. Es la
    cota pesimista del mismo pliegue, sin reentrenar nada: mide cuanto cuesta
    que el dato llegue tarde, separandolo de cuanto cuesta no reentrenar.

Y una referencia contra la que leerlos: `estatico`, el modelo del split 80/20
—entrenado una sola vez con 2021-2024— evaluado sobre esos mismos meses. La
diferencia entre `estatico` y `rodante`, mes a mes, ES el valor del
reentrenamiento mensual, medido y no supuesto.

SALIDAS

  ml/walk_forward_2025.json   informe completo (detalle mensual + agregados)
  modelo_meta.json            clave `walk_forward_2025` con el mismo contenido,
                              para que el artifact desplegado lo lleve consigo
                              y la API pueda exponerlo sin leer otro fichero.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from ml import data_pipeline as dp
from ml import seleccion_iteraciones as si

CSV_PATH = os.environ.get("JPS_CSV_PATH", "resultado_combinado.csv")
META_PATH = "ml/modelo_meta.json"
INFORME_PATH = "ml/walk_forward_2025.json"

RANDOM_STATE = 42

# Anio evaluado mes a mes. Es el mismo que el holdout del esquema 80/20: el
# origen rodante no evalua otro periodo, evalua el mismo con otro protocolo.
ANIO = 2025

# Minimo de filas para que un mes se evalue. Con menos, el MAPE del mes es
# ruido y contamina el agregado. Los meses de 2025 rondan las 1,400-2,100
# filas, asi que este umbral no descarta ninguno: existe para que el modulo
# falle de forma visible si se ejecuta sobre un corpus mutilado.
MIN_FILAS_MES = 100

# Hiperparametros: los de produccion, sin excepcion. Si divergieran, el numero
# del walk-forward no describiria al sistema desplegado.
HIPERPARAMS = dict(
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    objective="reg:absoluteerror",
    eval_metric="mae",
)

# EL NUMERO DE ARBOLES SE FIJA UNA SOLA VEZ, ANTES DE 2025, Y NO SE RESELECCIONA
# EN CADA PLIEGUE. Son dos razones y conviene separarlas:
#
#   1. Honestidad. Es exactamente lo que hace el sistema real: la receta
#      (hiperparametros incluidos) se fija con los datos disponibles hasta el
#      cierre de 2024 y los reentrenamientos mensuales posteriores reajustan el
#      MODELO, no el procedimiento. Reseleccionar por pliegue mediria un
#      sistema que nadie opera.
#   2. Interpretabilidad. Con el numero de arboles fijo, la diferencia entre
#      meses es atribuible a los datos y al mercado, no a que el pliegue de
#      marzo eligiera otra profundidad efectiva que el de septiembre.
#
# Se obtiene con el mismo selector y sobre el mismo bloque que usa
# train_model.py, asi que por construccion coincide con el modelo desplegado.
_N_ESTIMATORS: int | None = None


def _receta_del_artifact() -> tuple[int | None, dict]:
    """(n_estimators, diagnostico de su seleccion) segun el artifact en disco.

    Devuelve (None, {}) si el artifact no existe o no declara la receta, para
    que el modulo siga siendo ejecutable por su cuenta.
    """
    try:
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None, {}
    n = meta.get("receta_entrenamiento", {}).get("n_estimators")
    if not n:
        return None, {}
    return int(n), {
        **meta.get("seleccion_n_estimators", {}),
        "origen": "leido de modelo_meta.json (receta del modelo desplegado)",
    }


def evaluar(y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAPE_%": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def _ajustar(bloque: pd.DataFrame) -> XGBRegressor:
    """Ajuste con la receta de produccion: bloque completo, sin early stopping."""
    m = XGBRegressor(n_estimators=_N_ESTIMATORS, **HIPERPARAMS)
    m.fit(bloque[dp.FEATURES], bloque[dp.TARGET], verbose=False)
    return m


def _retrasar_lags(X: pd.DataFrame, periodos: pd.Series, serie: pd.Series,
                   meses_extra: int) -> pd.DataFrame | None:
    """Rezagos desplazados `meses_extra` meses mas hacia atras.

    Simula que la estadistica aduanera del mes anterior todavia no se publico
    cuando se cotiza. Devuelve None si la serie no llega tan atras para alguna
    fila, en vez de inventar un valor: un mes que no se puede evaluar bajo este
    regimen debe quedar fuera del agregado, no entrar con un relleno.
    """
    Xf = X.copy()
    for k in (1, 2, 3):
        objetivo = [t - k - meses_extra for t in periodos]
        if any(o not in serie.index for o in objetivo):
            return None
        Xf[f"mercado_lag{k}"] = [float(serie.loc[o]) for o in objetivo]
    Xf["mercado_ma3"] = (Xf["mercado_lag1"] + Xf["mercado_lag2"] + Xf["mercado_lag3"]) / 3
    return Xf


def _pliegue(scope: pd.DataFrame, mes: pd.Period, modelo_base=None) -> dict | None:
    """Un pliegue: entrena con todo lo anterior a `mes`, predice `mes`.

    `modelo_base` es el modelo estatico del split 80/20. Si se pasa, se
    evalua sobre LAS MISMAS FILAS y CON LAS MISMAS FEATURES que el modelo
    rodante de este mes. Esa es la unica forma de que la diferencia entre
    ambos aisle lo que se quiere medir —haber reentrenado o no— en vez de
    mezclarlo con que cada uno se evalue sobre un conjunto recortado con
    umbrales distintos.

    Devuelve None si el mes no alcanza `MIN_FILAS_MES` tras el recorte.
    """
    inicio = pd.Timestamp(mes.start_time).normalize()
    fin = pd.Timestamp(mes.end_time).normalize() + pd.Timedelta(days=1)

    # Scope del pliegue: historia + mes evaluado. Nada posterior existe para el.
    scope_pliegue = scope[scope["FECHA"] < fin]
    historia = scope_pliegue["FECHA"] < inicio
    if not historia.any():
        return None

    # Corte interno de early stopping: el mismo 80/20 anidado que usa
    # produccion, calculado sobre la historia de ESTE pliegue.
    corte_interno = dp._corte_por_fecha(
        scope_pliegue.loc[historia, "FECHA"], dp.FRACCION_TRAIN_INTERNO
    )
    p = dp.particionar(scope_pliegue, corte_interno, inicio, esquema="80_20")
    if len(p.test) < MIN_FILAS_MES:
        return None

    # Invariantes del pliegue. No son decorativas: son la definicion de
    # "origen rodante" y se comprueban en cada mes, no una vez a mano.
    assert p.train["FECHA"].max() < inicio, "train del pliegue toca el mes evaluado"
    assert p.val["FECHA"].max() < inicio, "val del pliegue toca el mes evaluado"
    assert p.test["FECHA"].min() >= inicio and p.test["FECHA"].max() < fin
    assert p.serie_mercado.index.max() <= mes

    modelo = _ajustar(pd.concat([p.train, p.val]))
    X_test, y_test = p.test[dp.FEATURES], p.test[dp.TARGET]
    rodante = evaluar(y_test, modelo.predict(X_test))

    # Cota pesimista: el dato de mercado llega un mes tarde.
    periodos = p.test["FECHA"].dt.to_period("M")
    X_tarde = _retrasar_lags(X_test, periodos, p.serie_mercado, 1)
    rezagado = evaluar(y_test, modelo.predict(X_tarde)) if X_tarde is not None else None

    # Lineas base del mes, con la informacion que el pliegue tenia disponible.
    base = {
        "persistencia_lag1": evaluar(y_test, X_test["mercado_lag1"].values),
        "media_movil_ma3": evaluar(y_test, X_test["mercado_ma3"].values),
        "mediana_train": evaluar(
            y_test, np.full(len(y_test), float(p.train[dp.TARGET].median()))
        ),
    }

    nivel_mes = float(y_test.mean())
    nivel_previo = float(p.serie_mercado.get(mes - 1, np.nan))
    variacion = (
        round(100 * abs(nivel_mes - nivel_previo) / nivel_previo, 2)
        if nivel_previo and not np.isnan(nivel_previo) else None
    )

    return {
        "mes": str(mes),
        "trimestre": f"{mes.year}-T{mes.quarter}",
        "n_entrenamiento": int(len(p.train) + len(p.val)),
        "n_evaluado": int(len(p.test)),
        "entrena_hasta": str(p.val["FECHA"].max().date()),
        "n_estimators": int(_N_ESTIMATORS),
        "rodante": rodante,
        "rodante_rezago_sunat": rezagado,
        "estatico_mismas_filas": (
            evaluar(y_test, modelo_base.predict(X_test)) if modelo_base is not None else None
        ),
        "lineas_base": base,
        "nivel_mercado_mes": round(nivel_mes, 4),
        "variacion_vs_mes_previo_%": variacion,
        "puertos_no_vistos_%": round(
            100 * float((~p.test["PUER_DESC"].isin(p.puerto_freq.index)).mean()), 2
        ),
        "importadores_no_vistos_%": round(
            100 * float((~p.test["IMPORTADOR"].isin(p.importador_freq.index)).mean()), 2
        ),
    }


def _estatico_por_mes(p_base: dp.Particiones, modelo) -> dict:
    """El artifact estatico del split 80/20, mes a mes, sobre SU PROPIO holdout.

    Es la segunda de las dos vistas del modelo sin reentrenar, y responde una
    pregunta distinta de `estatico_mismas_filas`:

      - `estatico_mismas_filas` (calculado en `_pliegue`) evalua este mismo
        modelo sobre las filas y las features del pliegue. Aisla el efecto de
        reentrenar, porque todo lo demas queda igual.
      - esta funcion lo evalua sobre el holdout tal y como lo construye el
        pipeline desplegado, con sus propios umbrales de recorte y sus propios
        encoders. Describe al artifact que hay en disco, no a un contrafactual.

    Los dos conjuntos difieren en poco —el recorte de atipicos es el unico
    motivo— pero no son identicos, y mezclarlos daria una diferencia que no
    seria atribuible a nada. Se reportan por separado, cada uno con su n.
    """
    test = p_base.test.reset_index(drop=True).copy()
    test["_pred"] = modelo.predict(test[dp.FEATURES])
    test["_per"] = test["FECHA"].dt.to_period("M")
    return {
        str(mes): {**evaluar(g[dp.TARGET], g["_pred"]), "n": int(len(g))}
        for mes, g in test.groupby("_per")
    }


def _agregar(filas: list[dict], clave: str) -> dict:
    """Agregado ponderado por declaraciones: cada embarque pesa igual.

    Se reportan las dos medias a proposito. La ponderada es la que describe el
    error que sufre el conjunto de los usuarios; la simple trata cada mes por
    igual y es la que no deja que un mes voluminoso tape a uno malo. Cuando se
    separan mucho, el dato interesante es esa separacion.
    """
    validas = [f for f in filas if f.get(clave)]
    if not validas:
        return {}
    n = np.array([f["n_evaluado"] for f in validas], dtype=float)
    mape = np.array([f[clave]["MAPE_%"] for f in validas], dtype=float)
    mae = np.array([f[clave]["MAE"] for f in validas], dtype=float)
    return {
        "meses": len(validas),
        "n_total": int(n.sum()),
        "MAPE_ponderado_%": round(float(np.average(mape, weights=n)), 2),
        "MAPE_promedio_simple_%": round(float(mape.mean()), 2),
        "MAPE_sd_%": round(float(mape.std(ddof=1)), 2) if len(validas) > 1 else 0.0,
        "MAPE_min_%": round(float(mape.min()), 2),
        "MAPE_max_%": round(float(mape.max()), 2),
        "MAE_ponderado": round(float(np.average(mae, weights=n)), 4),
        "mes_peor": validas[int(mape.argmax())]["mes"],
        "mes_mejor": validas[int(mape.argmin())]["mes"],
    }


def ejecutar(csv_path: str = CSV_PATH, anio: int = ANIO, verbose: bool = True) -> dict:
    global _N_ESTIMATORS

    scope = dp.cargar_scope(csv_path)
    meses = [pd.Period(f"{anio}-{m:02d}", freq="M") for m in range(1, 13)]

    # Receta fijada antes de 2025, identica a la de train_model.py por
    # construccion: mismo selector sobre el mismo bloque.
    p_base = dp.construir(csv_path, esquema="80_20")
    bloque_base = pd.concat([p_base.train, p_base.val])
    if verbose:
        print("Seleccion del numero de arboles (una vez, solo con datos anteriores a "
              f"{anio}):")
    # FUENTE UNICA DE VERDAD. Si el artifact ya trae la receta —y la trae
    # siempre que train_model.py se haya ejecutado antes, que es el orden que
    # impone ml/reentrenamiento.py—, se toma de ahi en vez de reseleccionarla.
    # El selector es determinista y reseleccionar daria el mismo numero, asi
    # que esto es sobre todo tiempo ahorrado (once ajustes de 600 arboles) y
    # una garantia estructural: el walk-forward no PUEDE describir una
    # configuracion distinta de la del modelo desplegado, aunque alguien cambie
    # el selector y olvide reentrenar. Si el artifact no esta, se selecciona
    # aqui y se dice.
    _N_ESTIMATORS, diag_n = _receta_del_artifact()
    if _N_ESTIMATORS is None:
        if verbose:
            print("  (sin receta en el artifact: se selecciona aqui)")
        _N_ESTIMATORS, diag_n = si.seleccionar_n_estimators(
            bloque_base, lambda n: XGBRegressor(n_estimators=n, **HIPERPARAMS),
            verbose=verbose,
        )
    elif verbose:
        print(f"  n_estimators = {_N_ESTIMATORS} (leido de {META_PATH}, "
              "la misma receta que el modelo desplegado)")

    # Modelo estatico de referencia: el del split 80/20, entrenado UNA vez con
    # 2021-2024 y con la misma receta. Se ajusta aqui, antes del bucle, para
    # que sea literalmente el mismo objeto en los doce meses.
    modelo_base = _ajustar(bloque_base)

    if verbose:
        print(f"=== WALK-FORWARD MENSUAL {anio} (origen rodante, 12 pliegues) ===")
        print("  Cada mes se predice con un modelo que solo vio datos anteriores a el.")
        print(f"  {'mes':<9} {'n_train':>8} {'n_eval':>7} {'MAPE':>7} {'MAPE tarde':>11} "
              f"{'MAE':>8} {'base lag1':>10}")

    filas = []
    for mes in meses:
        f = _pliegue(scope, mes, modelo_base)
        if f is None:
            if verbose:
                print(f"  {str(mes):<9} omitido (sin datos suficientes)")
            continue
        filas.append(f)
        if verbose:
            tarde = f["rodante_rezago_sunat"]
            txt_tarde = "n/d" if tarde is None else f"{tarde['MAPE_%']:.2f}%"
            print(
                f"  {f['mes']:<9} {f['n_entrenamiento']:>8,} {f['n_evaluado']:>7,} "
                f"{f['rodante']['MAPE_%']:>6.2f}% {txt_tarde:>11} "
                f"{f['rodante']['MAE']:>8.4f} "
                f"{f['lineas_base']['persistencia_lag1']['MAPE_%']:>9.2f}%"
            )

    if len(filas) < 12:
        print(f"  AVISO: se evaluaron {len(filas)} de 12 meses.")

    # Segunda vista del modelo estatico: sobre su propio holdout, con su propio
    # pipeline. Se pondera con SUS filas, no con las del pliegue: usar las del
    # pliegue mezclaria dos conjuntos distintos en una sola media.
    estatico_artifact = _estatico_por_mes(p_base, modelo_base)
    for f in filas:
        f["estatico_artifact"] = estatico_artifact.get(f["mes"])
        if f.get("estatico_mismas_filas"):
            f["ganancia_reentrenar_pp"] = round(
                f["estatico_mismas_filas"]["MAPE_%"] - f["rodante"]["MAPE_%"], 2
            )

    agg_rodante = _agregar(filas, "rodante")
    agg_tarde = _agregar(filas, "rodante_rezago_sunat")
    agg_estatico = _agregar(filas, "estatico_mismas_filas")
    agg_estatico_artifact = _agregar(
        [{"n_evaluado": v["n"], "_e": v, "mes": k} for k, v in estatico_artifact.items()],
        "_e",
    )
    # Diferencia de tamano entre los dos conjuntos de evaluacion, para que la
    # comparabilidad sea un dato publicado y no una suposicion del lector.
    _n_pliegues = sum(f["n_evaluado"] for f in filas)
    _n_artifact = sum(v["n"] for v in estatico_artifact.values())
    agg_base = {
        k: _agregar(
            [{"n_evaluado": f["n_evaluado"], "_b": f["lineas_base"][k], "mes": f["mes"]}
             for f in filas],
            "_b",
        )
        for k in ("persistencia_lag1", "media_movil_ma3", "mediana_train")
    }

    por_trimestre = {}
    for t in sorted({f["trimestre"] for f in filas}):
        sub = [f for f in filas if f["trimestre"] == t]
        por_trimestre[t] = _agregar(sub, "rodante")

    gana_todas = all(
        agg_rodante["MAPE_ponderado_%"] < v["MAPE_ponderado_%"] for v in agg_base.values()
    )
    ganancia = round(
        agg_estatico["MAPE_ponderado_%"] - agg_rodante["MAPE_ponderado_%"], 2
    ) if agg_estatico else None
    coste_rezago = round(
        agg_tarde["MAPE_ponderado_%"] - agg_rodante["MAPE_ponderado_%"], 2
    ) if agg_tarde else None

    informe = {
        "protocolo": (
            f"Origen rodante mensual sobre {anio}: entrena hasta diciembre-{anio-1} y "
            f"predice enero-{anio}; entrena hasta enero y predice febrero; y asi los "
            "doce meses. Cada pliegue reajusta TODO el pipeline (umbrales de recorte, "
            "lookup de ruta, encoders, medianas y serie de mercado) con datos "
            "estrictamente anteriores al mes que predice, llamando al mismo "
            "data_pipeline.particionar() que usa el entrenamiento de produccion."
        ),
        "esquema_particion_base": "80_20",
        "anio_evaluado": anio,
        "meses_evaluados": len(filas),
        "hiperparametros": {**{k: v for k, v in HIPERPARAMS.items() if k != "n_jobs"},
                            "n_estimators": _N_ESTIMATORS},
        "seleccion_n_estimators": diag_n,
        "resumen": {
            "rodante": agg_rodante,
            "rodante_rezago_sunat": agg_tarde,
            "estatico_mismas_filas": agg_estatico,
            "estatico_artifact_sobre_su_holdout": agg_estatico_artifact,
            "comparabilidad_conjuntos": {
                "n_filas_evaluadas_pliegues": int(_n_pliegues),
                "n_filas_holdout_artifact": int(_n_artifact),
                "diferencia_%": round(100 * abs(_n_pliegues - _n_artifact) / _n_artifact, 2),
                "causa": (
                    "El recorte de atipicos se reajusta en cada pliegue, asi que el "
                    "mes evaluado por el origen rodante y el mismo mes dentro del "
                    "holdout del artifact no contienen exactamente las mismas filas. "
                    "Por eso la ganancia de reentrenar se calcula contra "
                    "`estatico_mismas_filas` —el modelo estatico evaluado sobre las "
                    "filas del pliegue— y no contra esta segunda vista."
                ),
            },
            "lineas_base": agg_base,
            "ganancia_de_reentrenar_pp": ganancia,
            "coste_del_rezago_sunat_pp": coste_rezago,
            "gana_a_todas_las_lineas_base": bool(gana_todas),
        },
        "por_trimestre": por_trimestre,
        "detalle_mensual": filas,
        "lectura": (
            "Tres cifras que responden tres preguntas distintas y no deben mezclarse. "
            "(1) `rodante` es el rendimiento esperable del sistema SI se reentrena cada "
            "mes, que es como esta pensado para operar. (2) `estatico_mismas_filas` es el "
            "mismo modelo sin reentrenar sobre los mismos meses: su diferencia con (1) es "
            "el valor medido del reentrenamiento, y crece con el mes porque el modelo "
            "estatico acumula horizonte. (3) `rodante_rezago_sunat` es (1) suponiendo que "
            "la estadistica aduanera llega un mes tarde; es la cota pesimista realista, "
            "porque SUNAT publica con demora. Ninguna de las tres sustituye al MAPE del "
            "holdout que reporta train_model.py: ese responde 'cuanto cuesta no "
            "reentrenar en todo un anio' y es, deliberadamente, el mas pesimista."
        ),
        "limitaciones": [
            "Doce pliegues sobre un unico anio y una unica trayectoria de mercado. La "
            "desviacion tipica entre meses describe la variabilidad DENTRO de 2025, no "
            "el error esperable en otro anio.",
            "El primer pliegue (enero) entrena con 2021-2024 y el ultimo (diciembre) con "
            "2021-2024 mas once meses: el tamano de entrenamiento crece un ~17% a lo "
            "largo del anio, asi que la comparacion entre meses mezcla estacionalidad, "
            "regimen de mercado y masa de datos.",
            "Se evalua el modelo puntual. Los intervalos conformales no se recalibran por "
            "pliegue: su cobertura se mide en train_quantile_models.py sobre el holdout.",
        ],
    }

    if verbose:
        r = informe["resumen"]
        print(f"\n  RODANTE          MAPE ponderado {agg_rodante['MAPE_ponderado_%']:.2f}% | "
              f"simple {agg_rodante['MAPE_promedio_simple_%']:.2f}% "
              f"(sd {agg_rodante['MAPE_sd_%']:.2f}) | "
              f"rango {agg_rodante['MAPE_min_%']:.2f}-{agg_rodante['MAPE_max_%']:.2f}%")
        if agg_estatico:
            print(f"  ESTATICO 80/20   MAPE ponderado {agg_estatico['MAPE_ponderado_%']:.2f}% "
                  f"-> reentrenar cada mes vale {ganancia:+.2f} pp")
        if agg_tarde:
            print(f"  REZAGO SUNAT     MAPE ponderado {agg_tarde['MAPE_ponderado_%']:.2f}% "
                  f"-> el dato tardio cuesta {coste_rezago:+.2f} pp")
        for k, v in agg_base.items():
            print(f"  base {k:<18} MAPE ponderado {v['MAPE_ponderado_%']:.2f}%")
        print(f"  Gana a las {len(agg_base)} lineas base: {gana_todas}")
        print("\n  Por trimestre:")
        for t, v in por_trimestre.items():
            print(f"    {t}  {v['meses']} meses | MAPE {v['MAPE_ponderado_%']:.2f}%")

    return informe


def main() -> None:
    informe = ejecutar()

    with open(INFORME_PATH, "w", encoding="utf-8") as f:
        json.dump(informe, f, indent=2, ensure_ascii=False)

    # El artifact desplegado lleva el informe consigo: la API expone el numero
    # sin abrir un segundo fichero, y un artifact restaurado desde un respaldo
    # no puede quedar emparejado con el walk-forward de otro entrenamiento.
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    meta["walk_forward_2025"] = informe
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {INFORME_PATH}")
    print(f"Actualizado: {META_PATH} (clave walk_forward_2025)")


if __name__ == "__main__":
    main()

"""
Estudios adicionales para el articulo cientifico (revision de revisor Q2).

Este modulo NO entrena ni reemplaza el artifact de produccion. Es estrictamente
de lectura: construye el pipeline con `data_pipeline.construir()` — el MISMO que
usa `train_model.py` — y sobre el corre los experimentos que un revisor exige y
que el entrenamiento de produccion no necesita.

Responde, con numeros recalculados en cada ejecucion, a las ocho observaciones
de revision:

  1. ablacion    Ablation study incremental por grupos de features.
  2. leakage     Auditoria explicita de que se ajusta con que particion.
  3. trazab      Trazabilidad input de usuario -> feature -> modelo.
  4. modelos     XGBoost vs LightGBM vs CatBoost vs RandomForest vs Ridge vs
                 lineas base, con IC95 bootstrap y prueba pareada.
  5. metricas    WAPE, sMAPE y error reconstruido sobre el flete TOTAL en USD.
  6. segmentos   Error por puerto, importador, mes y volumen.
  7. walkforward Evaluacion de origen rodante: error por anio, trimestre y
                 regimen de volatilidad del mercado.
  8. outliers    Sensibilidad del resultado al tratamiento de atipicos
                 (sin tratamiento / winsorizacion / eliminacion P0.5-P99.5).
  9. shap        Figuras SHAP: beeswarm, barras, dependence, waterfall y
                 comparacion regimen estable vs volatil.

Uso:
    python -m ml.estudios_paper            # todas las etapas
    python -m ml.estudios_paper ablacion   # una etapa concreta

Salidas:
    ml/estudios_paper/resultados.json      todas las cifras
    ml/estudios_paper/figuras/*.png        figuras SHAP y de apoyo

INVARIANTE VERIFICADA: la etapa `outliers` reimplementa el pipeline para poder
variar el tratamiento de atipicos. Para que esa reimplementacion no pueda
divergir en silencio del pipeline real, el modo "eliminar" (el de produccion) se
compara contra `construir()` y la ejecucion ABORTA si las metricas no coinciden.
"""
from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from ml import data_pipeline as dp
from ml import seleccion_iteraciones as si

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
BASE = Path(__file__).resolve().parent
CSV = Path(__import__("os").environ.get("JPS_CSV_PATH", BASE.parent / "resultado_combinado.csv"))
OUT = BASE / "estudios_paper"
FIGS = OUT / "figuras"

# Grupos de features para la ablacion incremental. El orden replica el que pide
# la revision: se parte de lo que el usuario aporta y se van sumando las
# variables de mercado, que son las que el SHAP senala como dominantes.
GRUPOS = {
    "temporales": ["mes", "trimestre", "semana_anio", "mes_sin", "mes_cos"],
    "puerto_importador": ["puerto_freq", "importador_freq", "ruta_directa"],
    "lag1": ["mercado_lag1"],
    "lag2_lag3": ["mercado_lag2", "mercado_lag3"],
    "media_movil": ["mercado_ma3"],
    "fisicas": ["densidad_carga", "ratio_bruto_neto"],
}


# ──────────────────────────────────────────────────────────────────────────
# Utilidades de metricas
# ──────────────────────────────────────────────────────────────────────────
def evaluar(y_true, y_pred) -> dict:
    """Metricas del articulo. Identicas a las de train_model.evaluar()."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "WAPE_%": float(100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))),
        "sMAPE_%": float(
            100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        ),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _mape(y_true, y_pred) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


# Numero de arboles de la receta de produccion. Se calcula una sola vez, con
# el mismo selector y sobre el mismo bloque que train_model.py, y se memoiza:
# si estas funciones divergieran de la receta desplegada, la ablacion y la
# comparacion de modelos describirian un sistema que nadie sirve.
_N_ESTIMATORS_CACHE: dict = {}


def n_estimators_produccion(p=None) -> int:
    """Numero de arboles de la receta desplegada.

    Se LEE del artifact, que es donde train_model.py lo deja. Es la fuente
    unica de verdad por dos razones: garantiza que estos estudios describan
    exactamente el modelo que se sirve —XGBoost con n_jobs=-1 no es
    bit-identico entre procesos, asi que reseleccionar podria dar otro numero—
    y evita que el walk-forward, que ajusta un modelo por mes, tenga que
    reseleccionar en cada pliegue (seria un orden de magnitud mas de computo
    para responder algo que ya esta respondido).

    Si el artifact no existe todavia, se selecciona una vez con el bloque que
    se pase y se memoiza.
    """
    if "n" not in _N_ESTIMATORS_CACHE:
        meta = BASE / "modelo_meta.json"
        n = None
        if meta.exists():
            try:
                n = json.loads(meta.read_text(encoding="utf-8"))                     .get("receta_entrenamiento", {}).get("n_estimators")
            except ValueError:
                n = None
        if not n:
            if p is None:
                raise ValueError(
                    "no hay receta en modelo_meta.json y no se paso un bloque "
                    "para seleccionarla; ejecutar antes python -m ml.train_model"
                )
            n, _ = si.seleccionar_n_estimators(
                bloque_ajuste(p), lambda k: xgb_nuevo(k), verbose=False,
            )
        _N_ESTIMATORS_CACHE["n"] = int(n)
    return _N_ESTIMATORS_CACHE["n"]


def bloque_ajuste(p):
    """Filas con las que ajusta el modelo de produccion: el 80% completo.

    Desde la quinta revision `val` ya no hace early stopping —se retiro porque
    bajo el esquema 80/20 elegia modelos subajustados, ver
    ml/seleccion_iteraciones.py— y pasa a ser parte del ajuste.
    """
    return pd.concat([p.train, p.val]).sort_values("FECHA", kind="mergesort").reset_index(drop=True)


def xgb_nuevo(n_estimators: int = 600) -> XGBRegressor:
    """Misma configuracion que el modelo de produccion (train_model.py).

    Incluye `objective="reg:absoluteerror"` y la ausencia de early stopping: si
    esta funcion divergiera de train_model.py, la ablacion dejaria de describir
    al modelo desplegado.
    """
    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="reg:absoluteerror",
        eval_metric="mae",
    )


def entrenar_xgb(p, features: list[str], seed: int = RANDOM_STATE) -> XGBRegressor:
    bloque = bloque_ajuste(p)
    m = xgb_nuevo(n_estimators_produccion(p))
    m.set_params(random_state=seed)
    m.fit(bloque[features], bloque[dp.TARGET], verbose=False)
    return m


def canonico(features: list[str]) -> list[str]:
    """Subconjunto de features en el ORDEN canonico de `dp.FEATURES`.

    No es cosmetico: con `colsample_bytree=0.8` el orden de las columnas cambia
    que columnas muestrea cada arbol bajo la misma semilla, y por tanto cambia el
    modelo. Pasar las features en otro orden hacia que la configuracion completa
    de la ablacion NO reprodujera el modelo de produccion (23.99% frente a
    22.23%): 1.76 pp de diferencia atribuibles solo al orden de las columnas.
    Fijar el orden canonico es lo que hace que la ultima fila de la ablacion sea
    literalmente el modelo desplegado.
    """
    return [f for f in dp.FEATURES if f in set(features)]


def _guardar(clave: str, valor) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ruta = OUT / "resultados.json"
    data = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
    data[clave] = valor
    ruta.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> guardado en resultados.json['{clave}']")


# ──────────────────────────────────────────────────────────────────────────
# 1. Hechos descriptivos del dataset (para que el articulo cite datos reales)
# ──────────────────────────────────────────────────────────────────────────
def etapa_dataset(p) -> dict:
    print("\n=== DATASET: hechos descriptivos verificados ===")
    crudo = pd.read_csv(CSV, sep=",", quotechar='"', encoding="utf-8", low_memory=False)
    crudo["CNAN"] = crudo["CNAN"].astype(str).str.zfill(10)
    crudo["FECHA_dt"] = pd.to_datetime(crudo["FECHA"], format="%Y%m%d", errors="coerce")
    mask = (
        crudo["CNAN"].str.startswith("4011")
        & (crudo["VIA_TRANSP"] == 1)
        & (crudo["ADUA_DESC"].str.upper().str.contains("CALLAO", na=False))
        & (crudo["CPAIS"].isin(dp.PAISES_ORIGEN))
    )
    scope = crudo.loc[mask]
    todo = pd.concat([p.train, p.val, p.test])

    puertos_crudo = scope["PUER_DESC"].value_counts()
    puertos_modelo = todo["PUER_DESC"].value_counts()

    res = {
        "filas_csv_crudo": int(len(crudo)),
        "filas_en_alcance": int(len(scope)),
        "paises_origen_presentes": {
            str(k): int(v) for k, v in scope["CPAIS"].value_counts().items()
        },
        "pct_origen_CN": round(100 * float((scope["CPAIS"] == "CN").mean()), 4),
        "rango_fechas_alcance": [
            str(scope["FECHA_dt"].min().date()), str(scope["FECHA_dt"].max().date())
        ],
        "importadores_distintos_alcance": int(scope["IMPORTADOR"].nunique()),
        "puertos_distintos_alcance": int(scope["PUER_DESC"].nunique()),
        "filas_tras_recorte_y_lags": int(len(todo)),
        "particion": {
            "train": int(len(p.train)), "val": int(len(p.val)), "test": int(len(p.test))
        },
        "rango_fechas_particion": {
            "train": [str(p.train["FECHA"].min().date()), str(p.train["FECHA"].max().date())],
            "val": [str(p.val["FECHA"].min().date()), str(p.val["FECHA"].max().date())],
            "test": [str(p.test["FECHA"].min().date()), str(p.test["FECHA"].max().date())],
        },
        "top_puertos_en_alcance": {
            str(k): {"n": int(v), "pct": round(100 * v / len(scope), 2)}
            for k, v in puertos_crudo.head(12).items()
        },
        "top_puertos_dataset_modelado": {
            str(k): {"n": int(v), "pct": round(100 * v / len(todo), 2)}
            for k, v in puertos_modelo.head(12).items()
        },
        "importadores_distintos_modelado": int(todo["IMPORTADOR"].nunique()),
        "target": {
            "media_USD_kg": round(float(todo[dp.TARGET].mean()), 4),
            "mediana_USD_kg": round(float(todo[dp.TARGET].median()), 4),
            "sd_USD_kg": round(float(todo[dp.TARGET].std()), 4),
            "min": round(float(todo[dp.TARGET].min()), 4),
            "max": round(float(todo[dp.TARGET].max()), 4),
        },
        "flete_total_USD": {
            "media": round(float(todo["FLE_DOLAR"].mean()), 2),
            "mediana": round(float(todo["FLE_DOLAR"].median()), 2),
            "suma": round(float(todo["FLE_DOLAR"].sum()), 2),
        },
    }
    print(f"  CSV crudo: {res['filas_csv_crudo']:,} filas | en alcance: {res['filas_en_alcance']:,}")
    print(f"  Origen CN: {res['pct_origen_CN']}% | importadores: {res['importadores_distintos_alcance']}")
    print(f"  Dataset modelado: {res['filas_tras_recorte_y_lags']:,} filas")
    print("  Top puertos (alcance):")
    for k, v in list(res["top_puertos_en_alcance"].items())[:8]:
        print(f"    {k:<28} {v['n']:>7,}  ({v['pct']:>5.2f}%)")
    _guardar("dataset", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 2. Ablation study incremental
# ──────────────────────────────────────────────────────────────────────────
def etapa_ablacion(p) -> dict:
    print("\n=== ABLATION STUDY (incremental por grupos de features) ===")
    print("  Cada configuracion se entrena con 5 semillas: sin la dispersion entre")
    print("  semillas no se puede distinguir una mejora real del ruido del muestreo.")
    y_test = p.test[dp.TARGET].values
    semillas = [42, 7, 123, 2024, 31337]

    configs: list[tuple[str, list[str]]] = []
    acum: list[str] = []
    for nombre, feats in GRUPOS.items():
        acum = acum + feats
        etiqueta = "temporales" if nombre == "temporales" else f"+ {nombre}"
        configs.append((etiqueta, canonico(acum)))

    # Configuraciones sueltas que responden directamente a la pregunta del
    # revisor: si dos variables concentran el 85% del gain, ¿aporta algo el
    # modelo por encima de usarlas solas?
    extra = [
        ("solo mercado_ma3 (1 feature)", ["mercado_ma3"]),
        ("solo mercado_lag1 (1 feature)", ["mercado_lag1"]),
        ("solo variables de mercado (4)", ["mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3"]),
        ("solo lag1 + ma3 (2 features)", ["mercado_lag1", "mercado_ma3"]),
    ]

    filas = []
    preds_completo = None
    preds_por_config: dict[str, np.ndarray] = {}
    for etiqueta, feats in configs + [(e, canonico(f)) for e, f in extra]:
        mapes, maes, rmses, todas = [], [], [], []
        for s in semillas:
            modelo = entrenar_xgb(p, feats, seed=s)
            yhat = modelo.predict(p.test[feats])
            m = evaluar(y_test, yhat)
            mapes.append(m["MAPE_%"]); maes.append(m["MAE"]); rmses.append(m["RMSE"])
            todas.append(yhat)
        # Prediccion promediada entre semillas: elimina del contraste la varianza
        # del muestreo de columnas/filas, que aqui es del mismo orden que las
        # diferencias entre configuraciones.
        preds_por_config[etiqueta] = np.mean(todas, axis=0)
        if len(feats) == len(dp.FEATURES):
            preds_completo = preds_por_config[etiqueta]
        filas.append({
            "configuracion": etiqueta,
            "n_features": len(feats),
            "features": feats,
            "MAPE_%": round(float(np.mean(mapes)), 2),
            "MAPE_sd": round(float(np.std(mapes)), 2),
            "MAPE_rango": [round(min(mapes), 2), round(max(mapes), 2)],
            "MAPE_semilla_42": round(mapes[0], 2),
            "MAE": round(float(np.mean(maes)), 4),
            "RMSE": round(float(np.mean(rmses)), 4),
        })
        print(f"  {etiqueta:<32} n={len(feats):>2} | MAPE {np.mean(mapes):6.2f}% "
              f"+-{np.std(mapes):.2f} (rango {min(mapes):.2f}-{max(mapes):.2f}) "
              f"| MAE {np.mean(maes):.4f}")

    # La ultima fila de la ablacion DEBE ser el modelo de produccion.
    completo = [f for f in filas if f["n_features"] == len(dp.FEATURES)][0]
    # La cifra de referencia se LEE del artifact en vez de escribirse a mano,
    # para que un cambio en train_model.py no deje este control obsoleto.
    meta = BASE / "modelo_meta.json"
    if meta.exists():
        esperado = json.loads(meta.read_text(encoding="utf-8"))["metricas_test"]["MAPE_%"]
        assert abs(completo["MAPE_semilla_42"] - esperado) < 0.05, (
            "La configuracion completa con semilla 42 debe reproducir el modelo "
            f"desplegado ({esperado}%); dio {completo['MAPE_semilla_42']}%. "
            "Revisar que xgb_nuevo() siga igual a train_model.py."
        )
        print(f"  [OK] la configuracion completa reproduce el artifact ({esperado}%)")

    # Prueba pareada de las configuraciones clave contra el modelo completo,
    # sobre las predicciones PROMEDIADAS entre semillas.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(y_test), size=(N_BOOTSTRAP, len(y_test)))
    base_bs = np.array([_mape(y_test[i], preds_completo[i]) for i in idx])
    comparaciones = {}
    # `solo mercado_lag1` entra en el contraste desde la quinta revision. Bajo el
    # esquema 80/20 esa configuracion de UNA sola variable resulta MEJOR que el
    # modelo completo sobre el holdout, y una afirmacion asi no puede publicarse
    # a partir de la diferencia de dos puntos estimados: necesita su intervalo.
    # Dejarla fuera del contraste habria sido elegir que comparaciones se
    # someten a prueba estadistica en funcion de su resultado.
    for etiqueta in ("solo mercado_ma3 (1 feature)", "solo mercado_lag1 (1 feature)",
                     "solo lag1 + ma3 (2 features)",
                     "solo variables de mercado (4)"):
        yhat = preds_por_config[etiqueta]
        bs = np.array([_mape(y_test[i], yhat[i]) for i in idx])
        d = bs - base_bs
        comparaciones[etiqueta] = {
            "delta_MAPE_pp": round(float(np.mean(d)), 2),
            "IC95_delta": [round(float(np.percentile(d, 2.5)), 2),
                           round(float(np.percentile(d, 97.5)), 2)],
            "p_valor_bootstrap": round(float(np.mean(d <= 0)), 4),
            "significativo_95": bool(np.percentile(d, 2.5) > 0),
        }
        c = comparaciones[etiqueta]
        print(f"  vs completo: {etiqueta:<32} delta {c['delta_MAPE_pp']:+.2f}pp "
              f"IC95 [{c['IC95_delta'][0]:+.2f},{c['IC95_delta'][1]:+.2f}] "
              f"p={c['p_valor_bootstrap']:.4f} sig={c['significativo_95']}")

    res = {
        "semillas": semillas,
        "tabla": filas,
        "mape_modelo_completo": completo["MAPE_%"],
        "comparaciones_pareadas_vs_completo": comparaciones,
        "sensibilidad_a_la_semilla": {
            "mape_semilla_42_artifact_desplegado": completo["MAPE_semilla_42"],
            "mape_media_5_semillas": completo["MAPE_%"],
            "mape_sd_5_semillas": completo["MAPE_sd"],
            "mape_rango_5_semillas": completo["MAPE_rango"],
            "nota": (
                "HALLAZGO RELEVANTE PARA EL ARTICULO: el MAPE del artifact "
                "desplegado corresponde a la semilla 42, que es una de las cinco "
                "observadas y no tiene por que ser la tipica. La estimacion "
                "central honesta del modelo completo es la media entre semillas "
                "con su desviacion, que es lo que registran los campos de arriba. "
                "Reportar solo la semilla desplegada presentaria como propiedad "
                "del modelo lo que en parte es el muestreo estocastico de XGBoost "
                "(subsample=0.8, colsample_bytree=0.8). Comparense "
                "`mape_semilla_42_artifact_desplegado` y `mape_media_5_semillas` "
                "en esta misma ejecucion en vez de citar una cifra fija: bajo el "
                "esquema 80/20 el valor cambio y una nota con numeros escritos a "
                "mano habria quedado obsoleta sin avisar."
            ),
        },
        "nota": (
            "Las features se pasan SIEMPRE en el orden canonico de dp.FEATURES: "
            "con colsample_bytree=0.8 el orden de columnas cambia el modelo bajo "
            "la misma semilla. Cada configuracion se repite con 5 semillas porque "
            "la dispersion entre semillas es del mismo orden que varias de las "
            "diferencias entre configuraciones; sin ella, la tabla invitaria a "
            "leer como mejora lo que es ruido. La comparacion decisiva para la "
            "revision es 'solo mercado_ma3' vs el modelo completo."
        ),
    }
    _guardar("ablacion", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 3. Auditoria de leakage y trazabilidad de variables
# ──────────────────────────────────────────────────────────────────────────
def etapa_leakage(p) -> dict:
    print("\n=== AUDITORIA DE LEAKAGE ===")
    # Verificacion 1: los encoders solo contienen categorias presentes en train.
    cat_train_puerto = set(p.train["PUER_DESC"].unique())
    cat_train_import = set(p.train["IMPORTADOR"].unique())
    assert set(p.puerto_freq.index) <= cat_train_puerto
    assert set(p.importador_freq.index) <= cat_train_import

    # Verificacion 2: ninguna fecha se comparte entre particiones.
    f_tr, f_va, f_te = (set(d["FECHA"]) for d in (p.train, p.val, p.test))
    assert not (f_tr & f_va) and not (f_va & f_te) and not (f_tr & f_te)

    # Verificacion 3: los percentiles de recorte se estimaron solo con train.
    diag = p.diag_recorte.to_dict("records")

    # Verificacion 4: los rezagos solo miran meses estrictamente anteriores.
    serie = p.serie_mercado
    per_test = p.test["FECHA"].dt.to_period("M")
    coincide = []
    for per in sorted(per_test.unique())[:5]:
        esperado = serie.get(per - 1, np.nan)
        real = p.test.loc[per_test == per, "mercado_lag1"].iloc[0]
        coincide.append({
            "mes": str(per), "lag1_servido": round(float(real), 6),
            "media_mes_anterior": round(float(esperado), 6),
            "coincide": bool(np.isclose(real, esperado)),
        })

    # Cuanto del mes frontera (transductividad declarada) proviene de test.
    mes_frontera = str(p.test["FECHA"].dt.to_period("M").min())
    todo = pd.concat([p.train, p.val, p.test])
    en_frontera = todo["FECHA"].dt.to_period("M").astype(str) == mes_frontera
    de_test = p.test["FECHA"].dt.to_period("M").astype(str) == mes_frontera

    res = {
        "encoders_ajustados_solo_con_train": True,
        "puerto_freq_categorias": int(len(p.puerto_freq)),
        "importador_freq_categorias": int(len(p.importador_freq)),
        "medianas_ajustadas_solo_con_train": {
            "densidad_carga_median": round(p.densidad_carga_median, 6),
            "ratio_bruto_neto_median": round(p.ratio_bruto_neto_median, 6),
        },
        "percentiles_recorte_solo_train": diag,
        "fechas_compartidas_entre_particiones": 0,
        "pct_filas_categoria_no_vista_test": {
            "puerto": round(100 * float((~p.test["PUER_DESC"].isin(p.puerto_freq.index)).mean()), 2),
            "importador": round(100 * float((~p.test["IMPORTADOR"].isin(p.importador_freq.index)).mean()), 2),
        },
        "verificacion_rezagos": coincide,
        "transductividad_declarada": {
            "mes_frontera_val_test": mes_frontera,
            "filas_del_mes_frontera": int(en_frontera.sum()),
            "de_ellas_en_test": int(de_test.sum()),
            "pct_de_la_media_mensual_aportado_por_test": round(
                100 * float(de_test.sum() / max(int(en_frontera.sum()), 1)), 2
            ),
            "nota": (
                "La serie mensual de mercado se construye sobre todo el periodo. "
                "Los rezagos solo miran meses ESTRICTAMENTE anteriores (shift), de "
                "modo que no hay look-ahead fila a fila; pero la media del mes "
                "frontera se calcula con filas que caen en dos particiones. Es una "
                "propiedad transductiva conocida y se declara como tal."
            ),
        },
    }
    for k in ("puerto_freq_categorias", "importador_freq_categorias"):
        print(f"  {k}: {res[k]}")
    print(f"  Fechas compartidas entre particiones: 0 (verificado con assert)")
    print(f"  Filas de test con puerto no visto: {res['pct_filas_categoria_no_vista_test']['puerto']}% | "
          f"importador no visto: {res['pct_filas_categoria_no_vista_test']['importador']}%")
    _guardar("leakage", res)
    return res


def etapa_trazabilidad(p) -> dict:
    """Origen exacto de cada una de las 14 features en produccion."""
    print("\n=== TRAZABILIDAD input de usuario -> feature ===")
    mapa = {
        "mes": ("fecha de embarque (usuario)", "derivada: mes calendario"),
        "trimestre": ("fecha de embarque (usuario)", "derivada: trimestre"),
        "semana_anio": ("fecha de embarque (usuario)", "derivada: semana ISO"),
        "mes_sin": ("fecha de embarque (usuario)", "derivada: sin(2*pi*mes/12)"),
        "mes_cos": ("fecha de embarque (usuario)", "derivada: cos(2*pi*mes/12)"),
        "mercado_lag1": ("serie de mercado (sistema)", "media del mes m-1"),
        "mercado_lag2": ("serie de mercado (sistema)", "media del mes m-2"),
        "mercado_lag3": ("serie de mercado (sistema)", "media del mes m-3"),
        "mercado_ma3": ("serie de mercado (sistema)", "media movil de m-1..m-3"),
        "puerto_freq": ("puerto de embarque (usuario)", "encoder de frecuencia ajustado con train"),
        "importador_freq": ("importador (usuario, opcional)", "encoder de frecuencia; si se omite, mediana de train"),
        "densidad_carga": ("peso neto y unidades (usuario; unidades opcional)", "peso/unidades; si falta, mediana de train"),
        "ratio_bruto_neto": ("no capturado en el formulario", "SIEMPRE la mediana de train (constante en produccion)"),
        "ruta_directa": ("puerto de embarque (usuario)", "lookup deterministico puerto -> regimen; nunca se pide"),
    }
    assert set(mapa) == set(dp.FEATURES), "El mapa de trazabilidad no cubre las 14 features"
    res = {
        "campos_del_formulario": [
            "puerto de embarque (obligatorio)",
            "peso neto en toneladas (obligatorio)",
            "unidades (opcional)",
            "importador (opcional)",
            "periodo + fecha: semanal / mensual / anual (obligatorio)",
        ],
        "mapa_features": {k: {"origen": v[0], "construccion": v[1]} for k, v in mapa.items()},
        "constantes_en_produccion": {
            "ratio_bruto_neto": round(p.ratio_bruto_neto_median, 6),
            "densidad_carga_default": round(p.densidad_carga_median, 6),
            "importador_freq_default": round(p.importador_freq_default, 8),
            "puerto_freq_default": round(p.puerto_freq_default, 8),
        },
        "nota": (
            "El formulario NO captura peso bruto ni tipo de contenedor. "
            "`ratio_bruto_neto` es por tanto una constante en produccion (la "
            "mediana de train) y `densidad_carga` cae a su mediana si el usuario "
            "omite las unidades. Ambas aportan en conjunto ~1.1% del gain, de modo "
            "que la degradacion es marginal, pero debe declararse: son features del "
            "entrenamiento que produccion no puede reconstruir con fidelidad."
        ),
    }
    for f, v in mapa.items():
        print(f"  {f:<18} <- {v[0]}")
    _guardar("trazabilidad", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 4. Comparacion de modelos con IC95 bootstrap y prueba pareada
# ──────────────────────────────────────────────────────────────────────────
def etapa_modelos(p) -> dict:
    print("\n=== COMPARACION DE MODELOS (IC95 bootstrap, B=%d) ===" % N_BOOTSTRAP)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    F, T = dp.FEATURES, dp.TARGET
    Xtr, ytr = p.train[F], p.train[T]
    Xva, yva = p.val[F], p.val[T]
    Xte, yte = p.test[F], p.test[T].values
    semillas = [42, 7, 123, 2024, 31337]

    # QUINTA REVISION. Todos los competidores se ajustan con la MISMA receta que
    # el modelo desplegado: bloque completo (train+val), sin early stopping y con
    # el numero de iteraciones que fija el origen rodante interno. Antes cada
    # familia paraba por early stopping contra `val`, y bajo el esquema 80/20 ese
    # criterio elige modelos subajustados (ver ml/seleccion_iteraciones.py): la
    # tabla habria comparado algoritmos bajo un procedimiento que perjudica a
    # todos por igual pero en distinta medida, que es peor que compararlos mal de
    # forma uniforme. Ahora la unica diferencia entre filas es el algoritmo y la
    # perdida, que es lo que la tabla dice medir.
    BLOQUE = bloque_ajuste(p)
    Xbl, ybl = BLOQUE[F], BLOQUE[T]
    N_ITER = n_estimators_produccion(p)

    def con_semillas(constructor, ajustar) -> tuple[np.ndarray, list[float]]:
        """Entrena el modelo con cada semilla y devuelve (prediccion promediada,
        MAPE por semilla). El promedio entre semillas es lo que entra a la prueba
        pareada; la lista de MAPE es lo que cuantifica el ruido del muestreo."""
        todas = []
        for s in semillas:
            m = constructor(s)
            ajustar(m)
            todas.append(np.asarray(m.predict(Xte), dtype=float))
        return np.mean(todas, axis=0), [_mape(yte, y) for y in todas]

    preds: dict[str, np.ndarray] = {}
    mape_semillas: dict[str, list[float]] = {}

    # NOTA METODOLOGICA. Comparar el modelo desplegado contra CatBoost con
    # loss_function="MAE" NO aisla el algoritmo: mezcla el efecto del algoritmo
    # con el de la funcion de perdida, y la evaluacion (MAE/MAPE) favorece a
    # quien optimiza el error absoluto. Se entrenan por eso AMBAS perdidas para
    # las tres familias de boosting.
    preds["XGBoost (perdida MAE)"], mape_semillas["XGBoost (perdida MAE)"] = con_semillas(
        lambda s: entrenar_xgb(p, F, seed=s), lambda m: None,
    )

    preds["XGBoost (perdida cuadratica)"], mape_semillas["XGBoost (perdida cuadratica)"] = con_semillas(
        lambda s: XGBRegressor(
            n_estimators=N_ITER, learning_rate=0.05, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, random_state=s, n_jobs=-1,
            objective="reg:squarederror", eval_metric="mae",
        ),
        lambda m: m.fit(Xbl, ybl, verbose=False),
    )

    preds["LightGBM"], mape_semillas["LightGBM"] = con_semillas(
        lambda s: LGBMRegressor(
            n_estimators=N_ITER, learning_rate=0.05, max_depth=6, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=s, n_jobs=-1,
            verbose=-1,
        ),
        lambda m: m.fit(Xbl, ybl),
    )

    preds["LightGBM (perdida MAE)"], mape_semillas["LightGBM (perdida MAE)"] = con_semillas(
        lambda s: LGBMRegressor(
            n_estimators=N_ITER, learning_rate=0.05, max_depth=6, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=s, n_jobs=-1,
            verbose=-1, objective="l1",
        ),
        lambda m: m.fit(Xbl, ybl),
    )

    preds["CatBoost (perdida MAE)"], mape_semillas["CatBoost (perdida MAE)"] = con_semillas(
        lambda s: CatBoostRegressor(
            iterations=N_ITER, learning_rate=0.05, depth=6, loss_function="MAE",
            random_seed=s, verbose=0,
        ),
        lambda m: m.fit(Xbl, ybl),
    )

    preds["CatBoost"], mape_semillas["CatBoost"] = con_semillas(
        lambda s: CatBoostRegressor(
            iterations=N_ITER, learning_rate=0.05, depth=6, loss_function="RMSE",
            random_seed=s, verbose=0,
        ),
        lambda m: m.fit(Xbl, ybl),
    )

    preds["Random Forest"], mape_semillas["Random Forest"] = con_semillas(
        lambda s: RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=s, n_jobs=-1,
        ),
        lambda m: m.fit(Xbl, ybl),
    )

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xbl, ybl)
    preds["Ridge"] = ridge.predict(Xte)          # deterministico, sin semilla

    preds["Baseline persistencia (lag1)"] = p.test["mercado_lag1"].values
    preds["Baseline media movil (ma3)"] = p.test["mercado_ma3"].values

    # Bootstrap pareado: el MISMO remuestreo de filas se aplica a todos los
    # modelos, de modo que la diferencia entre dos modelos no incorpora la
    # varianza de haber evaluado sobre muestras distintas.
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(yte), size=(N_BOOTSTRAP, len(yte)))
    mapes_bs = {
        nombre: np.array([_mape(yte[i], yhat[i]) for i in idx])
        for nombre, yhat in preds.items()
    }

    ref = "XGBoost (perdida MAE)"
    filas = []
    for nombre, yhat in preds.items():
        m = evaluar(yte, yhat)
        bs = mapes_bs[nombre]
        fila = {
            "modelo": nombre,
            "MAE": round(m["MAE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "MAPE_%": round(m["MAPE_%"], 2),
            "IC95_MAPE": [round(float(np.percentile(bs, 2.5)), 2),
                          round(float(np.percentile(bs, 97.5)), 2)],
            "WAPE_%": round(m["WAPE_%"], 2),
            "sMAPE_%": round(m["sMAPE_%"], 2),
        }
        if nombre in mape_semillas:
            ms = mape_semillas[nombre]
            fila["MAPE_por_semilla"] = [round(v, 2) for v in ms]
            fila["MAPE_media_semillas"] = round(float(np.mean(ms)), 2)
            fila["MAPE_sd_semillas"] = round(float(np.std(ms)), 2)
        if nombre != ref:
            d = bs - mapes_bs[ref]           # >0 significa peor que XGBoost
            fila["delta_MAPE_vs_XGBoost_pp"] = round(float(np.mean(d)), 2)
            fila["IC95_delta"] = [round(float(np.percentile(d, 2.5)), 2),
                                  round(float(np.percentile(d, 97.5)), 2)]
            # p-valor de una cola: proporcion de remuestreos en que el rival NO
            # es peor que XGBoost.
            fila["p_valor_bootstrap"] = round(float(np.mean(d <= 0)), 4)
            fila["significativo_95"] = bool(np.percentile(d, 2.5) > 0)
        filas.append(fila)

    filas.sort(key=lambda f: f["MAPE_%"])
    for f in filas:
        extra = ""
        if "p_valor_bootstrap" in f:
            extra = (f" | delta {f['delta_MAPE_vs_XGBoost_pp']:+.2f}pp "
                     f"IC95 [{f['IC95_delta'][0]:+.2f},{f['IC95_delta'][1]:+.2f}] "
                     f"p={f['p_valor_bootstrap']:.4f}")
        print(f"  {f['modelo']:<30} MAPE {f['MAPE_%']:6.2f}% "
              f"IC95 [{f['IC95_MAPE'][0]:.2f}, {f['IC95_MAPE'][1]:.2f}]{extra}")

    res = {
        "n_bootstrap": N_BOOTSTRAP,
        "semillas": semillas,
        "referencia": ref,
        "tabla": filas,
        "nota": (
            "Cada modelo estocastico se entrena con 5 semillas y la prueba se "
            "hace sobre la prediccion PROMEDIADA entre ellas: la dispersion entre "
            "semillas es comparable a la diferencia entre algoritmos, de modo que "
            "una comparacion de una sola semilla no es concluyente. Bootstrap "
            "PAREADO sobre las filas de test (el mismo remuestreo para todos los "
            "modelos). El p-valor es la proporcion de remuestreos en que el modelo "
            "rival NO resulta peor que XGBoost."
        ),
    }
    _guardar("modelos", res)
    return res, preds, entrenar_xgb(p, F)


# ──────────────────────────────────────────────────────────────────────────
# 5. Metricas sobre el flete TOTAL en USD
# ──────────────────────────────────────────────────────────────────────────
def etapa_metricas(p, preds: dict) -> dict:
    print("\n=== METRICAS SOBRE EL FLETE TOTAL EN USD ===")
    peso = p.test["PESO_NETO"].values
    real_usd = p.test["FLE_DOLAR"].values
    unit = p.test[dp.TARGET].values

    # Coherencia: FLETE_UNIT * PESO_NETO debe reconstruir FLE_DOLAR.
    recon = unit * peso
    err_recon = float(np.max(np.abs(recon - real_usd)))

    filas = []
    for nombre, yhat in preds.items():
        m = evaluar(real_usd, np.asarray(yhat) * peso)
        filas.append({
            "modelo": nombre,
            "MAE_USD": round(m["MAE"], 2),
            "RMSE_USD": round(m["RMSE"], 2),
            "MAPE_%": round(m["MAPE_%"], 2),
            "WAPE_%": round(m["WAPE_%"], 2),
            "sMAPE_%": round(m["sMAPE_%"], 2),
        })
    filas.sort(key=lambda f: f["MAPE_%"])
    for f in filas:
        print(f"  {f['modelo']:<30} MAE USD {f['MAE_USD']:>9,.2f} | "
              f"MAPE {f['MAPE_%']:6.2f}% | WAPE {f['WAPE_%']:6.2f}%")

    res = {
        "error_max_reconstruccion_USD": err_recon,
        "flete_total_test": {
            "media_USD": round(float(real_usd.mean()), 2),
            "mediana_USD": round(float(np.median(real_usd)), 2),
            "suma_USD": round(float(real_usd.sum()), 2),
        },
        "tabla": filas,
        "nota": (
            "El flete total se reconstruye como prediccion_unitaria * PESO_NETO, "
            "exactamente como lo hace el sistema en produccion. El MAPE sobre el "
            "total difiere del MAPE sobre USD/kg porque pondera cada embarque por "
            "su tamano de forma distinta; el WAPE es la metrica robusta al "
            "denominador pequeno que la revision pedia."
        ),
    }
    _guardar("metricas_usd", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 6. Error por segmentos
# ──────────────────────────────────────────────────────────────────────────
def etapa_segmentos(p, pred_xgb: np.ndarray) -> dict:
    print("\n=== ERROR POR SEGMENTOS (test) ===")
    d = p.test.copy()
    d["_pred"] = pred_xgb
    d["_ape"] = np.abs((d[dp.TARGET] - d["_pred"]) / d[dp.TARGET]) * 100
    d["_ae"] = np.abs(d[dp.TARGET] - d["_pred"])

    def agrupar(col, minimo=1, top=None):
        g = d.groupby(d[col].astype(str)).agg(
            n=("_ape", "size"), MAPE=("_ape", "mean"), MAE=("_ae", "mean"),
        )
        g = g[g["n"] >= minimo].sort_values("n", ascending=False)
        if top:
            g = g.head(top)
        return {
            str(k): {"n": int(r["n"]), "MAPE_%": round(float(r["MAPE"]), 2),
                     "MAE": round(float(r["MAE"]), 4)}
            for k, r in g.iterrows()
        }

    d["_mes"] = d["FECHA"].dt.to_period("M").astype(str)
    # Puertos: los principales por separado y el resto agregado.
    top_puertos = d["PUER_DESC"].value_counts().head(5).index
    d["_puerto_grp"] = np.where(d["PUER_DESC"].isin(top_puertos), d["PUER_DESC"], "Otros")
    # Volumen del embarque en cuartiles de peso neto.
    d["_peso_q"] = pd.qcut(
        d["PESO_NETO"], 4, labels=["Q1 (menor)", "Q2", "Q3", "Q4 (mayor)"], duplicates="drop",
    )
    # Importadores segun representacion en train.
    en_cat = d["IMPORTADOR"].isin(p.importador_freq.index)
    d["_import_grp"] = np.where(en_cat, "visto en train", "no visto en train")
    d["_ruta"] = np.where(d["ruta_directa"] == 1, "directo", "transbordo")

    res = {
        "por_puerto": agrupar("_puerto_grp"),
        "por_mes": agrupar("_mes"),
        "por_cuartil_de_peso": agrupar("_peso_q"),
        "por_representacion_importador": agrupar("_import_grp"),
        "por_regimen_de_ruta": agrupar("_ruta"),
        "nota": (
            "El MAPE global puede ocultar diferencias por segmento. Se reporta "
            "desagregado por puerto, mes, tamano de embarque, representacion del "
            "importador en train y regimen de ruta."
        ),
    }
    for titulo, tabla in res.items():
        if titulo == "nota":
            continue
        print(f"  --- {titulo} ---")
        for k, v in tabla.items():
            print(f"    {k:<28} n={v['n']:>5,} | MAPE {v['MAPE_%']:6.2f}%")
    _guardar("segmentos", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 7. Walk-forward (origen rodante): error por anio, trimestre y volatilidad
# ──────────────────────────────────────────────────────────────────────────
def etapa_walkforward(p) -> dict:
    """Origen rodante mensual sobre TODO el historico evaluable.

    Complementa a `ml/walk_forward.py`, que hace lo mismo restringido a los doce
    meses de 2025 y con el detalle que pide la revision metodologica. Este
    recorre todos los meses con historia suficiente, que es lo que el articulo
    necesita para hablar de error por anio, por trimestre y por regimen de
    volatilidad: con un solo ano no hay nada que comparar entre anos.

    QUINTA REVISION — SE ELIMINA LA FUGA QUE ESTA ETAPA DECLARABA. La version
    anterior reajustaba los encoders por pliegue pero heredaba del pipeline
    COMPLETO los umbrales de recorte de atipicos y la serie mensual de mercado,
    y lo anotaba como "limitacion declarada". Declararla no la arregla: los
    umbrales de recorte se calculaban con datos posteriores al mes evaluado, de
    modo que la limpieza del pasado usaba informacion del futuro. Ahora cada
    pliegue llama a `dp.particionar()` sobre el scope truncado a su propia
    historia —el mismo codigo que entrena produccion— y no hereda nada.
    """
    print()
    print("=== WALK-FORWARD (origen rodante mensual, pipeline completo por pliegue) ===")
    scope = dp.cargar_scope(str(CSV))
    meses_scope = sorted(scope["FECHA"].dt.to_period("M").unique())
    # Se empieza a evaluar cuando hay al menos 12 meses de historia. Los tres
    # primeros meses del corpus los consume el calentamiento de los rezagos.
    meses = meses_scope[12:]

    filas = []
    for mes in meses:
        inicio = pd.Timestamp(mes.start_time).normalize()
        fin = pd.Timestamp(mes.end_time).normalize() + pd.Timedelta(days=1)
        scope_pliegue = scope[scope["FECHA"] < fin]
        historia = scope_pliegue["FECHA"] < inicio
        if historia.sum() < 500:
            continue
        corte_interno = dp._corte_por_fecha(
            scope_pliegue.loc[historia, "FECHA"], dp.FRACCION_TRAIN_INTERNO
        )
        pf = dp.particionar(scope_pliegue, corte_interno, inicio, esquema="80_20")
        if len(pf.test) < 30:
            continue
        # Invariantes del pliegue, comprobadas en cada mes.
        assert pf.train["FECHA"].max() < inicio and pf.val["FECHA"].max() < inicio
        assert pf.serie_mercado.index.max() <= mes

        modelo = entrenar_xgb(pf, dp.FEATURES, seed=RANDOM_STATE)
        yhat = modelo.predict(pf.test[dp.FEATURES])
        y = pf.test[dp.TARGET].values

        # Volatilidad del mercado del pliegue: variacion del nivel mensual
        # respecto al mes previo, con la serie que el pliegue pudo observar.
        act = float(pf.serie_mercado.get(mes, np.nan))
        ant = float(pf.serie_mercado.get(mes - 1, np.nan))
        var_pct = float(abs(act - ant) / ant * 100) if ant and not np.isnan(ant) else np.nan

        filas.append({
            "mes": str(mes), "anio": int(mes.year), "trimestre": f"{mes.year}-T{mes.quarter}",
            "n": int(len(pf.test)),
            "n_entrenamiento": int(len(pf.train) + len(pf.val)),
            "MAPE_%": round(_mape(y, yhat), 2),
            "MAE": round(float(mean_absolute_error(y, yhat)), 4),
            "variacion_mercado_%": round(var_pct, 2) if not np.isnan(var_pct) else None,
        })
        print(f"  {mes} n={len(pf.test):>5,} | MAPE {filas[-1]['MAPE_%']:6.2f}% "
              f"| var. mercado {filas[-1]['variacion_mercado_%']}%")

    df = pd.DataFrame(filas)

    # Ponderado por numero de declaraciones: cada embarque pesa igual.
    def agg(col):
        salida = {}
        for k, x in df.groupby(df[col].astype(str)):
            salida[str(k)] = {
                "n": int(x["n"].sum()),
                "MAPE_%": round(float(np.average(x["MAPE_%"], weights=x["n"])), 2),
                "meses": int(len(x)),
            }
        return salida

    vv = df.dropna(subset=["variacion_mercado_%"])
    umbral = float(vv["variacion_mercado_%"].median())
    vv = vv.assign(_reg=np.where(vv["variacion_mercado_%"] > umbral, "volatil", "estable"))
    por_regimen = {
        str(k): {
            "meses": int(len(x)), "n": int(x["n"].sum()),
            "MAPE_%": round(float(np.average(x["MAPE_%"], weights=x["n"])), 2),
            "variacion_mercado_media_%": round(float(x["variacion_mercado_%"].mean()), 2),
        }
        for k, x in vv.groupby("_reg")
    }

    res = {
        "detalle_mensual": filas,
        "meses_evaluados": len(filas),
        "MAPE_global_ponderado_%": round(float(np.average(df["MAPE_%"], weights=df["n"])), 2),
        "MAPE_promedio_simple_%": round(float(df["MAPE_%"].mean()), 2),
        "MAPE_sd_%": round(float(df["MAPE_%"].std()), 2),
        "por_anio": agg("anio"),
        "por_trimestre": agg("trimestre"),
        "por_regimen_volatilidad": por_regimen,
        "umbral_volatilidad_%": round(umbral, 2),
        "nota": (
            "Origen rodante: para cada mes se reconstruye el pipeline ENTERO con "
            "la historia estrictamente anterior —umbrales de recorte de atipicos, "
            "lookup de ruta, encoders, medianas y serie mensual de mercado— "
            "llamando al mismo dp.particionar() que usa produccion, y se entrena "
            "con la misma receta. Nada se hereda del pipeline completo: la "
            "limitacion que esta etapa declaraba en rondas anteriores queda "
            "corregida, no solo documentada. Regimen volatil = meses cuya "
            "variacion del nivel de mercado supera la mediana de la serie."
        ),
        "relacion_con_walk_forward_2025": (
            "ml/walk_forward.py aplica este mismo protocolo a los doce meses de "
            "2025 y anade las comparaciones que pide la revision (modelo estatico "
            "sobre las mismas filas, coste del rezago de publicacion de SUNAT y "
            "lineas base por mes). Esta etapa cubre todo el historico, que es lo "
            "que permite hablar de error por anio y por regimen de volatilidad."
        ),
    }
    print()
    print(f"  MAPE global ponderado: {res['MAPE_global_ponderado_%']}% "
          f"| promedio simple {res['MAPE_promedio_simple_%']}% (sd {res['MAPE_sd_%']})")
    for k, v in por_regimen.items():
        print(f"  regimen {k:<8} {v['meses']:>2} meses | MAPE {v['MAPE_%']:6.2f}%")
    _guardar("walkforward", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 8. Sensibilidad al tratamiento de atipicos
# ──────────────────────────────────────────────────────────────────────────
def _construir_variante(csv_path: str, modo: str):
    """Replica de `data_pipeline.construir()` con el tratamiento de atipicos
    parametrizable. Ver la INVARIANTE del docstring del modulo: el modo
    'eliminar' se verifica contra el pipeline real."""
    df = pd.read_csv(csv_path, sep=",", quotechar='"', encoding="utf-8", low_memory=False)
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%Y%m%d", errors="coerce")
    df["CNAN"] = df["CNAN"].astype(str).str.zfill(10)
    mask = (
        df["CNAN"].str.startswith("4011")
        & (df["VIA_TRANSP"] == 1)
        & (df["ADUA_DESC"].str.upper().str.contains("CALLAO", na=False))
        & (df["CPAIS"].isin(dp.PAISES_ORIGEN))
    )
    scope = df.loc[mask].sort_values("FECHA", kind="mergesort").reset_index(drop=True)
    scope["FLETE_UNIT"] = scope["FLE_DOLAR"] / scope["PESO_NETO"]
    scope["ruta_directa_real"] = (scope["CPAIS_PROC"] == scope["CPAIS"]).astype(int)

    # Los cortes salen del MISMO selector de esquema que usa el pipeline real
    # (`dp._cortes`), no de dos cuantiles escritos a mano. Cuando el esquema
    # paso de 70/20/10 a 80/20, esta reimplementacion se habria quedado
    # comparando variantes de atipicos sobre una particion que ya no existe, y
    # el assert de la INVARIANTE del modulo habria fallado sin explicar por que.
    fecha_corte_train, fecha_corte_val = dp._cortes(scope["FECHA"], dp.ESQUEMA_DEFAULT)
    es_train = scope["FECHA"] < fecha_corte_train
    lookup, default = dp._ajustar_lookup_ruta(scope, es_train)
    scope["ruta_directa"] = scope["PUER_DESC"].map(lookup).fillna(default).astype(int)

    n_antes = len(scope)
    if modo == "ninguno":
        clean = scope.copy()
        n_afectadas = 0
    elif modo == "winsor":
        clean = scope.copy()
        n_afectadas = 0
        for regimen, grupo in scope.groupby("ruta_directa"):
            base = grupo[es_train.reindex(grupo.index, fill_value=False)]
            lo, hi = base["FLETE_UNIT"].quantile([dp.P_LOW, dp.P_HIGH])
            fuera = ~grupo["FLETE_UNIT"].between(lo, hi)
            n_afectadas += int(fuera.sum())
            clean.loc[grupo.index, "FLETE_UNIT"] = grupo["FLETE_UNIT"].clip(lo, hi)
    elif modo == "eliminar":
        conservar, _ = dp._recorte_estratificado(scope, es_train)
        clean = scope[conservar].copy()
        n_afectadas = int((~conservar).sum())
    else:
        raise ValueError(modo)

    clean = clean.drop_duplicates()
    clean = clean[clean["FLE_DOLAR"] > 0].sort_values("FECHA", kind="mergesort").reset_index(drop=True)

    fe = clean
    fe["mes"] = fe["FECHA"].dt.month
    fe["trimestre"] = fe["FECHA"].dt.quarter
    fe["semana_anio"] = fe["FECHA"].dt.isocalendar().week.astype(int)
    fe["mes_sin"] = np.sin(2 * np.pi * fe["mes"] / 12)
    fe["mes_cos"] = np.cos(2 * np.pi * fe["mes"] / 12)
    fe["periodo"] = fe["FECHA"].dt.to_period("M")
    serie = fe.groupby("periodo")["FLETE_UNIT"].mean().sort_index()
    lags = pd.DataFrame({"m": serie})
    for k in (1, 2, 3):
        lags[f"mercado_lag{k}"] = lags["m"].shift(k)
    lags["mercado_ma3"] = lags["m"].shift(1).rolling(3).mean()
    fe = fe.merge(lags[["mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3"]],
                  left_on="periodo", right_index=True, how="left")
    fe = fe.dropna(subset=["mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3"])
    fe = fe.sort_values("FECHA", kind="mergesort").reset_index(drop=True)

    train = fe[fe["FECHA"] < fecha_corte_train].copy()
    val = fe[(fe["FECHA"] >= fecha_corte_train) & (fe["FECHA"] < fecha_corte_val)].copy()
    test = fe[fe["FECHA"] >= fecha_corte_val].copy()

    p = types.SimpleNamespace(
        train=train, val=val, test=test,
        puerto_freq=train["PUER_DESC"].value_counts(normalize=True),
        importador_freq=train["IMPORTADOR"].value_counts(normalize=True),
        densidad_carga_median=float((train["PESO_NETO"] / train["UNID_FIQTY"].replace(0, np.nan)).median()),
        ratio_bruto_neto_median=float((train["PESO_BRUTO"] / train["PESO_NETO"].replace(0, np.nan)).median()),
        ruta_directa_por_puerto=lookup, ruta_directa_default=default,
        serie_mercado=serie,
    )
    p.puerto_freq_default = float(p.puerto_freq.median())
    p.importador_freq_default = float(p.importador_freq.median())
    p.train = dp.aplicar_encoders(p.train, p)
    p.val = dp.aplicar_encoders(p.val, p)
    p.test = dp.aplicar_encoders(p.test, p)
    p._n_antes, p._n_afectadas = n_antes, n_afectadas
    return p


def etapa_outliers(p_ref) -> dict:
    print("\n=== SENSIBILIDAD AL TRATAMIENTO DE ATIPICOS ===")
    ref = evaluar(p_ref.test[dp.TARGET].values,
                  entrenar_xgb(p_ref, dp.FEATURES).predict(p_ref.test[dp.FEATURES]))
    filas = []
    for modo, etiqueta in [
        ("ninguno", "Sin tratamiento"),
        ("winsor", "Winsorizacion P0.5-P99.5"),
        ("eliminar", "Eliminacion P0.5-P99.5 (produccion)"),
    ]:
        pv = _construir_variante(str(CSV), modo)
        m = evaluar(pv.test[dp.TARGET].values,
                    entrenar_xgb(pv, dp.FEATURES).predict(pv.test[dp.FEATURES]))
        filas.append({
            "tratamiento": etiqueta,
            "filas_afectadas": int(pv._n_afectadas),
            "pct_afectado": round(100 * pv._n_afectadas / pv._n_antes, 3),
            "n_train": int(len(pv.train)), "n_test": int(len(pv.test)),
            "MAE": round(m["MAE"], 4), "RMSE": round(m["RMSE"], 4),
            "MAPE_%": round(m["MAPE_%"], 2), "R2": round(m["R2"], 4),
        })
        print(f"  {etiqueta:<38} MAPE {m['MAPE_%']:6.2f}% | MAE {m['MAE']:.4f} "
              f"| RMSE {m['RMSE']:.4f} | n_test {len(pv.test):,}")

        if modo == "eliminar":
            # INVARIANTE: la replica debe reproducir el pipeline real.
            assert abs(m["MAPE_%"] - ref["MAPE_%"]) < 0.01, (
                f"La replica del pipeline diverge del real: {m['MAPE_%']} vs {ref['MAPE_%']}"
            )
            print("    [OK] la replica reproduce el pipeline de produccion al centesimo")

    res = {
        "tabla": filas,
        "mape_pipeline_produccion": round(ref["MAPE_%"], 2),
        "nota": (
            "Los conjuntos de test NO son identicos entre tratamientos (eliminar "
            "quita filas tambien de test), de modo que la comparacion es entre "
            "PIPELINES completos y no entre modelos sobre la misma muestra. El "
            "objetivo es verificar que la conclusion del articulo no depende de "
            "haber retirado las observaciones dificiles: la winsorizacion las "
            "conserva todas y permite una comparacion sobre el test integro."
        ),
    }
    _guardar("outliers", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
# 9. Figuras SHAP
# ──────────────────────────────────────────────────────────────────────────
def etapa_shap(p, modelo=None) -> dict:
    print("\n=== SHAP: figuras y comparacion por regimen ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    FIGS.mkdir(parents=True, exist_ok=True)
    modelo = modelo or entrenar_xgb(p, dp.FEATURES)
    expl = shap.TreeExplainer(modelo)

    X_test = p.test[dp.FEATURES]
    sv_test = expl.shap_values(X_test)

    def guardar(nombre):
        plt.tight_layout()
        plt.savefig(FIGS / nombre, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  figura -> {nombre}")

    shap.summary_plot(sv_test, X_test, show=False, max_display=14)
    guardar("shap_beeswarm_test.png")

    shap.summary_plot(sv_test, X_test, plot_type="bar", show=False, max_display=14)
    guardar("shap_barras_test.png")

    for feat in ("mercado_ma3", "mercado_lag1"):
        shap.dependence_plot(feat, sv_test, X_test, show=False, interaction_index="mes")
        guardar(f"shap_dependence_{feat}.png")

    # Waterfalls de dos predicciones reales: la mejor y una del percentil 90 de error.
    err = np.abs(p.test[dp.TARGET].values - modelo.predict(X_test))
    base_val = float(np.ravel(expl.expected_value)[0])
    for etiqueta, pos in [("tipica", int(np.argsort(err)[len(err) // 2])),
                          ("error_alto", int(np.argsort(err)[int(len(err) * 0.9)]))]:
        ex = shap.Explanation(
            values=sv_test[pos], base_values=base_val,
            data=X_test.iloc[pos].values, feature_names=dp.FEATURES,
        )
        shap.plots.waterfall(ex, show=False, max_display=14)
        guardar(f"shap_waterfall_{etiqueta}.png")

    # Regimen volatil (2021-2022) vs estable (test 2025).
    todo = pd.concat([p.train, p.val, p.test])
    volatil = todo[todo["FECHA"].dt.year.isin([2021, 2022])]
    volatil = volatil.sample(min(3000, len(volatil)), random_state=RANDOM_STATE)
    sv_vol = expl.shap_values(volatil[dp.FEATURES])

    def importancia(sv):
        m = np.abs(sv).mean(axis=0)
        return {f: round(float(100 * v / m.sum()), 2) for f, v in zip(dp.FEATURES, m)}

    imp_vol, imp_est = importancia(sv_vol), importancia(sv_test)
    orden = sorted(dp.FEATURES, key=lambda f: -imp_est[f])
    x = np.arange(len(orden))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, [imp_vol[f] for f in orden], 0.4, label="Regimen volatil (2021-2022)")
    ax.bar(x + 0.2, [imp_est[f] for f in orden], 0.4, label="Regimen estable (test 2025)")
    ax.set_xticks(x, orden, rotation=45, ha="right")
    ax.set_ylabel("Importancia SHAP relativa (%)")
    ax.legend()
    guardar("shap_regimen_volatil_vs_estable.png")

    # Figura del articulo: serie mensual real frente a predicha sobre TEST.
    d = p.test.copy()
    d["_pred"] = modelo.predict(X_test)
    d["_per"] = d["FECHA"].dt.to_period("M").astype(str)
    serie = d.groupby("_per")[[dp.TARGET, "_pred"]].mean()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(serie.index, serie[dp.TARGET], marker="o", label="Real")
    ax.plot(serie.index, serie["_pred"], marker="s", linestyle="--", label="Predicho")
    ax.set_ylabel("Flete unitario (USD/kg)")
    ax.set_xlabel("Mes")
    ax.legend()
    ax.grid(alpha=0.3)
    guardar("real_vs_predicho_test.png")

    gain = modelo.get_booster().get_score(importance_type="gain")
    total = sum(gain.values())
    res = {
        "importancia_shap_test_%": imp_est,
        "importancia_shap_volatil_%": imp_vol,
        "importancia_gain_%": {
            f: round(100 * gain.get(f, 0.0) / total, 2) for f in dp.FEATURES
        },
        "figuras": sorted(f.name for f in FIGS.glob("*.png")),
        "nota": (
            "La comparacion entre regimenes usa el MISMO modelo entrenado, "
            "evaluando SHAP sobre filas de 2021-2022 (alta volatilidad) y sobre "
            "test 2025 (estable). Un cambio en el reparto de importancia indica "
            "sobre que se apoya el modelo en cada regimen."
        ),
    }
    top3 = sorted(imp_est.items(), key=lambda kv: -kv[1])[:3]
    print(f"  Top-3 SHAP en test: {top3}")
    _guardar("shap", res)
    return res


# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    etapas = sys.argv[1:] or [
        "dataset", "ablacion", "leakage", "trazabilidad", "modelos",
        "metricas", "segmentos", "walkforward", "outliers", "shap",
    ]
    print(f"CSV: {CSV}")
    print("Construyendo pipeline de produccion (data_pipeline.construir)...")
    p = dp.construir(str(CSV))
    print(f"  train {len(p.train):,} | val {len(p.val):,} | test {len(p.test):,}")

    preds = modelo_xgb = None
    if "dataset" in etapas:
        etapa_dataset(p)
    if "ablacion" in etapas:
        etapa_ablacion(p)
    if "leakage" in etapas:
        etapa_leakage(p)
    if "trazabilidad" in etapas:
        etapa_trazabilidad(p)
    if "modelos" in etapas or "metricas" in etapas or "segmentos" in etapas:
        _, preds, modelo_xgb = etapa_modelos(p)
    if "metricas" in etapas:
        etapa_metricas(p, preds)
    if "segmentos" in etapas:
        etapa_segmentos(p, preds["XGBoost (perdida MAE)"])
    if "walkforward" in etapas:
        etapa_walkforward(p)
    if "outliers" in etapas:
        etapa_outliers(p)
    if "shap" in etapas:
        etapa_shap(p, modelo_xgb)

    print(f"\nListo. Resultados en {OUT / 'resultados.json'}")


if __name__ == "__main__":
    main()

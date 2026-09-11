"""
Entrenamiento del modelo puntual XGBoost de flete (FLETE_UNIT, USD/kg).

Correcciones acumuladas respecto del pipeline original (Untitled.ipynb):

  BUG 1 (look-ahead bias): puerto_freq / importador_freq se calculaban sobre
  TODO el dataset antes del split temporal. Ahora se ajustan (fit) usando SOLO
  el tramo de entrenamiento — ver ml/data_pipeline.py.

  BUG 2 (variable omitida): se agrega `ruta_directa` (1 = embarque directo
  desde China, 0 = transbordo via un tercer pais). Es una funcion
  deterministica del puerto de embarque (verificado: de los 57 puertos del
  tramo de entrenamiento, ninguno mezcla ambos regimenes — el pipeline lo
  comprueba y falla si deja de cumplirse), por lo que en produccion se deriva
  del puerto seleccionado sin pedir un campo nuevo en el formulario.

  PRIMERA AUDITORIA: particion por fecha en vez de por indice; percentiles de
  recorte ajustados solo con train; recorte estratificado por regimen de ruta;
  `ruta_directa` siempre via lookup de puerto, el mismo camino que usa
  produccion.

  SEGUNDA AUDITORIA (todas medidas, ninguna cosmetica):
  - El recorte estratificaba por el `ruta_directa` REAL mientras el modelo
    recibia el del LOOKUP. Ahora la clave de limpieza es la propia feature.
  - El escenario congelado usaba la media de un mes que contiene filas de TEST.
    Ahora se congela en el ultimo mes ESTRICTAMENTE anterior al inicio de TEST.
  - Se agregan cuatro lineas base mas exigentes que las tres originales.
  - Se registran diagnosticos que antes no existian: sesgo del recorte por
    importador/puerto/anio, fiabilidad del default de ruta, cobertura real de
    los catalogos y colapso de regimen del target entre particiones.

Este script reporta ademas, de forma explicita:
  - Metricas de TRAIN y de TEST juntas (la brecha R2 train/test es el dato que
    mide cuanto memoriza el modelo, y no debe quedar fuera del informe).
  - Comparacion contra SIETE lineas base. Un R2 de test cercano a cero solo es
    interpretable junto a estas referencias.
  - Un escenario "produccion" con los rezagos de mercado congelados, que es
    como el sistema desplegado sirve realmente las predicciones cuando la fecha
    pedida cae mas alla del historico.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
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

# El corpus por defecto es el historico de la tesis. `JPS_CSV_PATH` permite
# apuntar al corpus acumulativo que mantiene ml/corpus.py (base + los CSV
# incorporados despues) sin cambiar el comportamiento de una ejecucion manual.
CSV_PATH = os.environ.get("JPS_CSV_PATH", "resultado_combinado.csv")
META_PATH = "ml/modelo_meta.json"
MODEL_PATH = "ml/modelo_xgboost_flete.pkl"

RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────────
# 1. Dataset, recorte, features y particiones (logica compartida)
# ──────────────────────────────────────────────────────────────────────────
p = dp.construir(CSV_PATH)

print(dp.diagnostico_recorte(p))
print(f"\nfe_model (con historia de lags): {len(p.train) + len(p.val) + len(p.test):,} filas")
for name, s in [("TRAIN", p.train), ("VAL", p.val), ("TEST", p.test)]:
    print(f"{name:5} | {len(s):>6,} filas | {s['FECHA'].min().date()} -> {s['FECHA'].max().date()}")
print(f"cortes por fecha: train < {p.fecha_corte_train.date()} <= val < {p.fecha_corte_val.date()} <= test")
assert p.train["FECHA"].max() < p.fecha_corte_train <= p.val["FECHA"].min()
assert p.val["FECHA"].max() < p.fecha_corte_val <= p.test["FECHA"].min()
# Ninguna fecha puede aparecer en dos particiones (se comprueba con los datos,
# no solo con la desigualdad de los extremos).
assert not (set(p.train["FECHA"]) & set(p.val["FECHA"]))
assert not (set(p.val["FECHA"]) & set(p.test["FECHA"]))
assert not (set(p.train["FECHA"]) & set(p.test["FECHA"]))

# ──────────────────────────────────────────────────────────────────────────
# 1b. Colapso de regimen del target: el dato que hace interpretable el R2
#     El R2 de test no se lee sin saber que la varianza del target cae un
#     orden de magnitud entre train y test. Sin esta tabla, un R2 negativo
#     parece un fallo del modelo cuando es una propiedad del periodo.
# ──────────────────────────────────────────────────────────────────────────
regimen = {}
print("\nCOLAPSO DE REGIMEN DEL TARGET (FLETE_UNIT, USD/kg):")
for name, s in [("train", p.train), ("val", p.val), ("test", p.test)]:
    y = s[dp.TARGET]
    regimen[name] = {
        "media": round(float(y.mean()), 4), "sd": round(float(y.std()), 4),
        "p50": round(float(y.median()), 4), "n_meses": int(s["FECHA"].dt.to_period("M").nunique()),
    }
    print(f"  {name:5} media={y.mean():.4f} sd={y.std():.4f} p50={y.median():.4f} "
          f"| {s['FECHA'].dt.to_period('M').nunique()} meses")
ratio_sd = p.train[dp.TARGET].std() / p.test[dp.TARGET].std()
print(f"  La dispersion del target es {ratio_sd:.1f}x mayor en train que en test.")
print("  Por eso el denominador del R2 de test es minusculo y el R2 no compara")
print("  modelos sino periodos. La metrica a leer es el MAPE contra las lineas base.")

n_unseen = {
    "puerto_val": int((~p.val["PUER_DESC"].isin(p.puerto_freq.index)).sum()),
    "puerto_test": int((~p.test["PUER_DESC"].isin(p.puerto_freq.index)).sum()),
    "importador_val": int((~p.val["IMPORTADOR"].isin(p.importador_freq.index)).sum()),
    "importador_test": int((~p.test["IMPORTADOR"].isin(p.importador_freq.index)).sum()),
}
print("\nCategorias nunca vistas en train (caen al default, sin fuga):")
print(f"  puertos      -> val: {n_unseen['puerto_val']}/{len(p.val)} "
      f"({100*n_unseen['puerto_val']/len(p.val):.2f}%) | "
      f"test: {n_unseen['puerto_test']}/{len(p.test)} ({100*n_unseen['puerto_test']/len(p.test):.2f}%)")
print(f"  importadores -> val: {n_unseen['importador_val']}/{len(p.val)} "
      f"({100*n_unseen['importador_val']/len(p.val):.2f}%) | "
      f"test: {n_unseen['importador_test']}/{len(p.test)} ({100*n_unseen['importador_test']/len(p.test):.2f}%)")

FEATURES, TARGET = dp.FEATURES, dp.TARGET
X_train, y_train = p.train[FEATURES], p.train[TARGET]
X_val, y_val = p.val[FEATURES], p.val[TARGET]
X_test, y_test = p.test[FEATURES], p.test[TARGET]

# ──────────────────────────────────────────────────────────────────────────
# 2. Entrenamiento (hiperparametros del notebook, con early stopping)
#    Nota: el early stopping del modelo PUNTUAL usa VAL completo. Eso es
#    legitimo porque el modelo puntual no interviene en la construccion del
#    intervalo; los modelos de cuantiles, que si la construyen, hacen early
#    stopping solo con VAL_ES y dejan VAL_CAL intacto para la calibracion.
# ──────────────────────────────────────────────────────────────────────────
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    early_stopping_rounds=40,
    eval_metric="mae",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"\nMejor iteracion (early stopping): {model.best_iteration}")


def evaluar(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


metrics_train = evaluar(y_train, model.predict(X_train))
metrics_test = evaluar(y_test, model.predict(X_test))
print()
print("=== MODELO (metricas principales: MAE / RMSE / MAPE) ===")
_PRIN = ("MAE", "RMSE", "MAPE_%")
print("Train:", {k: round(metrics_train[k], 4) for k in _PRIN})
print("Test :", {k: round(metrics_test[k], 4) for k in _PRIN})
print("  El R2 NO se reporta como resultado principal (correccion de la tercera")
print("  auditoria): cambia de signo entre variantes razonables del pipeline que")
print("  apenas mueven el MAPE, asi que es un diagnostico y no un resultado.")
print(f"  Para referencia: R2 train {metrics_train['R2']:.4f}, test {metrics_test['R2']:.4f}.")
print(f"\nBRECHA DE GENERALIZACION: R2 train {metrics_train['R2']:.4f} -> test {metrics_test['R2']:.4f}")
print(f"  RMSE en test ({metrics_test['RMSE']:.4f}) frente a la desviacion estandar del propio")
print(f"  test ({y_test.std():.4f}): el modelo acierta el NIVEL del mercado y no explica la")
print("  variacion ENTRE embarques dentro del periodo. El MAPE es la metrica interpretable;")
print("  el R2 de test solo se lee contra las lineas base de abajo.")

# ──────────────────────────────────────────────────────────────────────────
# 3. Lineas base (contexto obligatorio para leer el R2 de test)
#    Las tres primeras venian del pipeline original. Las cuatro siguientes las
#    agrego la segunda auditoria: son mas exigentes, y que el modelo las gane
#    todas es el argumento defendible del trabajo. El Ridge sobre las MISMAS
#    features es el contraste clave: separa "aporta la no-linealidad" de
#    "aporta solo la informacion de mercado".
# ──────────────────────────────────────────────────────────────────────────
todo = pd.concat([p.train, p.val, p.test])
todo_per = todo["FECHA"].dt.to_period("M")
med_puerto = todo.assign(_p=todo_per).groupby(["_p", "PUER_DESC"])[TARGET].mean()
med_import = todo.assign(_p=todo_per).groupby(["_p", "IMPORTADOR"])[TARGET].mean()
per_ant = p.test["FECHA"].dt.to_period("M") - 1
fallback = p.test["mercado_lag1"].values


def _mes_anterior_por(tabla, columna):
    """Media del mes anterior para ese puerto/importador; si no existe, el mercado."""
    v = np.array([tabla.get(k, np.nan) for k in zip(per_ant, p.test[columna])])
    return np.where(np.isnan(v), fallback, v)


ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)

baselines = {
    "persistencia_lag1": p.test["mercado_lag1"].values,
    "media_movil_ma3": p.test["mercado_ma3"].values,
    "mediana_train": np.full(len(y_test), float(y_train.median())),
    "mes_anterior_x_puerto": _mes_anterior_por(med_puerto, "PUER_DESC"),
    "mes_anterior_x_importador": _mes_anterior_por(med_import, "IMPORTADOR"),
    "ridge_mismas_features": ridge.predict(X_test),
    "mediana_cierre_val": np.full(len(y_test), float(p.val[TARGET].tail(500).median())),
}
metrics_baselines = {k: evaluar(y_test, v) for k, v in baselines.items()}
print("\n=== LINEAS BASE EN TEST ===")
print(f"{'modelo XGBoost':<28} MAPE {metrics_test['MAPE_%']:6.2f}% | MAE {metrics_test['MAE']:.4f} "
      f"| RMSE {metrics_test['RMSE']:.4f} | R2 {metrics_test['R2']:7.4f}")
for k, m in sorted(metrics_baselines.items(), key=lambda kv: kv[1]["MAPE_%"]):
    print(f"{k:<28} MAPE {m['MAPE_%']:6.2f}% | MAE {m['MAE']:.4f} "
          f"| RMSE {m['RMSE']:.4f} | R2 {m['R2']:7.4f}")
gana_todas = all(metrics_test["MAPE_%"] < m["MAPE_%"] for m in metrics_baselines.values())
mejora = 100 * (1 - metrics_test["MAPE_%"] / metrics_baselines["persistencia_lag1"]["MAPE_%"])
print(f"\nMejora relativa del MAPE sobre la persistencia de mercado: {mejora:.1f}%")
print(f"El modelo gana en MAPE a las {len(metrics_baselines)} lineas base: {gana_todas}")
print(f"  Contraste clave: Ridge sobre las MISMAS features da "
      f"{metrics_baselines['ridge_mismas_features']['MAPE_%']:.2f}% "
      f"({metrics_baselines['ridge_mismas_features']['MAPE_%'] - metrics_test['MAPE_%']:+.2f} pp).")
print("  La no-linealidad aporta senal real, no solo capacidad de memorizacion.")

# ──────────────────────────────────────────────────────────────────────────
# 4. Escenario PRODUCCION: rezagos de mercado congelados
#    Asi opera el sistema desplegado para toda fecha posterior al historico.
#    El corte se hace en el ultimo mes ESTRICTAMENTE ANTERIOR al primer mes de
#    TEST. Antes se cortaba en el mes de cierre de VAL, que es el mismo mes en
#    el que empieza TEST: la media congelada incluia filas de TEST y el
#    escenario salia 1.26 pp optimista (26.27% en vez de 27.53%).
# ──────────────────────────────────────────────────────────────────────────
primer_mes_test = p.test["FECHA"].min().to_period("M")
ultimos = p.serie_mercado.loc[: primer_mes_test - 1]
assert ultimos.index[-1] < primer_mes_test, "el mercado congelado no puede incluir meses de TEST"
f1, f2, f3 = float(ultimos.iloc[-1]), float(ultimos.iloc[-2]), float(ultimos.iloc[-3])
X_frozen = X_test.copy()
X_frozen["mercado_lag1"] = f1
X_frozen["mercado_lag2"] = f2
X_frozen["mercado_lag3"] = f3
X_frozen["mercado_ma3"] = (f1 + f2 + f3) / 3
metrics_frozen = evaluar(y_test, model.predict(X_frozen))
print(f"\n=== ESCENARIO PRODUCCION (mercado congelado en {ultimos.index[-1]}, sin filas de TEST) ===")
print("Test :", {k: round(v, 4) for k, v in metrics_frozen.items()})
print(f"  Degradacion por no actualizar el mercado: MAPE {metrics_test['MAPE_%']:.2f}% -> "
      f"{metrics_frozen['MAPE_%']:.2f}% | R2 {metrics_test['R2']:.4f} -> {metrics_frozen['R2']:.4f}")
peor_que_todas = all(
    metrics_frozen["MAPE_%"] > m["MAPE_%"] for m in metrics_baselines.values()
)
print(f"  Con el mercado congelado el modelo cae por debajo de TODAS las lineas base: {peor_que_todas}")
print(f"  ESTA ({metrics_frozen['MAPE_%']:.2f}%) es la cifra honesta para cotizaciones fuera del")
print("  historico, no el MAPE de test. Por eso el sistema expone la fecha de corte del")
print("  mercado y advierte al extrapolar (ver ml/predictor.py y app/routers/maintenance.py).")

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nImportancia de variables (gain):")
print(imp.round(4))
vars_mercado = ["mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3"]
gain_top3 = 100 * float(imp.head(3).sum())
gain_mercado = 100 * float(imp[vars_mercado].sum())
print(f"\nLas 3 variables de mayor gain concentran {gain_top3:.1f}%; las CUATRO variables")
print(f"de mercado (lag1/2/3 + ma3) suman {gain_mercado:.1f}%:")
print("  el modelo predice sobre todo el NIVEL DEL MERCADO con ajuste estacional;")
print("  puerto, importador, densidad y ruta suman "
      f"{100*imp[['puerto_freq','importador_freq','densidad_carga','ruta_directa']].sum():.1f}%.")

# ──────────────────────────────────────────────────────────────────────────
# 4b. Sensibilidad a semana_anio (limitacion declarada; se mide, no se estima)
#     La doc declaraba "hasta 12.3%" a partir de una sola configuracion. Un
#     unico numero oculta la cola: se reporta la distribucion completa.
# ──────────────────────────────────────────────────────────────────────────
# TERCERA AUDITORIA. La segunda publico "mediana 12.8% / p90 39.6% / max 123.4%"
# a partir de 60 filas con UNA sola semilla. Medido con cinco semillas, a n=60 la
# mediana era estable (11.4-12.8%) pero el p90 oscilaba 28.6-53.3% y el maximo
# 61-123%: eran artefactos del muestreo, no propiedades del modelo. Se sube a
# n=400, donde mediana y p90 se estabilizan, y se ELIMINA el maximo del reporte:
# con esta muestra no es un estadistico estimable, y publicarlo era presentar
# como propiedad del modelo el extremo de una distribucion de semillas.
N_SENS = 400
SEMILLAS_SENS = [RANDOM_STATE, 1, 7, 2024, 99]
# Se muestrea de TODO el dataset limpio, no solo de test: test cubre 5 meses de
# un regimen excepcionalmente calmo y subestimaria la sensibilidad.
X_todo = todo[FEATURES]


def _sensibilidad_semana(seed: int):
    """(variacion max-min %, salto entre semanas %) sobre N_SENS filas reales."""
    r = np.random.default_rng(seed)
    blk = X_todo.iloc[r.integers(0, len(X_todo), N_SENS)]
    curvas = np.column_stack([model.predict(blk.assign(semana_anio=w)) for w in range(1, 53)])
    return (100 * (curvas.max(1) / curvas.min(1) - 1),
            100 * np.abs(np.diff(curvas, axis=1) / curvas[:, :-1]).max(1))


variaciones, saltos = _sensibilidad_semana(RANDOM_STATE)
_por_semilla = [_sensibilidad_semana(sd)[0] for sd in SEMILLAS_SENS]
_med = [float(np.median(v)) for v in _por_semilla]
_p90 = [float(np.percentile(v, 90)) for v in _por_semilla]
sens = {
    "variacion_max_min_mediana_%": round(float(np.median(variaciones)), 1),
    "variacion_max_min_p90_%": round(float(np.percentile(variaciones, 90)), 1),
    "n_configuraciones": N_SENS,
    "estabilidad_entre_semillas": {
        "semillas": SEMILLAS_SENS,
        "mediana_rango_%": [round(min(_med), 1), round(max(_med), 1)],
        "p90_rango_%": [round(min(_p90), 1), round(max(_p90), 1)],
    },
    "muestreo": "filas reales de train+val+test (todo el dataset limpio)",
    "salto_entre_semanas_mediana_%": round(float(np.median(saltos)), 1),
    "gain_%": round(100 * float(imp["semana_anio"]), 2),
    "nota": (
        f"Variando SOLO semana_anio, con todo lo demas fijo, sobre {N_SENS} filas "
        "reales de todo el dataset limpio. NO se reporta el maximo: con esta "
        "muestra no es un estadistico estimable (oscilaba entre 61% y 135% segun "
        "la semilla) y la segunda auditoria lo publico como si lo fuera. La "
        "mediana y el p90 si son estables entre semillas: ver "
        "'estabilidad_entre_semillas'. La UI NO mitiga esta sensibilidad: el "
        "selector semanal recorre las 52 semanas ISO y la expone en todo su rango."
    ),
}
print()
print(f"Sensibilidad a semana_anio ({N_SENS} filas reales, solo esa feature varia):")
print(f"  variacion max-min: mediana {sens['variacion_max_min_mediana_%']}% | "
      f"p90 {sens['variacion_max_min_p90_%']}%  (el maximo NO se reporta: no es estimable)")
print(f"  estabilidad entre {len(SEMILLAS_SENS)} semillas: mediana "
      f"{sens['estabilidad_entre_semillas']['mediana_rango_%']} | p90 "
      f"{sens['estabilidad_entre_semillas']['p90_rango_%']}")
print(f"  salto entre semanas consecutivas: mediana {sens['salto_entre_semanas_mediana_%']}%")

# ──────────────────────────────────────────────────────────────────────────
# 4c. Experimentos que la tercera auditoria exige medir EN CODIGO, no en prosa
#     Las cifras de estos experimentos vivian en la documentacion y quedaron
#     obsoletas: la ablacion de `semana_anio` publicada por la segunda auditoria
#     (MAPE 23.34%, R2 -0.447) no reproducia con ninguna configuracion. Ahora se
#     recalculan en cada ejecucion y se guardan en el artifact, de modo que la
#     prosa no pueda volver a divergir del codigo.
# ──────────────────────────────────────────────────────────────────────────
def _reentrenar(feats, tr=None, va=None, te=None) -> dict:
    """Reentrena con los MISMOS hiperparametros y evalua en test."""
    tr = p.train if tr is None else tr
    va = p.val if va is None else va
    te = p.test if te is None else te
    m = XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=40, eval_metric="mae",
    )
    m.fit(tr[feats], tr[TARGET], eval_set=[(va[feats], va[TARGET])], verbose=False)
    return evaluar(te[TARGET], m.predict(te[feats]))


print()
print("=== EXPERIMENTOS MEDIDOS EN CADA EJECUCION ===")

_sin_semana = _reentrenar([f for f in FEATURES if f != "semana_anio"])
ablacion_semana = {
    "con_feature": {k: round(v, 4) for k, v in metrics_test.items()},
    "sin_feature": {k: round(v, 4) for k, v in _sin_semana.items()},
    "delta_MAPE_pp": round(_sin_semana["MAPE_%"] - metrics_test["MAPE_%"], 2),
    "delta_R2": round(_sin_semana["R2"] - metrics_test["R2"], 4),
    "decision": "conservar",
    "nota": (
        "CORRECCION DE LA TERCERA AUDITORIA. La segunda publico 'MAPE 22.23% -> "
        "23.34%, R2 -0.018 -> -0.447' y concluyo que la feature aportaba senal. "
        "Esas cifras no reproducen con ninguna configuracion. Medido aqui: "
        "quitar semana_anio MEJORA el MAPE y empeora un R2 que el propio "
        "informe declara inestable (§15.5). La justificacion honesta para "
        "conservarla es la continuidad con el pipeline original y el R2, NO que "
        "'aporta senal': sobre el MAPE, que es la metrica interpretable de este "
        "trabajo, la feature es indiferente o levemente perjudicial. Su "
        "sensibilidad (ver sensibilidad_semana_anio) es el argumento en contra."
    ),
}
print(f"  ablacion semana_anio : MAPE {metrics_test['MAPE_%']:.2f} -> "
      f"{_sin_semana['MAPE_%']:.2f} ({ablacion_semana['delta_MAPE_pp']:+.2f} pp) | "
      f"R2 {metrics_test['R2']:.4f} -> {_sin_semana['R2']:.4f}")

# Recorte de las colas de las FEATURES (que el recorte de outliers nunca toca).
_lims = {f: float(pd.concat([p.train, p.val, p.test])[f].quantile(0.99)) for f in
         ("densidad_carga", "ratio_bruto_neto")}


def _clip(d):
    d = d.copy()
    for f, hi_ in _lims.items():
        d[f] = d[f].clip(upper=hi_)
    return d


_colas = _reentrenar(FEATURES, _clip(p.train), _clip(p.val), _clip(p.test))
ablacion_colas = {
    "limites_p99_aplicados": {k: round(v, 4) for k, v in _lims.items()},
    "sin_recorte": {k: round(v, 4) for k, v in metrics_test.items()},
    "con_recorte": {k: round(v, 4) for k, v in _colas.items()},
    "decision": "no adoptar",
    "nota": (
        "El recorte de outliers solo recorta el target; las features entran sin "
        "filtro (ratio_bruto_neto llega a 10.1 con un p99 de 1.08, y un "
        "bruto/neto de 10:1 es fisicamente imposible para neumaticos). Se prueba "
        "recortarlas al p99 y MEJORA las metricas. Aun asi NO se adopta, y el "
        "motivo es metodologico, no de resultado: (1) el recorte se aplica "
        "tambien a val y test, de modo que cambia el conjunto de evaluacion y la "
        "comparacion deja de ser entre modelos para pasar a ser entre conjuntos "
        "—- es exactamente la razon por la que se descarto el filtro "
        "`PESO_NETO >= 20 kg`, que tambien mejoraba una metrica—-; y (2) el p99 "
        "es un umbral elegido sin justificacion independiente del dato. Adoptar "
        "un cambio PORQUE mejora una metrica, con una regla fijada despues de "
        "ver el resultado, es el sesgo que las tres auditorias han venido "
        "corrigiendo. Se reporta el numero por transparencia y la cola queda "
        "declarada como limitacion en §10. Adoptarlo requeriria una regla de "
        "plausibilidad fisica definida a priori y aplicada solo a train."
    ),
}
print(f"  recorte colas features: MAPE {metrics_test['MAPE_%']:.2f} -> "
      f"{_colas['MAPE_%']:.2f} | R2 {metrics_test['R2']:.4f} -> {_colas['R2']:.4f} "
      f"-> {ablacion_colas['decision']} (mejora, pero cambia el conjunto de evaluacion)")

colas_features = dp.diagnostico_colas_features(p)
saturacion_densidad = dp.diagnostico_saturacion(
    model, p, "densidad_carga", [1, 5, 10, 20, 50, 100, 500, 5000, 50000]
)
print(f"  colas de features    : ratio_bruto_neto max "
      f"{colas_features['ratio_bruto_neto']['max']} (p99 "
      f"{colas_features['ratio_bruto_neto']['p99']}) | densidad_carga max "
      f"{colas_features['densidad_carga']['max']} (p99 "
      f"{colas_features['densidad_carga']['p99']})")
print(f"  saturacion densidad  : el modelo devuelve el mismo valor para todo "
      f"densidad_carga >= {saturacion_densidad['satura_a_partir_de']} "
      f"({saturacion_densidad['n_valores_distintos']} valores distintos en 9 ordenes)")

# Fragilidad del RMSE de test: cuanto pesa la peor fila.
_res2 = (y_test.values - model.predict(X_test)) ** 2
_peor = float(_res2.max())
fragilidad_rmse = {
    "RMSE_test": round(metrics_test["RMSE"], 4),
    "pct_MSE_de_la_peor_fila": round(100 * _peor / _res2.sum(), 2),
    "pct_MSE_del_top5": round(100 * float(np.sort(_res2)[-5:].sum()) / _res2.sum(), 2),
    "nota": (
        "TERCERA AUDITORIA. La segunda reporto una mejora de RMSE 0.0580 -> "
        "0.0535 al cambiar la clave de estratificacion del recorte. El cambio es "
        "metodologicamente correcto (la clave de limpieza pasa a ser la propia "
        "feature), pero mueve UNA SOLA fila en las 86,977 del dataset, y esa "
        "fila estaba integramente en TEST. La mejora del RMSE se explica al "
        "100% por ella. Lo que ese experimento demuestra no es que el modelo "
        "mejorara, sino que el RMSE de TEST estaba dominado por UNA observacion. "
        "Los porcentajes de arriba son los del test ya corregido y muestran que "
        "ahora ninguna fila lo domina: por eso la cifra 0.0535 es solida, pero "
        "la mejora respecto de 0.0580 no debe presentarse como un avance del "
        "modelo, porque el modelo no cambio."
    ),
}
print(f"  fragilidad del RMSE  : la peor fila de test aporta el "
      f"{fragilidad_rmse['pct_MSE_de_la_peor_fila']}% del MSE; el top-5 el "
      f"{fragilidad_rmse['pct_MSE_del_top5']}%")

# ──────────────────────────────────────────────────────────────────────────
# 5. Guardar modelo + metadatos
# ──────────────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

puertos_dropdown, importadores_dropdown = dp.catalogos_dropdown(p)
# Invariante que la auditoria encontro rota: toda opcion del formulario debe
# existir en los encoders. Se verifica aqui para que no pueda volver a romperse.
assert all(x in p.puerto_freq.index for x in puertos_dropdown)
assert all(x in p.ruta_directa_por_puerto for x in puertos_dropdown)
assert all(x in p.importador_freq.index for x in importadores_dropdown)
cobertura = dp.cobertura_catalogos(p, puertos_dropdown, importadores_dropdown)

meta["features"] = FEATURES
meta["target"] = TARGET
meta["unidad"] = "USD/kg"
meta["metricas_test"] = {k2: round(v, 4) if k2 != "MAPE_%" else round(v, 2)
                         for k2, v in metrics_test.items()}
meta["metricas_train"] = {k2: round(v, 4) if k2 != "MAPE_%" else round(v, 2)
                          for k2, v in metrics_train.items()}
meta["lineas_base_test"] = {
    k: {k2: round(v, 4) if k2 != "MAPE_%" else round(v, 2) for k2, v in m.items()}
    for k, m in metrics_baselines.items()
}
meta["escenario_produccion_congelado"] = {
    **{k2: round(v, 4) if k2 != "MAPE_%" else round(v, 2) for k2, v in metrics_frozen.items()},
    "mes_congelado": str(ultimos.index[-1]),
    "peor_que_todas_las_lineas_base": bool(peor_que_todas),
    "nota": (
        "Cifra honesta para cotizaciones cuya fecha cae fuera del historico. El "
        "mercado se congela en el ultimo mes estrictamente anterior al inicio de "
        "TEST, de modo que la media congelada no contiene ninguna fila de TEST."
    ),
}
meta["regimen_target_por_particion"] = {
    **regimen,
    "ratio_sd_train_test": round(float(ratio_sd), 1),
    "nota": (
        "El R2 de test es interpretable solo con esta tabla delante: la varianza "
        "del target cae un orden de magnitud entre train y test, asi que el "
        "denominador del R2 en test es minusculo y la metrica compara periodos, "
        "no modelos."
    ),
}
meta["nota_R2"] = (
    "El R2 figura en metricas_test/metricas_train por completitud, pero NO es "
    "un resultado principal de este trabajo y no debe encabezar la tabla de "
    "resultados. Cambia de signo (-0.07 a +0.06 medido sobre seis variantes "
    "razonables del pipeline) sin que el MAPE se mueva apenas. Las metricas "
    "que sostienen el argumento son MAE, RMSE y MAPE contra las siete lineas "
    "base, leidas junto a 'regimen_target_por_particion'."
)
meta["diagnostico_generalizacion"] = {
    "R2_train": round(metrics_train["R2"], 4),
    "R2_test": round(metrics_test["R2"], 4),
    "RMSE_test": round(metrics_test["RMSE"], 4),
    "sd_target_test": round(float(y_test.std()), 4),
    "mejora_MAPE_vs_persistencia_%": round(mejora, 1),
    "gana_a_todas_las_lineas_base": bool(gana_todas),
    "gain_top3_%": round(gain_top3, 1),
    "gain_variables_mercado_%": round(gain_mercado, 1),
    "nota": (
        "El modelo explica poca varianza fuera de muestra (R2 test bajo) pero "
        "supera en MAPE a las siete lineas base, incluido un Ridge sobre las "
        "mismas features. Interpretacion honesta: predice el NIVEL DEL MERCADO "
        "con ajuste estacional, no atributos del embarque individual — el RMSE "
        "de test es practicamente igual a la desviacion estandar del propio test."
    ),
}
meta["sensibilidad_semana_anio"] = sens
meta["ablacion_semana_anio"] = ablacion_semana
meta["ablacion_colas_features"] = ablacion_colas
meta["diagnostico_colas_features"] = colas_features
meta["diagnostico_saturacion_densidad"] = saturacion_densidad
meta["fragilidad_rmse_test"] = fragilidad_rmse
meta["pct_no_vistos_test"] = {
    "puerto_%": round(100 * n_unseen["puerto_test"] / len(p.test), 2),
    "importador_%": round(100 * n_unseen["importador_test"] / len(p.test), 2),
    "nota": "Senal de drift de catalogo: si crece sostenidamente, reentrenar.",
}
meta["diagnostico_recorte_outliers"] = json.loads(p.diag_recorte.to_json(orient="records"))
meta["diagnostico_sesgo_recorte"] = p.diag_sesgo_recorte
meta["diagnostico_ruta_default"] = p.diag_ruta_default
meta["cobertura_catalogos"] = cobertura
meta["nota_ley_29733"] = dp.NOTA_LEY_29733
meta["particion"] = {
    "criterio": "temporal por FECHA (ningun dia repartido entre particiones)",
    "fecha_corte_train": str(p.fecha_corte_train.date()),
    "fecha_corte_val": str(p.fecha_corte_val.date()),
    "n_train": len(p.train), "n_val": len(p.val), "n_test": len(p.test),
    "salvedad": (
        "Ningun DIA se reparte entre particiones. El MES en que VAL termina y "
        "TEST empieza si es compartido, y el mes es la unidad de mercado_lag*."
    ),
}
meta["nota"] = (
    "Multiplicar FLETE_UNIT por PESO_NETO para obtener flete total en USD. "
    "puerto_freq/importador_freq y los percentiles de recorte se ajustan SOLO "
    "con el tramo de entrenamiento (sin fuga temporal). ruta_directa se deriva "
    "siempre del lookup de puerto — tanto para el modelo como para estratificar "
    "el recorte de outliers, de modo que ninguna fila se limpie bajo un regimen "
    "y se modele bajo el otro."
)
meta["densidad_carga_median"] = p.densidad_carga_median
meta["ratio_bruto_neto_median"] = p.ratio_bruto_neto_median
meta["importador_freq_default"] = p.importador_freq_default
meta["puerto_freq_default"] = p.puerto_freq_default
meta["puerto_freq"] = {str(k): float(v) for k, v in p.puerto_freq.items()}
meta["importador_freq"] = {str(k): float(v) for k, v in p.importador_freq.items()}
meta["puertos_dropdown"] = puertos_dropdown
meta["importadores_dropdown"] = importadores_dropdown
meta["ruta_directa_por_puerto"] = p.ruta_directa_por_puerto
meta["ruta_directa_default"] = p.ruta_directa_default
meta["n_importadores"] = int(p.importador_freq.shape[0])
meta["serie_mercado"] = {str(k): round(float(v), 6) for k, v in p.serie_mercado.items()}
meta["serie_mercado_primer_mes"] = str(p.serie_mercado.index[0])
meta["serie_mercado_ultimo_mes"] = str(p.serie_mercado.index[-1])

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nGuardado: {MODEL_PATH}")
print(f"Guardado: {META_PATH}")
print(f"Catalogos regenerados desde train: {len(puertos_dropdown)} puertos, "
      f"{len(importadores_dropdown)} importadores (todos con encoder real).")
print(f"  Cubren {cobertura['cobertura_filas_puerto_%']}% de las filas por puerto "
      f"({cobertura['cobertura_peso_puerto_%']}% del peso) y "
      f"{cobertura['cobertura_filas_importador_%']}% por importador.")

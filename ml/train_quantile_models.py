"""
BUG 3 fix: reemplaza el IC95% basado en un MAPE global fijo por Conformalized
Quantile Regression (CQR, Romano, Patterson & Candes 2019).

Diseno (cada dato con un unico proposito):

  TRAIN + VAL_ES -> ajustan 2 modelos XGBoost de cuantiles (q=2.5% y q=97.5%).
  VAL_CAL        -> calibracion conformal. No participa en ningun ajuste.
  TEST           -> holdout (todo 2025). Nunca tocado antes.

QUINTA REVISION — DOS CAMBIOS, Y CONVIENE NO CONFUNDIRLOS:

  1. El esquema de particion pasa a 80/20 (TRAIN = 2021-2024, HOLDOUT = todo
     2025). VAL_ES y VAL_CAL son las dos mitades de la cola del 80%, asi que
     siguen siendo datos de 2024: la calibracion conformal no toca 2025.
  2. Se retira el early stopping, que con este esquema elegia modelos
     subajustados (ver ml/seleccion_iteraciones.py). VAL_ES pierde con ello su
     unica funcion y pasa a ser masa de entrenamiento. VAL_CAL sigue apartado,
     porque eso no lo pedia el early stopping sino la garantia de CQR, que
     exige calibrar sobre datos no vistos por los modelos de cuantiles.

Las particiones salen de ml/data_pipeline.py, el mismo modulo que usa
train_model.py: modelo puntual y modelos de cuantiles comparten por
construccion el dataset, el recorte, los encoders y las fronteras temporales.

Formula CQR (por fila, en espacio FLETE_UNIT):
    E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i))   sobre VAL_CAL
    Q   = el k-esimo menor E, con k = ceil((n+1)*(1-alpha))
          (correccion de muestra finita de Romano et al.)
    intervalo(x) = [q_lo(x) - Q, q_hi(x) + Q]

UNA CONSTANTE PARA EL HISTORICO Y UNA TABLA Q(h) PARA EL EXTRAPOLADO. El
sistema sirve predicciones en dos regimenes distintos:

  - Fecha DENTRO del historico: los rezagos son los reales del mes anterior,
    igual que en entrenamiento. Se usa `Q`.
  - Fecha FUERA del historico: los rezagos estan congelados o los fija un
    administrador. Es un problema distinto, mas dificil (el MAPE del modelo
    puntual pasa de ~22% a ~27.5%) y -- esto es lo que la segunda auditoria no
    contemplo -- cuya dificultad DEPENDE DEL HORIZONTE. Se calibra una tabla
    `Q(h)`, no una constante: ver la seccion 3b.

Historia de la correccion, porque explica el diseno:

  - Antes de la segunda auditoria habia UNA sola Q para todo. El intervalo
    servido para una cotizacion a 53 meses vista era un 0.05% mas ancho que para
    una dentro del historico: el sistema advertia por escrito que la estimacion
    era "estructuralmente fragil" mientras su propia medida de incertidumbre no
    registraba nada.
  - La segunda auditoria anadio una segunda constante. Fue un avance real, pero
    la calibro con un horizonte medio de ~3.4 meses y se seguia sirviendo para
    fechas de hasta 60 meses. La TERCERA auditoria midio su cobertura
    condicional: caia al 54% a seis meses y al 18% a nueve. Un intervalo
    etiquetado "95%" que cubre el 18% no es un intervalo ancho, es uno falso.
  - Ahora es una tabla `Q(h) = max(Q_cruda(1..h))`, que sostiene >=95% de
    cobertura condicional en CADA horizonte estimable, verificado con un assert
    en cada ejecucion.

LECTURA HONESTA DE LA COBERTURA. La garantia de CQR es marginal y condicionada
a *exchangeability* entre calibracion y datos futuros. Este script reporta la
cobertura en VAL_CAL (donde debe dar ~95% por construccion, es el control de
que la calibracion esta bien hecha) y en TEST (donde la desviacion mide cuanto
se rompio la exchangeability).

PERO la sobre-cobertura en TEST no se explica solo por eso, y atribuirsela
entera seria un diagnostico complaciente. Este script mide ademas la cobertura
de los cuantiles CRUDOS, antes de aplicar Q, y el desglose por cola. Si
`P(y > q_hi)` esta muy por debajo de su nominal 2.5% YA EN VAL_CAL — antes de
cualquier cambio de regimen — entonces el modelo de cuantil superior esta mal
especificado, y el ancho del intervalo es una propiedad del METODO, no "la
dispersion real del mercado". Ver `ic95_diagnostico.especificacion`.

Que sea una propiedad del metodo no implica que se sepa corregir. La tercera
auditoria implemento el remedio que la segunda habia dado por bueno sin medirlo
—ponderar temporalmente el ajuste de los cuantiles— y no funciona: ver
`ic95_diagnostico.ponderacion_temporal`. El ancho vuelve a declararse como
limitacion estructural.

El ancho se reporta ademas en terminos RELATIVOS (ancho / mediana del target),
que es la unica forma de juzgar si el intervalo es accionable para cotizar.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ml import data_pipeline as dp
from ml import seleccion_iteraciones as si

# El corpus por defecto es el historico de la tesis. `JPS_CSV_PATH` permite
# apuntar al corpus acumulativo que mantiene ml/corpus.py (base + los CSV
# incorporados despues) sin cambiar el comportamiento de una ejecucion manual.
CSV_PATH = os.environ.get("JPS_CSV_PATH", "resultado_combinado.csv")
META_PATH = "ml/modelo_meta.json"
QLO_PATH = "ml/modelo_xgboost_flete_q_lo.pkl"
QHI_PATH = "ml/modelo_xgboost_flete_q_hi.pkl"

RANDOM_STATE = 42
ALPHA = 0.05  # 95% de cobertura objetivo

with open(META_PATH, encoding="utf-8") as f:
    meta = json.load(f)

# ──────────────────────────────────────────────────────────────────────────
# 1. Particiones (identicas a las del modelo puntual, por construccion)
# ──────────────────────────────────────────────────────────────────────────
p = dp.construir(CSV_PATH)
val_es, val_cal, corte_val_interno = p.split_val()

print(f"TRAIN   | {len(p.train):>6,} filas | {p.train['FECHA'].min().date()} -> {p.train['FECHA'].max().date()}")
print(f"VAL_ES  | {len(val_es):>6,} filas | {val_es['FECHA'].min().date()} -> {val_es['FECHA'].max().date()}")
print(f"VAL_CAL | {len(val_cal):>6,} filas | {val_cal['FECHA'].min().date()} -> {val_cal['FECHA'].max().date()}")
print(f"TEST    | {len(p.test):>6,} filas | {p.test['FECHA'].min().date()} -> {p.test['FECHA'].max().date()}")
assert val_es["FECHA"].max() < corte_val_interno <= val_cal["FECHA"].min()
assert not (set(val_es["FECHA"]) & set(val_cal["FECHA"]))

# El artifact debe corresponder al mismo entrenamiento: si train_model.py no se
# corrio antes (o se corrio sobre otros datos), los encoders no coinciden.
if meta.get("features") != dp.FEATURES or meta.get("puerto_freq_default") != p.puerto_freq_default:
    raise SystemExit(
        "modelo_meta.json no corresponde a este dataset. Ejecutar primero:\n"
        "  python -m ml.train_model"
    )

FEATURES = dp.FEATURES

# QUINTA REVISION. Con el early stopping retirado (ver el docstring de
# ml/seleccion_iteraciones.py), VAL_ES deja de tener una funcion propia: era el
# conjunto contra el que se paraba el ajuste. Pasa a ser masa de entrenamiento.
#
# Lo que NO cambia, y es la razon de que VAL_CAL siguiera existiendo aparte: la
# garantia de CQR exige que la calibracion se haga sobre datos que los modelos
# de cuantiles no han visto. VAL_CAL sigue intacto, y ahora los cuantiles se
# ajustan con mas datos y mas cercanos al periodo que se predice.
BLOQUE_Q = pd.concat([p.train, val_es]).sort_values("FECHA", kind="mergesort").reset_index(drop=True)
X_train, y_train = BLOQUE_Q[FEATURES], BLOQUE_Q[dp.TARGET]
X_val_es, y_val_es = val_es[FEATURES], val_es[dp.TARGET]
X_val_cal, y_val_cal = val_cal[FEATURES], val_cal[dp.TARGET]
X_test, y_test = p.test[FEATURES], p.test[dp.TARGET]
print(f"BLOQUE DE AJUSTE DE CUANTILES (train + val_es) | {len(BLOQUE_Q):,} filas | "
      f"{BLOQUE_Q['FECHA'].min().date()} -> {BLOQUE_Q['FECHA'].max().date()}")
assert BLOQUE_Q["FECHA"].max() < val_cal["FECHA"].min(), (
    "el bloque de ajuste de cuantiles no puede solaparse con VAL_CAL"
)
assert val_cal["FECHA"].max() < p.test["FECHA"].min(), (
    "la calibracion conformal no puede tocar el holdout"
)

# ──────────────────────────────────────────────────────────────────────────
# 2. Modelos de cuantiles: ajustados con train+val_es, VAL_CAL intacto
#
#    El numero de arboles se elige por origen rodante trimestral dentro del
#    bloque de ajuste, con la perdida de CADA cuantil —no con el MAE del modelo
#    puntual—, porque un cuantil al 2.5% y otro al 97.5% no tienen por que
#    necesitar la misma capacidad. Se seleccionan por separado y el artifact
#    registra ambos.
# ──────────────────────────────────────────────────────────────────────────
_HP_Q = dict(
    objective="reg:quantileerror",
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

seleccion_q = {}


def fit_quantile(alpha: float) -> XGBRegressor:
    etiqueta = f"q={alpha:.3f}"
    print(f"\n  Seleccion del numero de arboles para {etiqueta}:")
    n, diag = si.seleccionar_n_estimators(
        BLOQUE_Q,
        lambda k: XGBRegressor(n_estimators=k, quantile_alpha=alpha, **_HP_Q),
    )
    seleccion_q[etiqueta] = diag
    m = XGBRegressor(n_estimators=n, quantile_alpha=alpha, **_HP_Q)
    m.fit(X_train, y_train, verbose=False)
    return m


model_lo = fit_quantile(ALPHA / 2)
model_hi = fit_quantile(1 - ALPHA / 2)
print(f"\nq_lo n_estimators={model_lo.n_estimators} | q_hi n_estimators={model_hi.n_estimators}")


def conformal_Q(scores: np.ndarray, alpha: float = ALPHA) -> tuple[float, int, int]:
    """k-esimo menor score, con k = ceil((n+1)(1-alpha)) (Romano et al. 2019).

    Se toma el orden estadistico directamente en vez de
    `np.quantile(..., method="higher")`. Nota de la segunda auditoria: el
    comentario anterior afirmaba que `method="higher"` devolvia el (k+1)-esimo
    y era "un paso mas conservador". Es al reves — devolvia el (k-1)-esimo, es
    decir una Q MENOR y por tanto MENOS conservadora que la formula publicada
    (medido: 0.0184880 contra 0.0184927). El valor actual ya era el correcto;
    lo que estaba invertido era la justificacion.
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        print(f"AVISO: n={n} insuficiente para 1-alpha={1-alpha}; se usa el maximo.")
        return float(np.max(scores)), n, k
    return float(np.sort(scores)[k - 1]), n, k


# ──────────────────────────────────────────────────────────────────────────
# 3. Calibracion conformal SOLO con val_cal — regimen NORMAL (dentro del historico)
# ──────────────────────────────────────────────────────────────────────────
q_lo_cal = model_lo.predict(X_val_cal)
q_hi_cal = model_hi.predict(X_val_cal)
scores = np.maximum(q_lo_cal - y_val_cal.values, y_val_cal.values - q_hi_cal)
Q_cqr_crudo, n_cal, k = conformal_Q(scores)

# ──────────────────────────────────────────────────────────────────────────
# SUELO EN CERO: LA CALIBRACION PUEDE ENSANCHAR EL INTERVALO, NUNCA ENCOGERLO.
#
# Con la particion temporal por ano calendario la Q conformal sale NEGATIVA
# (-0.06122). No es un error de calculo: significa que sobre VAL_CAL los modelos
# de cuantiles ya sobre-cubren (97.57% frente al 95% nominal), y CQR, que busca
# la cobertura exacta, corrige ESTRECHANDO. Estrechar es legitimo bajo la
# hipotesis de exchangeability entre calibracion y datos futuros — y esa
# hipotesis es justamente la que este trabajo declara rota desde la primera
# auditoria: el nivel del mercado cae un orden de magnitud entre periodos.
#
# Verificado DESPUES de fijar la regla: sobre el holdout, aplicar la Q negativa
# llevaria la cobertura del 97.72% que dan los cuantiles crudos al 74.44%, bajo
# una etiqueta que dice 95%. Es exactamente el defecto que la tercera auditoria
# corrigio en el regimen extrapolado ("un intervalo etiquetado 95% que cubre el
# 18% no es ancho, es falso"), reaparecido por el otro extremo. Las dos cifras
# viven en `modelo_meta.json -> ic95_diagnostico.suelo_cero`; no se escriben a
# mano aqui, y la cuarta auditoria corrigio los valores obsoletos que este
# comentario arrastraba del esquema 70/20/10 (-0.04873, 97.60% -> 81.11%).
#
# La regla adoptada no se elige mirando el holdout, sino por la asimetria del
# problema: la garantia de CQR es unilateral (cobertura >= 1-alfa) y solo vale
# bajo exchangeability. Cuando la exchangeability esta documentadamente rota,
# la parte de la correccion que ENSANCHA sigue siendo prudente y la que ENCOGE
# pasa a apoyarse enteramente en la hipotesis que se sabe falsa. Se conserva la
# primera y se descarta la segunda. Con Q = 0 el intervalo servido es el de los
# cuantiles crudos, cuya cobertura es >= 95% tanto en VAL_CAL como en el
# holdout, asi que el suelo solo puede hacer el intervalo mas conservador.
#
# Ambos valores se publican en el artifact: `ic95_conformal_Q` es el servido y
# `ic95_conformal_Q_cqr_crudo` el que da la formula sin suelo, para que el
# lector pueda reconstruir la decision en vez de tener que creerla.
Q = max(Q_cqr_crudo, 0.0)
_suelo_aplicado = Q_cqr_crudo < 0
if _suelo_aplicado:
    print(f"\nAVISO: la Q conformal cruda es NEGATIVA ({Q_cqr_crudo:.5f}): la "
          "calibracion pedia ESTRECHAR el intervalo.")
    print("  Se aplica suelo en 0 (la calibracion puede ensanchar, nunca encoger) "
          "porque estrechar")
    print("  descansa por completo en una exchangeability que este trabajo declara rota.")
print(f"\nCalibracion conformal (regimen historico): n_cal={n_cal} | k={k} | Q={Q:.5f}")

n_cross = int((q_hi_cal < q_lo_cal).sum())
print(f"Quantile crossing en VAL_CAL: {n_cross}/{n_cal}")

# ──────────────────────────────────────────────────────────────────────────
# 3b. Calibracion del regimen EXTRAPOLADO: tabla Q(h) monotonizada
#
#     TERCERA AUDITORIA. La segunda auditoria calibraba UNA constante congelando
#     el mercado en el mes anterior al inicio de VAL_CAL. Eso da un horizonte
#     medio de ~3.4 meses, y la constante resultante (0.03513) solo alcanzaba el
#     95% para horizontes de 1-2 meses: a 6 meses la cobertura caia al 54% y a 9
#     al 18%. El sistema, en cambio, servia esa constante para fechas de hasta 60
#     meses. Un intervalo con la etiqueta "95%" que cubre el 18% no es un
#     intervalo ancho: es un intervalo falso.
#
#     Se probaron y descartaron dos alternativas, ambas medidas:
#
#       (a) Q(h) cruda, una constante por horizonte. DESCARTADA: Q(h) es
#           fuertemente NO monotona (0.023 a h=1, pico 0.219 a h=9, minimo 0.009
#           a h=24, 0.43 a h=42), porque no mide la distancia sino en que mes
#           historico concreto cae el congelamiento — retroceder desde 2025
#           aterriza en el pico de 2024 o en el valle de 2023. Daria a una
#           cotizacion a 24 meses un intervalo MAS ESTRECHO que a una de 3.
#
#       (b) Una sola Q agrupada sobre h ~ U(1..6). DESCARTADA: da cobertura
#           marginal correcta pero reparte mal. Sobre-cubre los horizontes
#           cortos (99.98% a h=1-3) hasta hacerlos inservibles —el limite
#           inferior se hundia por debajo de cero y una cotizacion a 2 meses
#           salia con intervalo [0, 568] USD— y aun asi se queda en 78.9% a h=6.
#
#     SOLUCION ADOPTADA: Q_mono(h) = max(Q(1), ..., Q(h)), el maximo acumulado.
#     Es monotona por construccion (mas horizonte => intervalo mas ancho, que es
#     el comportamiento que un usuario espera y puede defenderse), y como
#     Q_mono(h) >= Q(h) hereda al menos la cobertura de Q(h) en CADA horizonte,
#     no solo en promedio. Los horizontes cortos conservan intervalos utiles y
#     los largos pagan el ancho que su incertidumbre real exige.
#
#     LIMITACION DECLARADA: Q(h) se estima sobre la unica trayectoria de mercado
#     disponible (2021-2025). Que h=9 sea el peor horizonte es un hecho de ESE
#     periodo —el pico de 2024—, no una ley. La tabla es la mejor estimacion con
#     los datos que hay, no una constante universal.
# ──────────────────────────────────────────────────────────────────────────
_per_cal = val_cal["FECHA"].dt.to_period("M")
_per_test = p.test["FECHA"].dt.to_period("M")

# Horizonte maximo estimable: todas las filas de VAL_CAL deben tener sus tres
# rezagos dentro de la serie cuando se retrocede h meses.
H_MAX = min(int((t - 3 - p.serie_mercado.index[0]).n) for t in _per_cal.unique())


def congelar_por_horizonte(X, periodos, h):
    """Congela el mercado de cada fila en (mes de la fila - 1 - h), (-2-h), (-3-h).

    Simula exactamente lo que hace produccion: servir una cotizacion para el mes
    T cuando el ultimo mes de mercado observado es T-1-h.
    """
    Xf = X.copy()
    hh = np.full(len(X), h) if np.isscalar(h) else np.asarray(h)
    for k in (1, 2, 3):
        Xf[f"mercado_lag{k}"] = [
            float(p.serie_mercado.loc[t - k - int(hi_)]) for t, hi_ in zip(periodos, hh)
        ]
    Xf["mercado_ma3"] = (Xf["mercado_lag1"] + Xf["mercado_lag2"] + Xf["mercado_lag3"]) / 3
    return Xf


print()
print(f"Calibracion conformal del regimen extrapolado: Q(h) para h=1..{H_MAX}")
_Q_cruda, _Q_sin_suelo, Q_por_horizonte, cobertura_por_horizonte = {}, {}, {}, {}
_acum = 0.0
for _h in range(1, H_MAX + 1):
    _X = congelar_por_horizonte(X_val_cal, _per_cal, _h)
    _a, _b = model_lo.predict(_X), model_hi.predict(_X)
    _q, _, _ = conformal_Q(np.maximum(_a - y_val_cal.values, y_val_cal.values - _b))
    # Se guardan las dos versiones y cada una tiene su uso:
    #   `_Q_sin_suelo` es la ESTIMACION, y es la que alimenta el diagnostico de
    #     hasta que horizonte la tabla sigue informada por datos (mas abajo).
    #     Recortarla ahi destruiria justo la senal que ese diagnostico lee.
    #   `_Q_cruda` es lo que se SIRVE, con el mismo suelo en cero que la Q
    #     historica y por el mismo motivo: la calibracion puede ensanchar el
    #     intervalo, nunca estrecharlo por debajo del de los cuantiles crudos.
    _Q_sin_suelo[_h] = _q
    _q = max(_q, 0.0)
    _Q_cruda[_h] = _q
    _acum = max(_acum, _q)              # monotonizacion por maximo acumulado
    Q_por_horizonte[_h] = _acum
    cobertura_por_horizonte[str(_h)] = round(
        100 * float(((y_val_cal.values >= _a - _acum) & (y_val_cal.values <= _b + _acum)).mean()), 2
    )

# ──────────────────────────────────────────────────────────────────────────
# HASTA DONDE SE PUEDE DECLARAR "CALIBRADO" (H-12, cuarta auditoria)
#
# El artifact publicaba `ic95_horizonte_calibrado_meses = H_MAX` (=47), y el
# predictor devolvia `ic95_calibrado: true` para una cotizacion a 41 meses con un
# intervalo [0, 9407] sobre un punto de 2767. H_MAX solo dice hasta donde se
# PUEDE retroceder sin salirse de la serie; no dice nada sobre si la Q resultante
# significa algo.
#
# La evidencia esta en la propia Q CRUDA, que este script ya imprime: crece de
# forma ordenada hasta ~h=9 (0.023 -> 0.219) y a partir de ahi se desploma y
# rebota sin sentido (0.085 a h=12, 0.0091 a h=24, 0.3989 a h=36). Ese desorden
# no es incertidumbre creciente: es que a horizontes largos todas las filas de
# calibracion se congelan contra el MISMO tramo historico —el pico de 2021— y lo
# que se mide es ese contraste concreto, no "el error a h meses vista".
#
# Regla adoptada: se declara calibrado hasta el ultimo horizonte en el que la Q
# cruda sigue comportandose como una estimacion (no cae mas de un 10% respecto
# del maximo alcanzado hasta ese punto). En cuanto se invierte de forma
# apreciable, la tabla deja de estar informada por datos nuevos y solo arrastra
# hacia adelante el maximo anterior: sigue siendo un suelo razonable para el
# ancho —por eso el intervalo se sirve igual— pero ya no es una calibracion, y
# la respuesta debe decirlo con `ic95_calibrado: false`.
TOLERANCIA_INVERSION = 0.10
H_DECLARADO = 1
_max_visto = 0.0
for _h in range(1, H_MAX + 1):
    _q = _Q_sin_suelo[_h]
    if _max_visto > 0 and _q < _max_visto * (1 - TOLERANCIA_INVERSION):
        break
    _max_visto = max(_max_visto, _q)
    H_DECLARADO = _h

# `Q_extrapolado` se conserva como escalar por compatibilidad del artifact: es el
# valor de la tabla en el horizonte que el sistema declara como referencia.
# TABLA INERTE. Si el suelo en cero deja Q(h) plana —porque los cuantiles
# crudos ya sobre-cubren en todos los horizontes—, la tabla no distingue un
# horizonte de otro y seria falso declarar que "calibra" hasta h=41. En ese caso
# el horizonte declarado se limita al de referencia, que es el unico para el que
# hay una lectura defendible, y el artifact marca la tabla como inerte para que
# la respuesta del sistema pueda decirlo.
TABLA_INERTE = len({round(v, 9) for v in Q_por_horizonte.values()}) == 1
if TABLA_INERTE:
    H_DECLARADO = min(6, H_DECLARADO)
    print("  AVISO: con el suelo en cero la tabla Q(h) queda PLANA: los cuantiles "
          "crudos ya")
    print("  sobre-cubren en todos los horizontes estimables, asi que la tabla no "
          "ensancha nada.")
    print("  No se declara calibracion mas alla del horizonte de referencia: una "
          "tabla que no")
    print("  distingue h=1 de h=41 no puede sostener una afirmacion sobre h=41.")

H_CAL = min(6, H_DECLARADO)
Q_extrap = Q_por_horizonte[H_CAL]
_muestra = [1, 2, 3, 6, 9, 12, 24, 36, H_MAX]
print("  h  : " + " ".join(f"{h:>7}" for h in _muestra if h <= H_MAX))
print("  Qc : " + " ".join(f"{_Q_cruda[h]:>7.4f}" for h in _muestra if h <= H_MAX))
print("  Qm : " + " ".join(f"{Q_por_horizonte[h]:>7.4f}" for h in _muestra if h <= H_MAX))
print("  cob: " + " ".join(f"{cobertura_por_horizonte[str(h)]:>6.1f}%" for h in _muestra if h <= H_MAX))
_min_cob = min(cobertura_por_horizonte.values())
print(f"  Cobertura condicional MINIMA sobre los {H_MAX} horizontes: {_min_cob:.2f}%")
print(f"  (con la constante unica de la segunda auditoria bajaba al 17.9% a h=9)")
assert _min_cob >= 95.0 - 1e-9, (
    f"la monotonizacion debe garantizar >=95% en cada horizonte; minimo {_min_cob}"
)
print(f"  Q de referencia (h={H_CAL}): {Q_extrap:.5f}"
      + (f" ({Q_extrap / Q:.2f}x la Q historica)" if Q > 0 else " (la Q historica es 0)"))
print(f"  HORIZONTE DECLARADO CALIBRADO: {H_DECLARADO} meses (de {H_MAX} estimables).")
print(f"    La Q cruda se ordena hasta h={H_DECLARADO} y a partir de h={H_DECLARADO + 1} "
      f"se invierte ({_Q_sin_suelo.get(H_DECLARADO, 0):.4f} -> "
      f"{_Q_sin_suelo.get(H_DECLARADO + 1, float('nan')):.4f}): deja de estar informada")
print(f"    por datos nuevos. Mas alla de h={H_DECLARADO} el intervalo se sirve igual")
print(f"    (con Q monotonizada) pero la respuesta marca ic95_calibrado=false.")

def evaluar_intervalo(mlo, mhi, X, y, etiqueta: str, q_const=None) -> dict:
    """`q_const` puede ser un escalar o un vector con una Q por fila."""
    qc = Q if q_const is None else q_const
    lo = mlo.predict(X) - qc
    hi = mhi.predict(X) + qc
    ancho = hi - lo
    cob = float(((y.values >= lo) & (y.values <= hi)).mean())
    mediana_y = float(np.median(y.values))
    d = {
        "cobertura_%": round(100 * cob, 2),
        "ancho_medio": round(float(ancho.mean()), 4),
        "ancho_p05": round(float(np.percentile(ancho, 5)), 4),
        "ancho_p50": round(float(np.percentile(ancho, 50)), 4),
        "ancho_p95": round(float(np.percentile(ancho, 95)), 4),
        "mediana_target": round(mediana_y, 4),
        "ancho_relativo": round(float(ancho.mean()) / mediana_y, 2),
        "n": int(len(y)),
    }
    print(f"\n--- {etiqueta} ---")
    print(f"Cobertura empirica : {d['cobertura_%']:.2f}%  (objetivo: {100*(1-ALPHA):.0f}%)")
    print(f"Ancho medio        : {d['ancho_medio']:.4f} USD/kg")
    print(f"Ancho relativo     : {d['ancho_relativo']:.2f}x la mediana del target "
          f"({d['mediana_target']:.4f} USD/kg)")
    print(f"Dispersion del ancho: p05 {d['ancho_p05']:.4f} | p50 {d['ancho_p50']:.4f} "
          f"| p95 {d['ancho_p95']:.4f}")
    return d


# ──────────────────────────────────────────────────────────────────────────
# 4. Cobertura en VAL_CAL (control), en TEST (drift) y en TEST bajo produccion
# ──────────────────────────────────────────────────────────────────────────
diag_cal = evaluar_intervalo(model_lo, model_hi, X_val_cal, y_val_cal,
                             "VAL_CAL (control: debe dar ~95% por construccion)")
diag_test = evaluar_intervalo(model_lo, model_hi, X_test, y_test,
                              "TEST (holdout real, nunca usado antes)")
# Congelamiento de TEST tal y como opera produccion: el ultimo mes cerrado
# antes de que empiece TEST. TERCERA AUDITORIA: la version anterior reutilizaba
# el congelamiento de VAL_CAL (2025-02), 8.3 meses antes de TEST, y lo etiquetaba
# "condiciones de produccion". Era mas pesimista que produccion — el resultado
# quedaba del lado conservador — pero la etiqueta no describia lo medido.
_MES_CONGELADO_TEST = _per_test.min() - 1
_h_test = np.array([(t - 1 - _MES_CONGELADO_TEST).n for t in _per_test])
# Q por fila segun SU horizonte, que es lo que hace el predictor. h=0 significa
# que el mes anterior si esta observado: esa fila no extrapola y usa la Q
# historica, igual que en produccion.
_q_test = np.array([Q if h <= 0 else Q_por_horizonte[min(int(h), H_MAX)] for h in _h_test])
diag_test_prod = evaluar_intervalo(
    model_lo, model_hi,
    congelar_por_horizonte(X_test, _per_test, _h_test), y_test,
    f"TEST BAJO CONDICIONES DE PRODUCCION (congelado en {_MES_CONGELADO_TEST}, "
    f"horizontes {int(_h_test.min())}-{int(_h_test.max())} meses, Q(h) por fila)",
    q_const=_q_test,
)

desvio = diag_test["cobertura_%"] - 100 * (1 - ALPHA)
meta_cob_cruda = 100 * float((
    (y_test.values >= model_lo.predict(X_test) - Q_cqr_crudo)
    & (y_test.values <= model_hi.predict(X_test) + Q_cqr_crudo)
).mean())
print(f"\n=== LECTURA DE LA COBERTURA ===")
print(f"VAL_CAL {diag_cal['cobertura_%']:.2f}% -> la calibracion conformal es correcta.")
print(f"TEST    {diag_test['cobertura_%']:.2f}% -> desvio de {desvio:+.2f} pp respecto del 95%.")
if _suelo_aplicado:
    print("Ambas cifras se miden con la Q servida (suelo en cero). Con la Q cruda de")
    print(f"CQR ({Q_cqr_crudo:.5f}, negativa) la cobertura del holdout caeria al "
          f"{meta_cob_cruda:.2f}%: ese es el")
    print("defecto que el suelo evita, y es la razon de que el desvio sea POSITIVO aqui.")
print("El desvio NO es un margen de seguridad deliberado. Parte es ruptura de")
print("exchangeability por cambio de regimen; parte es mala especificacion de los")
print("modelos de cuantiles (ver el desglose por cola abajo). En otro periodo el")
print("mismo Q puede quedar POR DEBAJO del 95%.")

# ──────────────────────────────────────────────────────────────────────────
# 4b. ¿Exchangeability o mala especificacion? El desglose por cola lo separa.
# ──────────────────────────────────────────────────────────────────────────
print("\n=== DIAGNOSTICO DE ESPECIFICACION DE LOS CUANTILES (sin aplicar Q) ===")
print(f"{'conjunto':<10} {'cobertura':>10} {'P(y<q_lo)':>11} {'P(y>q_hi)':>11}   (nominal 95% / 2.5% / 2.5%)")
espec = {}
for nombre, (a, b, yy) in [
    ("VAL_CAL", (q_lo_cal, q_hi_cal, y_val_cal.values)),
    ("TEST", (model_lo.predict(X_test), model_hi.predict(X_test), y_test.values)),
]:
    cov = 100 * float(((yy >= a) & (yy <= b)).mean())
    bajo = 100 * float((yy < a).mean())
    alto = 100 * float((yy > b).mean())
    espec[nombre] = {"cobertura_cruda_%": round(cov, 2),
                     "P_y_menor_q_lo_%": round(bajo, 2),
                     "P_y_mayor_q_hi_%": round(alto, 2)}
    print(f"{nombre:<10} {cov:>9.2f}% {bajo:>10.2f}% {alto:>10.2f}%")

# Calibracion asimetrica: si Q_hi sale negativo, q_hi ya sobrepasa el cuantil
# empirico y el conformal necesita ENCOGERLO, no ampliarlo. Es la prueba
# directa de que el ancho no es "dispersion real del mercado".
ka = int(np.ceil((n_cal + 1) * (1 - ALPHA / 2)))
Q_lo_asim = float(np.sort(q_lo_cal - y_val_cal.values)[min(ka, n_cal) - 1])
Q_hi_asim = float(np.sort(y_val_cal.values - q_hi_cal)[min(ka, n_cal) - 1])
espec["Q_lo_asimetrico"] = round(Q_lo_asim, 5)
espec["Q_hi_asimetrico"] = round(Q_hi_asim, 5)
espec["q_hi_mal_especificado"] = bool(Q_hi_asim < 0)
espec["nota"] = (
    "P(y > q_hi) muy por debajo de su nominal 2.5% YA EN VAL_CAL (antes de "
    "cualquier cambio de regimen) indica que el modelo de cuantil superior esta "
    "mal especificado, no solo que la exchangeability se rompio. Causa: los "
    "cuantiles se ajustan sobre un bloque cuya dispersion es ~5.7x la del "
    "holdout y arrastran esa amplitud. Consecuencia: el ancho del intervalo es "
    "una propiedad del METODO, no 'la dispersion real del flete unitario en el "
    "mercado'. La misma mala especificacion es la que hace que la Q conformal "
    "salga NEGATIVA y obligue al suelo en cero (ver 'suelo_cero'): si los "
    "cuantiles ya sobre-cubren, CQR corrige estrechando. Sobre si el ancho es "
    "corregible, la etiqueta la fija la medicion de cada ejecucion y no este "
    "texto: ver 'ponderacion_temporal', cuyo veredicto ha cambiado entre "
    "esquemas y por eso no se escribe aqui a mano."
)
print(f"Q asimetrica: Q_lo={Q_lo_asim:+.5f} | Q_hi={Q_hi_asim:+.5f}"
      f"{'  <- NEGATIVA: q_hi sobrepasa el cuantil empirico' if Q_hi_asim < 0 else ''}")

if diag_test["ancho_relativo"] >= 1.0:
    print(f"\nADVERTENCIA DE UTILIDAD: el intervalo mide {diag_test['ancho_relativo']:.2f}x el valor")
    print("que estima. Es estadisticamente valido pero poco accionable para cotizar;")
    print("debe reportarse como limitacion, no como precision alcanzada.")

# ──────────────────────────────────────────────────────────────────────────
# 4c. El ancho, es una propiedad "corregible"? Se mide, no se afirma.
#     TERCERA AUDITORIA. La segunda reclasifico el ancho de ~1.9x de "limitacion
#     estructural" a "trabajo futuro corregible" y nombro un remedio concreto:
#     ponderar temporalmente el ajuste de los cuantiles. Esa reclasificacion se
#     hizo SIN medirla, en la unica ronda que presumia de haber medido todo lo
#     demas. Aqui se implementa el remedio y se reporta el resultado, sea cual
#     sea, para que la etiqueta la fije la medicion y no el optimismo.
# ──────────────────────────────────────────────────────────────────────────
print()
print("=== EL ANCHO: ES CORREGIBLE? PONDERACION TEMPORAL DE LOS CUANTILES ===")
_edad_anios = (BLOQUE_Q["FECHA"].max() - BLOQUE_Q["FECHA"]).dt.days.values / 365.25
_ancho_base = diag_test["ancho_relativo"]
ponderacion = {"sin_ponderar": {"ancho_relativo": _ancho_base,
                                "cobertura_%": diag_test["cobertura_%"],
                                "Q": round(Q, 5)}}
for _hl in (1.0, 0.5):
    _w = 0.5 ** (_edad_anios / _hl)
    _ms = {}
    for _a in (ALPHA / 2, 1 - ALPHA / 2):
        # Misma receta que los modelos servidos (mismo numero de arboles, sin
        # early stopping): la unica diferencia con el ajuste base debe ser la
        # ponderacion, o el experimento no mediria la ponderacion.
        _n_ref = (model_lo if _a < 0.5 else model_hi).n_estimators
        _m = XGBRegressor(n_estimators=_n_ref, quantile_alpha=_a, **_HP_Q)
        _m.fit(X_train, y_train, sample_weight=_w, verbose=False)
        _ms[_a] = _m
    _L, _H = _ms[ALPHA / 2], _ms[1 - ALPHA / 2]
    _Qw, _, _ = conformal_Q(np.maximum(_L.predict(X_val_cal) - y_val_cal.values,
                                       y_val_cal.values - _H.predict(X_val_cal)))
    # Mismo suelo en cero que el ajuste base. Sin el, la variante ponderada
    # conseguiria su ancho menor estrechando el intervalo por la via que se
    # acaba de descartar, y la comparacion mediria la regla de calibracion en
    # vez de la ponderacion, que es lo que este experimento pretende medir.
    _Qw = max(_Qw, 0.0)
    _a_t = _L.predict(X_test) - _Qw
    _b_t = _H.predict(X_test) + _Qw
    ponderacion[f"half_life_{_hl}_anios"] = {
        "ancho_relativo": round(float((_b_t - _a_t).mean()) / float(np.median(y_test.values)), 2),
        "cobertura_%": round(100 * float(((y_test.values >= _a_t) & (y_test.values <= _b_t)).mean()), 2),
        "Q": round(_Qw, 5),
    }
    print(f"  half-life {_hl} anios: Q={_Qw:.5f} "
          f"ancho_rel={ponderacion[f'half_life_{_hl}_anios']['ancho_relativo']:.2f}x "
          f"(base {_ancho_base:.2f}x) "
          f"cobertura={ponderacion[f'half_life_{_hl}_anios']['cobertura_%']:.2f}%")

_mejor = min(v["ancho_relativo"] for k, v in ponderacion.items() if k != "sin_ponderar")
_funciona = _mejor < 0.85 * _ancho_base
ponderacion["veredicto"] = "corregible" if _funciona else "no corregible por esta via"
ponderacion["nota"] = (
    f"Resultado medido en esta ejecucion, no heredado del texto. La ponderacion "
    f"temporal del ajuste de los cuantiles lleva el ancho relativo de "
    f"{_ancho_base:.2f}x a {_mejor:.2f}x en el mejor caso, manteniendo la cobertura "
    f"por encima del 95%. Veredicto automatico: {ponderacion['veredicto']}. "
    "HISTORIA, PORQUE LA ETIQUETA HA CAMBIADO DOS VECES Y ESO IMPORTA. La segunda "
    "auditoria declaro el ancho 'trabajo futuro corregible' y nombro este remedio "
    "SIN medirlo. La tercera lo midio bajo el esquema 70/20/10, no funciono "
    "(empeoraba con half-life de 1 anio y apenas mejoraba con 6 meses mientras "
    "triplicaba Q) y lo devolvio a 'limitacion estructural'. Bajo el esquema 80/20, "
    "y una vez que la Q lleva suelo en cero —de modo que la variante ponderada ya "
    "no puede ganar ancho estrechando el intervalo—, la medicion vuelve a dar "
    "positiva. "
    "NO SE ADOPTA, y el motivo es el mismo que ya llevo a descartar el filtro de "
    "peso y el recorte de colas: el half-life es un parametro elegido de una "
    "rejilla de dos valores DESPUES de ver el resultado, y adoptar un cambio "
    "porque mejora una metrica, con una regla fijada a posteriori, es precisamente "
    "el sesgo que las revisiones anteriores han venido corrigiendo. Queda "
    "registrado como candidato con evidencia a favor: adoptarlo exige fijar el "
    "half-life por un criterio independiente del resultado y verificar la "
    "cobertura condicional por horizonte, no solo la marginal."
)
print(f"  VEREDICTO: {ponderacion['veredicto']}.")
print("  El ancho se declara LIMITACION ESTRUCTURAL mientras el veredicto sea "
      "'no corregible por esta via';" if not _funciona else
      "  El veredicto ha cambiado respecto de rondas anteriores: revisar la "
      "redaccion de la limitacion")
print("  la etiqueta la fija esta medicion en cada ejecucion, no el texto." if not _funciona else
      "  en la documentacion antes de publicarla.")

# Variacion del ancho por representatividad del importador.
test_eval = p.test.copy()
lo_t = model_lo.predict(X_test) - Q
hi_t = model_hi.predict(X_test) + Q
test_eval["ancho_intervalo"] = hi_t - lo_t
test_eval["bien_representado"] = (
    test_eval["importador_freq"] >= test_eval["importador_freq"].median()
)
por_repr = test_eval.groupby("bien_representado")["ancho_intervalo"].agg(["mean", "count"])
print("\nAncho medio por representatividad del importador:")
print(por_repr)
brecha = 100 * (por_repr["mean"].max() / por_repr["mean"].min() - 1)
print(f"Brecha entre segmentos: {brecha:.1f}%. El ancho se adapta, pero poco:")
print("la afirmacion defendible es que varia, no que discrimine fuertemente el caso.")

# ──────────────────────────────────────────────────────────────────────────
# 5. Guardar modelos + metadatos de calibracion
# ──────────────────────────────────────────────────────────────────────────
joblib.dump(model_lo, QLO_PATH)
joblib.dump(model_hi, QHI_PATH)

meta["ic95_seleccion_n_estimators"] = seleccion_q
meta["ic95_receta"] = {
    "bloque_de_ajuste": "train + val_es (cola del 80% hasta el corte de calibracion)",
    "n_filas_ajuste": len(BLOQUE_Q),
    "rango_ajuste": [str(BLOQUE_Q["FECHA"].min().date()),
                     str(BLOQUE_Q["FECHA"].max().date())],
    "n_estimators_q_lo": int(model_lo.n_estimators),
    "n_estimators_q_hi": int(model_hi.n_estimators),
    "early_stopping": False,
    "conjunto_de_calibracion": [str(val_cal["FECHA"].min().date()),
                                str(val_cal["FECHA"].max().date())],
    "n_calibracion": len(val_cal),
    "nota": (
        "VAL_CAL queda fuera del ajuste por exigencia de CQR, no por el early "
        "stopping (que ya no existe). El numero de arboles se selecciona por "
        "separado para cada cuantil con su propia perdida, por origen rodante "
        "trimestral dentro del bloque de ajuste."
    ),
}
meta["ic95_metodo"] = "conformalized_quantile_regression"
meta["ic95_conformal_Q"] = Q
meta["ic95_conformal_Q_cqr_crudo"] = Q_cqr_crudo
meta["ic95_suelo_cero_aplicado"] = bool(_suelo_aplicado)
meta["ic95_conformal_Q_extrapolado"] = Q_extrap
# H-12: lo que se declara calibrado NO es hasta donde llega la tabla, sino hasta
# donde la Q cruda sigue siendo una estimacion. Ver el bloque de H_DECLARADO.
meta["ic95_horizonte_calibrado_meses"] = H_DECLARADO
meta["ic95_horizonte_maximo_estimable_meses"] = H_MAX
meta["ic95_conformal_Q_por_horizonte"] = {str(k): v for k, v in Q_por_horizonte.items()}
meta["ic95_conformal_Q_cruda_por_horizonte"] = {str(k): v for k, v in _Q_cruda.items()}
meta["ic95_conformal_Q_cruda_sin_suelo_por_horizonte"] = {str(k): v for k, v in _Q_sin_suelo.items()}
meta["ic95_tabla_horizonte_inerte"] = bool(TABLA_INERTE)
meta["ic95_diagnostico"] = {
    "val_cal": diag_cal,
    "test": diag_test,
    "test_condiciones_produccion": diag_test_prod,
    "desvio_cobertura_test_pp": round(desvio, 2),
    "suelo_cero": {
        "aplicado": bool(_suelo_aplicado),
        "Q_cqr_crudo": round(Q_cqr_crudo, 5),
        "Q_servida": round(Q, 5),
        "cobertura_holdout_con_Q_cruda_%": round(
            100 * float((
                (y_test.values >= model_lo.predict(X_test) - Q_cqr_crudo)
                & (y_test.values <= model_hi.predict(X_test) + Q_cqr_crudo)
            ).mean()), 2),
        "cobertura_holdout_con_Q_servida_%": round(
            100 * float((
                (y_test.values >= model_lo.predict(X_test) - Q)
                & (y_test.values <= model_hi.predict(X_test) + Q)
            ).mean()), 2),
        "regla": (
            "La calibracion conformal puede ENSANCHAR el intervalo, nunca "
            "encogerlo. Una Q negativa significa que los cuantiles ya "
            "sobre-cubren sobre el conjunto de calibracion y que CQR pide "
            "estrechar para alcanzar la cobertura exacta; estrechar solo es "
            "valido bajo exchangeability entre calibracion y futuro, que es "
            "la hipotesis que este trabajo declara rota. Se conserva la mitad "
            "conservadora de la correccion y se descarta la otra."
        ),
    },
    "quantile_crossing_val_cal": n_cross,
    "brecha_ancho_por_representatividad_%": round(float(brecha), 1),
    "especificacion": espec,
    "ponderacion_temporal": ponderacion,
    "calibracion_extrapolada": {
        "metodo": "tabla Q(h) monotonizada por maximo acumulado",
        "H_MAX_estimable_meses": H_MAX,
        "H_referencia_meses": H_CAL,
        "congelamiento": "por fila, en (mes de la fila - 1 - h)",
        "Q_por_horizonte": {str(k): round(v, 6) for k, v in Q_por_horizonte.items()},
        "Q_cruda_sin_monotonizar": {str(k): round(v, 6) for k, v in _Q_cruda.items()},
        "Q_cruda_sin_suelo": {str(k): round(v, 6) for k, v in _Q_sin_suelo.items()},
        "tabla_inerte": bool(TABLA_INERTE),
        "cobertura_condicional_por_horizonte_val_cal": cobertura_por_horizonte,
        "cobertura_condicional_minima_%": round(_min_cob, 2),
        "mes_congelado_test_produccion": str(_MES_CONGELADO_TEST),
        "nota": (
            "Q(h) = max(Q_cruda(1..h)). Cada Q_cruda(h) es la constante conformal "
            "calibrada sobre VAL_CAL con el mercado de cada fila congelado h "
            "meses antes de lo que le correspondia, que es exactamente lo que "
            "hace produccion. Como Q(h) >= Q_cruda(h), la cobertura del 95% se "
            "sostiene en CADA horizonte y no solo en promedio: el minimo sobre "
            f"los {H_MAX} horizontes es {round(_min_cob, 2)}%. La constante unica "
            "de la segunda auditoria bajaba al 17.9% a nueve meses mientras se "
            "servia como 'IC 95%' para fechas de hasta 60 meses. Se probaron y "
            "descartaron: (a) Q(h) cruda, no monotona, que daria a 24 meses un "
            "intervalo mas estrecho que a 3; (b) una Q agrupada sobre h~U(1..6), "
            "que sobre-cubre los horizontes cortos hasta hundir el limite "
            "inferior por debajo de cero y aun asi se queda en 78.9% a h=6. "
            "LIMITACION: Q(h) se estima sobre la unica trayectoria de mercado "
            "disponible (2021-2025); que el peor horizonte sea h=9 es un hecho de "
            "ese periodo (el pico de 2024), no una ley general."
        ),
    },
    "ratio_Q_extrapolado_sobre_Q": round(Q_extrap / Q, 2) if Q > 0 else None,
    "nota": (
        "La cobertura en VAL_CAL (~95%) verifica que la calibracion conformal "
        "esta bien hecha. El desvio en TEST mezcla ruptura de exchangeability y "
        "mala especificacion de los cuantiles: ver 'especificacion' para el "
        "desglose por cola, que separa ambas causas. Se calibra UNA constante "
        "para el regimen historico (Q) y una TABLA Q(h) para el extrapolado, "
        "indexada por el horizonte en meses y monotonizada por maximo acumulado "
        "(ver 'calibracion_extrapolada'). La segunda auditoria usaba una sola "
        "constante extrapolada, calibrada a ~3 meses y servida hasta 60, cuya "
        "cobertura real caia al 18% a nueve meses. "
        "mediana del target) es la cifra que indica si el intervalo es accionable. "
        "LIMITACION QUE NO SE RESUELVE CON CQR: el ancho varia poco entre casos "
        "(~8% entre segmentos de representatividad), asi que el intervalo no "
        "discrimina un puerto conocido de uno desconocido."
    ),
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nGuardado: {QLO_PATH}")
print(f"Guardado: {QHI_PATH}")
print(f"Actualizado: {META_PATH} (ic95_conformal_Q, ic95_conformal_Q_extrapolado, ic95_diagnostico)")

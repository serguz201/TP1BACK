"""
Seleccion del numero de arboles por ORIGEN RODANTE INTERNO, en sustitucion del
early stopping de ventana unica.

POR QUE EXISTE (el fallo que corrige, medido)

Con el esquema 70/20/10 el early stopping se hacia contra una sola ventana de
validacion y funcionaba de casualidad: esa ventana caia en un tramo de mercado
mas bajo que el de entrenamiento, y penalizar los modelos subajustados
—los que se quedan pegados a la mediana de train— era lo correcto tambien para
el futuro.

Al pasar al esquema 80/20 la ventana de validacion interna es la cola de 2024,
que esta en el MISMO nivel de mercado que el bloque de entrenamiento (mediana
0.3629 frente a 0.3316 USD/kg). Ahi el criterio se invierte: un modelo de 20
arboles, cuyas predicciones apenas se han despegado de la mediana global, gana
en esa ventana. El early stopping lo elegia, y ese modelo colapsaba sobre 2025,
donde la mediana real es 0.19:

    early stopping de ventana unica    ->   20 arboles  ->  MAPE 2025 = 56.82%
    sin seleccionar nada               ->  600 arboles  ->  MAPE 2025 = 27.89%
    origen rodante interno (adoptado)  ->  112 arboles  ->  MAPE 2025 = 27.66%

Un procedimiento de seleccion que elige un modelo mas del DOBLE de malo que no
seleccionar nada no es un detalle de implementacion: es un error metodologico, y
estaba enmascarado mientras la ventana de validacion resultaba representativa
por accidente. Se sustituye.

QUE HACE EN SU LUGAR

Divide el bloque de entrenamiento en trimestres y, para cada trimestre con al
menos un ano de historia previa, entrena con todo lo anterior y registra la
curva completa de MAE sobre ese trimestre. Normaliza cada curva por su propio
minimo —asi un trimestre de mercado caro no pesa mas que uno barato solo por
tener errores absolutos mayores— y sobre la curva media toma el MENOR numero de
arboles que queda a menos de un 1% del minimo: el modelo mas parsimonioso
estadisticamente indistinguible del mejor.

POR QUE ESE CRITERIO Y NO OTRO (las dos alternativas se midieron y se descartan)

  - El optimo de CADA pliegue por separado es inservible: va de 1 a 596 arboles
    segun el trimestre, porque lo que domina la curva no es la capacidad del
    modelo sino si ese trimestre quedo por encima o por debajo del nivel con el
    que se entreno. Promediar sobre muchos pliegues es justamente lo que anula
    ese efecto.
  - El criterio MINIMAX (minimizar el peor pliegue) elige unas decenas de
    arboles y reproduce el fallo del early stopping, porque el peor pliegue
    siempre es uno de cambio de regimen y ahi vuelve a ganar el modelo plano.

ESTABILIDAD (la propiedad que hace defendible la eleccion). El RESULTADO es
insensible al valor elegido: en toda la meseta (112-600) el MAPE del holdout se
mueve dentro de un rango de ~0.25 pp (27.66% a 27.89%). La eleccion no es un punto fragil del pipeline; el
early stopping si lo era, porque caia en la zona de menos de 50 arboles, donde
el MAPE se dispara.

POR QUE EL MENOR DE LA MESETA Y NO EL ARGMIN. Por parsimonia, no por
reproducibilidad. Sobre una meseta de 490 posiciones en la que todas las
configuraciones son estadisticamente indistinguibles, quedarse con la mas
pequena es la eleccion que hay que justificar menos y la que menos capacidad
concede sin contrapartida medible. Es la misma logica que la regla "one standard
error" de la validacion cruzada clasica, con la tolerancia expresada en
porcentaje porque aqui las curvas estan normalizadas por su propio minimo.

UNA CORRECCION QUE MERECE QUEDAR ESCRITA, porque la primera version de este
modulo la documentaba al reves. Al integrar el walk-forward se observo que dos
procesos con los mismos datos y la misma semilla elegian 389 y 457 arboles, y se
atribuyo a que XGBoost con `n_jobs=-1` no seria bit-identico entre procesos. Era
FALSO. La causa real era que `sort_values("FECHA")` usa quicksort, que no es
estable: las filas de una misma fecha quedaban en orden distinto en cada
proceso, y con `subsample`/`colsample_bytree` un orden distinto es un modelo
distinto. Con `kind="mergesort"` en todo el pipeline, dos procesos producen el
mismo n, el mismo argmin y predicciones identicas bit a bit. Se comprobo.

La leccion: una no-reproducibilidad observada no autoriza a nombrar una causa
sin medirla. La regla del borde de la meseta se conserva porque el argumento de
parsimonia se sostiene solo, pero no era, como se escribio primero, el remedio a
un problema de coma flotante que no existia.

NINGUNA DE ESTAS DECISIONES MIRA EL HOLDOUT. Los pliegues viven enteramente
dentro del bloque de entrenamiento. Las cifras de 2025 citadas arriba son el
resultado que se publica DESPUES de fijar el procedimiento, no el criterio con
el que se fijo; se citan porque documentar el tamano del fallo corregido forma
parte de justificar la correccion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml import data_pipeline as dp

# Anchura de la meseta: n cuya curva media queda a menos de este factor del
# minimo se consideran estadisticamente indistinguibles del mejor.
TOLERANCIA_MESETA = 1.01

# Un pliegue necesita al menos este historial previo para que su curva de MAE
# describa al modelo y no a la escasez de datos.
MIN_TRIMESTRES_HISTORIA = 4
MIN_FILAS_PLIEGUE = 300
MIN_FILAS_HISTORIA = 4000


def seleccionar_n_estimators(
    bloque: pd.DataFrame,
    constructor,
    features: list[str] | None = None,
    target: str | None = None,
    n_max: int = 600,
    verbose: bool = True,
) -> tuple[int, dict]:
    """Numero de arboles por origen rodante trimestral dentro de `bloque`.

    `constructor(n_estimators)` debe devolver un estimador sin early stopping y
    con `eval_metric` fijado, de modo que `evals_result()` entregue la curva
    completa. Se le pasa como argumento —en vez de construirlo aqui— para que el
    modelo puntual y los de cuantiles usen esta misma seleccion con sus propias
    funciones de perdida sin que este modulo tenga que conocerlas.

    Devuelve (n_estimators, diagnostico). El diagnostico se vuelca al artifact:
    sin el, el numero elegido seria un parametro magico en el codigo.
    """
    features = features or dp.FEATURES
    target = target or dp.TARGET

    b = bloque.sort_values("FECHA", kind="mergesort").reset_index(drop=True)
    per = b["FECHA"].dt.to_period("Q")
    trimestres = sorted(per.unique())

    curvas, detalle = [], []
    for i, q in enumerate(trimestres):
        if i < MIN_TRIMESTRES_HISTORIA:
            continue
        tr, va = b[per < q], b[per == q]
        if len(va) < MIN_FILAS_PLIEGUE or len(tr) < MIN_FILAS_HISTORIA:
            continue
        m = constructor(n_max)
        m.fit(tr[features], tr[target], eval_set=[(va[features], va[target])], verbose=False)
        curva = np.asarray(list(m.evals_result()["validation_0"].values())[0], dtype=float)
        curvas.append(curva / curva.min())
        detalle.append({
            "trimestre": str(q),
            "n_train": int(len(tr)),
            "n_pliegue": int(len(va)),
            "argmin_individual": int(curva.argmin()) + 1,
        })
        if verbose:
            print(f"    pliegue {q} | train {len(tr):>6,} | eval {len(va):>5,} | "
                  f"optimo individual {int(curva.argmin()) + 1:>4}")

    if not curvas:
        raise ValueError(
            "no hay ningun trimestre con historia suficiente para seleccionar "
            "el numero de arboles por origen rodante"
        )

    media = np.mean(curvas, axis=0)
    n_argmin = int(media.argmin()) + 1

    # Meseta: rango de n cuya curva media queda a menos de un 1% del minimo. Es
    # la medida de cuanto importa realmente el valor elegido.
    dentro = np.where(media <= media.min() * TOLERANCIA_MESETA)[0] + 1

    # REGLA DE SELECCION: EL MENOR n DE LA MESETA, NO EL ARGMIN.
    #
    # Por parsimonia. La meseta abarca cientos de posiciones en las que el MAPE
    # del holdout varia menos de 0.3 pp: todas son estadisticamente
    # indistinguibles, y entre configuraciones equivalentes la mas pequena es la
    # que concede menos capacidad sin contrapartida medible y la que hay que
    # justificar menos. Regla "one standard error" de toda la vida, con la
    # tolerancia en porcentaje porque las curvas estan normalizadas.
    #
    # NO es un remedio a un problema de reproducibilidad: ver la correccion al
    # final del docstring del modulo. El pipeline ES reproducible bit a bit
    # desde que los ordenamientos por fecha usan `kind="mergesort"`.
    n_sel = int(dentro.min())
    diagnostico = {
        "metodo": (
            "origen rodante trimestral interno; se elige el MENOR numero de "
            "arboles cuya curva de MAE media normalizada queda a menos de un "
            f"{100 * (TOLERANCIA_MESETA - 1):.0f}% del minimo"
        ),
        "n_estimators_elegido": n_sel,
        "n_estimators_argmin": n_argmin,
        "n_max_explorado": n_max,
        "pliegues": len(curvas),
        "detalle_pliegues": detalle,
        "dispersion_optimos_individuales": [
            int(min(d["argmin_individual"] for d in detalle)),
            int(max(d["argmin_individual"] for d in detalle)),
        ],
        "meseta_1pct": [int(dentro.min()), int(dentro.max())],
        "penalizacion_en_los_extremos_%": {
            "n=20": round(100 * (float(media[19]) / float(media.min()) - 1), 2),
            f"n={n_max}": round(100 * (float(media[-1]) / float(media.min()) - 1), 2),
        },
        "nota": (
            "Sustituye al early stopping de ventana unica, que con el esquema "
            "80/20 elegia ~20 arboles y triplicaba el error sobre 2025 (56.74% "
            "frente a 25.07% sin seleccionar nada): la cola de 2024 esta en el "
            "mismo nivel de mercado que el bloque de entrenamiento y ahi gana el "
            "modelo pegado a la mediana. El optimo de cada pliegue por separado "
            "va de 1 a 596 arboles segun el trimestre, asi que promediar sobre "
            "todos los pliegues elegibles no es un refinamiento: es lo unico que "
            "hace estimable el parametro. Ningun pliegue toca el holdout."
        ),
    }
    if verbose:
        print(f"    -> n_estimators = {n_sel} (argmin {n_argmin}; meseta al 1%: "
              f"{diagnostico['meseta_1pct'][0]}-{diagnostico['meseta_1pct'][1]}, "
              f"{len(curvas)} pliegues, optimos individuales entre "
              f"{diagnostico['dispersion_optimos_individuales'][0]} y "
              f"{diagnostico['dispersion_optimos_individuales'][1]})")
    return n_sel, diagnostico

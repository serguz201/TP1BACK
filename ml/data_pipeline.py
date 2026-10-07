"""
Pipeline de datos compartido por train_model.py y train_quantile_models.py.

Existe para que el modelo puntual y los modelos de cuantiles no puedan divergir:
ambos importan de aqui el mismo filtro de alcance, el mismo recorte de outliers,
las mismas particiones y el mismo feature engineering. Cualquier cambio
metodologico se hace una sola vez, en este archivo.

Decisiones metodologicas implementadas aqui (todas verificables en la salida de
`diagnostico_recorte()` y en los prints de los scripts):

  1. PARTICION POR FECHA, NO POR INDICE. Los cortes se redondean al limite de
     dia siguiente, de modo que ninguna fecha aparece en dos particiones.
     Cortar por posicion dejaba filas del mismo dia (y por tanto de la misma
     nave/declaracion) a ambos lados de la frontera.

     ESQUEMA VIGENTE: "80_20" (ver `ESQUEMA_DEFAULT`). El corte principal es un
     ANIO CALENDARIO, no un cuantil: TRAIN = 2021-2024, HOLDOUT = todo 2025.
     Es lo que pidio la revision metodologica y tiene dos ventajas sobre el
     70/20/10 anterior: (a) la frontera es interpretable y reproducible sin
     recalcular cuantiles cada vez que crece el corpus, y (b) el holdout cubre
     los DOCE meses de un anio completo, de modo que la evaluacion no depende
     del regimen de mercado de un tramo corto y arbitrario (el TEST del
     esquema anterior eran 5 meses excepcionalmente calmos, y por eso su R2 no
     era interpretable). El esquema "70_20_10" se conserva intacto y
     seleccionable para poder reproducir los resultados historicos.

     SUBPARTICION INTERNA DEL 80% (esto es lo que hace que el holdout sea
     honesto). El 80% de entrenamiento se corta a su vez por fecha en un 80/20
     anidado. La cabeza ajusta los encoders, las medianas y los umbrales de
     recorte; la cola sirve para calibrar el intervalo conformal, que exige
     datos no vistos por los modelos de cuantiles. Consecuencia buscada:
     NINGUNA fila de 2025 interviene en ajuste, calibracion, seleccion de
     umbrales ni encoders. 2025 solo se lee para medir. La alternativa —usar
     2025 como conjunto de ajuste intermedio, que es la lectura literal de
     "validacion"— habria contaminado justo el conjunto sobre el que se
     reportan las metricas.

     EVALUACION QUE ACOMPANA AL ESQUEMA: el holdout 2025 da UNA foto (un
     modelo congelado en 2024-12 mirando doce meses hacia adelante, con hasta
     11 meses de horizonte). `ml/walk_forward.py` da la otra: reentrena mes a
     mes durante 2025 (entrena hasta diciembre-2024 y predice enero, entrena
     hasta enero y predice febrero, y asi los doce meses), que es como opera
     realmente un sistema reentrenado con periodicidad mensual. Las dos cifras
     deben leerse juntas: la primera acota el coste de NO reentrenar, la
     segunda el rendimiento esperable SI se reentrena.

  2. RECORTE DE OUTLIERS AJUSTADO SOLO CON TRAIN. Los percentiles P0.5-P99.5 se
     calculan usando exclusivamente el tramo de entrenamiento y luego se aplican
     a val/test. Calcularlos sobre el historico completo era una fuga (leve pero
     real) del futuro hacia la limpieza del pasado.

  3. RECORTE ESTRATIFICADO POR REGIMEN DE RUTA. Los percentiles se calculan por
     separado para embarques directos y de transbordo. Con un recorte global,
     el transbordo (4% de los datos, distribucion de flete distinta) perdia el
     5.60% de sus filas contra el 0.80% de los directos — un sesgo de 7x contra
     el segmento minoritario que la feature `ruta_directa` existe para modelar.

  4. `ruta_directa` SIEMPRE VIA LOOKUP DE PUERTO, TAMBIEN COMO CLAVE DE LIMPIEZA.
     El lookup se ajusta solo con train y se aplica igual a train, val, test y
     produccion. Antes, el entrenamiento usaba el valor real por fila
     (CPAIS_PROC == CPAIS) mientras que produccion usaba el lookup: las metricas
     se median con una ruta de computo que el sistema desplegado no puede
     reproducir.

     SEGUNDA AUDITORIA: ademas, el recorte del punto 3 estratificaba por el
     valor REAL mientras que el modelo recibia el valor del LOOKUP. Una fila
     podia limpiarse como transbordo (umbral 2.3411) y modelarse como directa.
     Eso dejo entrar una fila de 9.49 kg embarcada en HAMBURG con
     FLETE_UNIT = 2.2550 que por si sola producia el 14.75% del MSE de TEST.
     Ahora la clave de estratificacion ES la feature: se construye el lookup
     antes del recorte y se estratifica por el mismo valor que vera el modelo.

LIMITACION DECLARADA (no es un bug, es una propiedad del diseno): la serie
mensual de mercado que alimenta `mercado_lag1/2/3` y `mercado_ma3` se construye
sobre todo el periodo. Los rezagos solo miran meses estrictamente anteriores
(`shift`), asi que no hay look-ahead temporal — pero la media de un mes que
contiene filas de val/test se calcula con esas filas. Es informacion observable
en produccion (el mes anterior ya esta cerrado cuando se cotiza), por lo que se
mantiene; el diseno es transductivo y debe declararse como tal. Dos salvedades
que la segunda auditoria pidio explicitar:

  - Ningun DIA se reparte entre particiones. En el esquema 70_20_10 el MES
    2025-08 si caia en VAL y TEST a la vez, y el mes es la unidad en la que
    operan los rezagos. El esquema 80_20 ELIMINA ese caso en la frontera que
    importa: el corte 2021-2024 / 2025 es un limite de anio calendario, asi
    que ningun mes se reparte entre el 80% y el holdout. Subsiste, menor, en
    la frontera INTERNA del 80% (train / val interno), que si cae a mitad de
    mes; no afecta a la honestidad del holdout, solo al tamano efectivo del
    set de calibracion.
  - El argumento "el mes anterior ya esta cerrado" supone disponibilidad
    inmediata de la estadistica aduanera. SUNAT publica con rezago, asi que en
    produccion real los rezagos pueden llegar con un mes mas de retraso del que
    supone el entrenamiento. El escenario congelado de train_model.py acota ese
    riesgo por el extremo pesimista.

EXPERIMENTOS NEGATIVOS (probados y descartados; se documentan para que no se
vuelvan a proponer sin datos).

AVISO DE VIGENCIA: las cifras concretas de esta lista se midieron bajo el
esquema 70/20/10, cuyo TEST eran cinco meses de 2025, y NO son comparables con
las del esquema 80/20 vigente, cuyo holdout es el ano entero. Lo que sigue
vigente es la DECISION y su motivo —que es para lo que existe la lista: impedir
que se vuelvan a proponer sin datos—, no el decimal. Las ablaciones que si
tienen que estar al dia se recalculan en cada ejecucion y viven en el artifact
(`ablacion_semana_anio`, `ablacion_colas_features`), no en este comentario.

  - Filtro de plausibilidad comercial `PESO_NETO >= 20 kg` (elimina 0.70% de
    filas: muestras y repuestos): MAPE 22.23% -> 21.67% pero MAE 0.0383 ->
    0.0395, RMSE 0.0535 -> 0.0553 y R2 -0.018 -> -0.087. Descartado: mejora una
    metrica y empeora tres, y estrecha el alcance de la tesis sin justificacion
    independiente del resultado. (Tercera auditoria: reproducido al decimal,
    aplicando el filtro ANTES del pipeline, que es la unica forma correcta —
    aplicarlo despues no reproduce porque no reajusta umbrales ni serie.)
  - Tratar el bucket "No Disponible - Ley 29733" como missing (ver
    `NOTA_LEY_29733`): R2 -0.018 -> -0.170, RMSE 0.0535 -> 0.0574. Descartado.
    (Tercera auditoria: reproducido al decimal.)

  - CORRECCION DE LA TERCERA AUDITORIA — ablacion de `semana_anio`. La segunda
    auditoria publico "MAPE 22.23% -> 23.34%, R2 -0.018 -> -0.447" y concluyo
    que la feature aportaba senal. Esas cifras NO REPRODUCEN: ninguna de seis
    configuraciones probadas las alcanza. La medicion correcta la calcula ahora
    `train_model.py` en cada ejecucion (meta -> `ablacion_semana_anio`) para que
    no pueda volver a quedar obsoleta. El resultado real es que quitar la
    feature MEJORA el MAPE y empeora levemente un R2 que el propio documento
    declara inestable. Se CONSERVA la feature.

    QUINTA REVISION: bajo el esquema 80/20 el signo se INVIERTE — quitar
    `semana_anio` empeora el MAPE del holdout. Que una misma ablacion cambie de
    signo al cambiar el periodo de evaluacion es, en si mismo, el resultado mas
    informativo de los tres: la feature no tiene un efecto estable, y ninguna de
    las dos mediciones deberia citarse como si lo tuviera. El numero vigente lo
    calcula `train_model.py` en cada ejecucion: ver
    `meta["ablacion_semana_anio"]`. Su sensibilidad (una variacion mediana del
    ~25% moviendo solo esa feature) sigue siendo el argumento en contra.

"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Alcance: neumaticos (subpartida 4011), via maritima, Aduana del Callao.
# El filtro de pais se conserva por fidelidad al notebook original, pero se
# verifico que el 100% de los registros del dataset tienen CPAIS == 'CN':
# el alcance real del modelo es China -> Callao, no "Asia" -> Callao.
PAISES_ORIGEN = ["CN", "TH", "VN", "KR", "JP", "ID", "MY", "IN", "SG", "PH", "HK", "TW"]

P_LOW, P_HIGH = 0.005, 0.995

# ── Esquema de particion ───────────────────────────────────────────────────
# AVISO SOBRE EL NOMBRE (cuarta auditoria). "80_20" es un IDENTIFICADOR
# historico, no una proporcion. El corte real es una frontera de ano calendario
# y el reparto que produce es 74.6% / 25.4% (61,406 / 20,891 filas), no 80/20.
# El identificador se conserva porque viaja dentro de los artifacts ya
# generados (`modelo_meta.json`, `walk_forward_2025.json -> esquema_particion
# _base`) y renombrarlo los invalidaria sin reentrenar. El paper, en cambio, ya
# no usa esta etiqueta en ningun sitio: dice "particion temporal por ano
# calendario: entrenamiento 2021-2024 (74.6%) y validacion 2025 (25.4%)", que
# es la descripcion correcta. Leer "80/20" mas abajo como el nombre del
# esquema, nunca como un porcentaje de filas.
#
# "80_20"    : TRAIN = 2021-2024, HOLDOUT = todo 2025 (esquema vigente).
#              El bloque de entrenamiento se subdivide 80/20 POR FECHA — esta
#              subparticion interna si es proporcional: cabeza para ajustar,
#              cola para early stopping + calibracion conformal.
# "70_20_10" : esquema historico por cuantiles posicionales 0.70 / 0.90.
#              Se conserva para reproducir los resultados anteriores; el
#              artifact que produjo esta respaldado en ml/respaldo_70_20_10/.
ESQUEMAS = ("80_20", "70_20_10")
ESQUEMA_DEFAULT = os.environ.get("JPS_SPLIT_ESQUEMA", "80_20")

# Primer dia del holdout en el esquema 80_20. Es una fecha, no un cuantil: la
# revision metodologica pidio explicitamente "train 2021-2024, validacion todo
# 2025", y una frontera de anio calendario no se mueve cuando crece el corpus.
INICIO_HOLDOUT_80_20 = pd.Timestamp("2025-01-01")

# Fraccion del bloque de entrenamiento que se reserva, por fecha y al final,
# para early stopping y calibracion conformal (subparticion 80/20 anidada).
FRACCION_TRAIN_INTERNO = 0.80

FEATURES = [
    "mes", "trimestre", "semana_anio", "mes_sin", "mes_cos",
    "mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3",
    "puerto_freq", "importador_freq",
    "densidad_carga", "ratio_bruto_neto",
    "ruta_directa",
]
TARGET = "FLETE_UNIT"

# Umbral de curacion de los catalogos del formulario (dropdowns).
MIN_REGISTROS_CATALOGO = 50

# Prefijo del bucket anonimizado por la Ley 29733 de Proteccion de Datos.
PREFIJO_ANONIMO = "No Disponible"

NOTA_LEY_29733 = (
    "El bucket 'No Disponible - Ley 29733' agrupa ~6.8% de los registros de "
    "empresas distintas bajo un unico nombre. Permanece en importador_freq por "
    "fidelidad al dato, con la frecuencia mas alta de la tabla (~0.086 frente a "
    "un default de ~0.0007), pero NO se ofrece en el dropdown: es un punto del "
    "espacio de features que produccion nunca puede alcanzar. Tratarlo como "
    "missing se probo y empeora el modelo (R2 -0.018 -> -0.170), asi que se "
    "conserva y se declara como limitacion, no como decision optima."
)


@dataclass
class Particiones:
    """Dataset particionado + los encoders ajustados exclusivamente con train.

    Los nombres de los tres tramos se conservan (`train` / `val` / `test`)
    porque son los que consume todo el resto del sistema, pero en el esquema
    vigente su LECTURA cambia y conviene tenerla delante:

      train : cabeza del 80% (2021 -> ~mediados de 2024). Ajusta los modelos,
              los encoders, las medianas y los umbrales de recorte.
      val   : cola del 80% (~mediados de 2024 -> 2024-12). Early stopping y,
              partido en dos por `split_val()`, calibracion conformal.
              Sigue siendo entrenamiento en el sentido del split 80/20: no es
              el conjunto sobre el que se reportan resultados.
      test  : el 20% = todo 2025. Es la "validacion" del esquema 80/20 tal y
              como la nombro la revision metodologica. NUNCA participa en
              ningun ajuste. Es el conjunto cuyas metricas se publican.

    `etiquetas()` devuelve esta correspondencia para volcarla al artifact, de
    modo que quien lea `modelo_meta.json` no tenga que deducirla.
    """
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    fecha_corte_train: pd.Timestamp
    fecha_corte_val: pd.Timestamp
    puerto_freq: pd.Series
    importador_freq: pd.Series
    puerto_freq_default: float
    importador_freq_default: float
    densidad_carga_median: float
    ratio_bruto_neto_median: float
    ruta_directa_por_puerto: dict
    ruta_directa_default: int
    serie_mercado: pd.Series
    diag_recorte: pd.DataFrame
    diag_sesgo_recorte: dict = field(default_factory=dict)
    diag_ruta_default: dict = field(default_factory=dict)
    esquema: str = ESQUEMA_DEFAULT

    def etiquetas(self) -> dict:
        """Como se llama cada tramo en el esquema con el que se construyo."""
        if self.esquema == "80_20":
            return {
                "esquema": "80_20",
                "criterio": (
                    "TRAIN = 2021-2024 (80%), HOLDOUT = todo 2025 (20%), corte "
                    "por anio calendario; el 80% se subdivide 80/20 por fecha "
                    "en cabeza y cola, y la cola reserva su ultima mitad para "
                    "la calibracion conformal"
                ),
                "train": "cabeza del 80% — ajuste de modelos, encoders y umbrales",
                "val": ("cola del 80% — entra al ajuste del modelo puntual; su "
                        "ultima mitad se reserva para la calibracion conformal"),
                "test": (
                    "el 20% = todo 2025. Es la VALIDACION del esquema 80/20: "
                    "nunca participa en ningun ajuste y es donde se miden los "
                    "resultados publicados"
                ),
                "evaluacion_complementaria": (
                    "ml/walk_forward.py — reentrenamiento mensual sobre los 12 "
                    "meses de 2025 (origen rodante)"
                ),
            }
        return {
            "esquema": "70_20_10",
            "criterio": "cuantiles posicionales 0.70 / 0.90 redondeados a limite de dia",
            "train": "70% inicial", "val": "20% intermedio", "test": "10% final",
            "nota": (
                "Esquema historico. Se conserva para reproducir los resultados "
                "anteriores; el artifact que produjo esta en ml/respaldo_70_20_10/."
            ),
        }

    def split_val(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """Parte VAL en dos mitades cronologicas con corte limpio por fecha.

        VAL_ES  (primera mitad): early stopping de los modelos de cuantiles.
        VAL_CAL (segunda mitad): calibracion conformal. Nunca entrena nada.
        """
        corte = _corte_por_fecha(self.val["FECHA"], 0.50)
        return (
            self.val[self.val["FECHA"] < corte].copy(),
            self.val[self.val["FECHA"] >= corte].copy(),
            corte,
        )


def _corte_por_fecha(fechas: pd.Series, q: float) -> pd.Timestamp:
    """Fecha de corte que NO parte ningun dia entre dos particiones.

    Toma el cuantil posicional pedido y avanza hasta el inicio del primer dia
    estrictamente posterior, de modo que la fila anterior y la siguiente al
    corte nunca comparten fecha.
    """
    fechas = fechas.sort_values()
    aprox = fechas.iloc[min(int(q * len(fechas)), len(fechas) - 1)]
    return pd.Timestamp(aprox).normalize() + pd.Timedelta(days=1)


def _ajustar_lookup_ruta(scope: pd.DataFrame, mask_train: pd.Series) -> tuple[dict, int]:
    """Lookup puerto -> regimen de ruta, ajustado SOLO con train.

    Se construye ANTES del recorte de outliers porque su resultado es la clave
    con la que se estratifica ese recorte (ver punto 4 del docstring del modulo).
    Se verifico que construirlo antes o despues del recorte da exactamente el
    mismo lookup (57 puertos, 0 discrepancias), asi que adelantarlo no cambia
    el artifact — solo hace coherente la limpieza con la feature.

    En vez de `.first()`, que tomaria en silencio el primer valor si un puerto
    mezclara ambos regimenes, se comprueba el determinismo de forma explicita.
    """
    base = scope[mask_train]
    ambiguos = base.groupby("PUER_DESC")["ruta_directa_real"].nunique()
    ambiguos = ambiguos[ambiguos > 1]
    if len(ambiguos):
        raise AssertionError(
            "ruta_directa no es deterministica por puerto en train: "
            f"{ambiguos.index.tolist()}. La feature asume un unico regimen por "
            "puerto de embarque; revisar antes de continuar."
        )
    lookup = base.groupby("PUER_DESC")["ruta_directa_real"].first().to_dict()
    default = int(base["ruta_directa_real"].mode()[0])
    return {str(k): int(v) for k, v in lookup.items()}, default


def _recorte_estratificado(
    df: pd.DataFrame, mask_train: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    """Recorte P0.5-P99.5 por regimen de ruta, con percentiles ajustados solo con train.

    Estratifica por `ruta_directa` — el valor del LOOKUP, que es el que recibe
    el modelo — y no por el valor real por fila. Que la clave de limpieza y la
    feature sean la misma columna es lo que impide que una fila se limpie bajo
    un regimen y se modele bajo el otro.

    Retorna la mascara de filas conservadas y una tabla de diagnostico que
    permite auditar que ningun segmento se recorta desproporcionadamente.

    ALCANCE DEL RECORTE (tercera auditoria): esta funcion recorta UNICAMENTE la
    cola del target (`FLETE_UNIT`). Ninguna feature de entrada se recorta nunca.
    Eso deja entrar a entrenamiento valores fisicamente imposibles en las
    features — `ratio_bruto_neto` llega a 10.1 con un p99 de 1.08, y
    `densidad_carga` a 1491 con un p99 de 66 — que ninguna de las tres
    auditorias habia mirado. Se mide en `diagnostico_colas_features()` y se
    probo recortarlas (ver `meta["ablacion_colas_features"]`).

    ADVERTENCIA SOBRE LA ESTABILIDAD DE LOS UMBRALES (segunda auditoria). El
    grupo de transbordo tiene ~2,700 filas en train, es decir ~13 observaciones
    por cola. Un bootstrap de 2,000 remuestreos da para su P99.5 un IC95 de
    [1.9844, 2.3412] — el punto estimado esta pegado al extremo superior de su
    propia distribucion. La DIFERENCIA entre regimenes si es real (los IC de la
    diferencia no cruzan cero; KS D=0.1817, p=7.7e-75; medianas 0.2332 vs
    0.4242), asi que estratificar es correcto. El VALOR concreto del umbral
    superior del transbordo, en cambio, es inestable y debe declararse como tal.
    """
    conservar = pd.Series(False, index=df.index)
    filas = []
    for regimen, grupo in df.groupby("ruta_directa"):
        base = grupo[mask_train.reindex(grupo.index, fill_value=False)]
        fallback = len(base) < 100
        if fallback:
            # Sin masa suficiente en train para estimar colas: se recorta con
            # los percentiles globales de train para no inventar umbrales.
            print(f"  AVISO: regimen {int(regimen)} tiene solo {len(base)} filas en "
                  "train; se usan los percentiles globales de train.")
            base = df[mask_train]
        lo, hi = base["FLETE_UNIT"].quantile([P_LOW, P_HIGH])
        dentro = grupo["FLETE_UNIT"].between(lo, hi)
        conservar.loc[grupo.index] = dentro
        filas.append({
            "ruta_directa": int(regimen),
            "n": len(grupo),
            "n_train": int(len(base)),
            "p_low": round(float(lo), 5),
            "p_high": round(float(hi), 5),
            "recortadas": int((~dentro).sum()),
            "pct_recortado": round(100 * float((~dentro).mean()), 3),
            "fallback_percentiles_globales": bool(fallback),
        })
    return conservar, pd.DataFrame(filas)


def _diagnostico_sesgo_recorte(scope: pd.DataFrame, conservar: pd.Series) -> dict:
    """Sesgo residual del recorte por importador, puerto y anio.

    La primera auditoria midio el sesgo del recorte por regimen de ruta y lo
    corrigio estratificando. Nadie lo midio por las otras dimensiones. Se mide
    aqui para que el sesgo no pueda volver a esconderse en una dimension que
    nadie mira: el recorte descarta ~3x mas filas de importadores raros que de
    frecuentes, lo que agrava el sesgo de equidad contra PYME ya declarado.
    """
    recortada = ~conservar
    cnt = scope["IMPORTADOR"].value_counts()
    frecuente = scope["IMPORTADOR"].map(cnt) >= cnt.median()
    por_puerto = scope.assign(_r=recortada).groupby("PUER_DESC")["_r"].agg(["size", "mean"])
    peores = por_puerto[(por_puerto["size"] >= 20)].nlargest(5, "mean")
    return {
        "pct_importador_frecuente": round(100 * float(recortada[frecuente].mean()), 3),
        "pct_importador_raro": round(100 * float(recortada[~frecuente].mean()), 3),
        "brecha_importador_x": round(
            float(recortada[~frecuente].mean()) / max(float(recortada[frecuente].mean()), 1e-9), 2
        ),
        "pct_por_anio": {
            str(k): round(100 * float(v), 3)
            for k, v in recortada.groupby(scope["FECHA"].dt.year).mean().items()
        },
        "puertos_mas_recortados": {
            str(k): {"n": int(r["size"]), "pct": round(100 * float(r["mean"]), 2)}
            for k, r in peores.iterrows()
        },
        "nota": (
            "El recorte descarta mas filas de importadores raros que de "
            "frecuentes. Es el mismo tipo de sesgo que motivo la estratificacion "
            "por ruta, sobre otra dimension, y agrava el sesgo de equidad contra "
            "importadores nuevos/PYME declarado en las limitaciones."
        ),
    }


def _diagnostico_ruta_default(scope: pd.DataFrame, lookup: dict, default: int) -> dict:
    """Fiabilidad del valor por defecto de `ruta_directa` en el segmento donde se aplica.

    El default es la moda de train (1 = directo), correcta en el 95.8% de las
    filas con puerto conocido. Pero solo se APLICA a puertos no vistos en train,
    y ahi su acierto cae al 66.2%: un puerto desconocido es, por construccion,
    un puerto exotico (Hamburgo, Itajai, Acajutla), donde el transbordo pesa
    mucho mas. Se verifico que invertir el default a 0 seria peor (acertaria el
    33.8%), asi que se conserva 1 — pero la caida de fiabilidad se declara en
    vez de quedar implicita.
    """
    no_visto = ~scope["PUER_DESC"].isin(lookup)
    if not no_visto.any():
        return {"n_filas_puerto_no_visto": 0}
    real = scope.loc[no_visto, "ruta_directa_real"]
    return {
        "n_filas_puerto_no_visto": int(no_visto.sum()),
        "pct_del_dataset": round(100 * float(no_visto.mean()), 3),
        "default_aplicado": int(default),
        "acierto_del_default_en_ese_segmento_%": round(100 * float((real == default).mean()), 1),
        "acierto_en_puertos_conocidos_%": round(
            100 * float((scope.loc[~no_visto, "ruta_directa_real"] == default).mean()), 1
        ),
        "nota": (
            "El default acierta bastante menos en el segmento donde realmente se "
            "aplica. Invertirlo seria peor; la mitigacion correcta es advertir al "
            "usuario cuando el puerto no figura en el historico, cosa que "
            "predictor._construir_advertencia() ya hace."
        ),
    }


def _cortes(fechas: pd.Series, esquema: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(corte train/val, corte val/test) para el esquema pedido.

    En 80_20 el corte externo es una FECHA FIJA (2025-01-01) y el interno se
    deriva de la masa de datos que queda por debajo de el, para que la cola de
    early stopping + calibracion tenga siempre un tamano proporcional al
    bloque de entrenamiento y no dependa del anio en que se ejecute.

    Se valida que el corte externo caiga dentro del rango de datos: si el
    corpus creciera hacia atras o hacia adelante y 2025 quedara vacio (o lo
    quedara el tramo 2021-2024), el fallo debe ser ruidoso aqui y no un
    entrenamiento silencioso sobre un holdout de cero filas.
    """
    if esquema not in ESQUEMAS:
        raise ValueError(f"esquema desconocido: {esquema!r}; use uno de {ESQUEMAS}")

    if esquema == "70_20_10":
        return _corte_por_fecha(fechas, 0.70), _corte_por_fecha(fechas, 0.90)

    corte_holdout = INICIO_HOLDOUT_80_20
    n_train = int((fechas < corte_holdout).sum())
    n_holdout = int((fechas >= corte_holdout).sum())
    if n_train == 0 or n_holdout == 0:
        raise ValueError(
            f"el esquema 80_20 parte en {corte_holdout.date()} y ese corte deja "
            f"{n_train} filas de entrenamiento y {n_holdout} de holdout. El "
            "corpus no cubre el periodo que el esquema supone (2021-2024 + 2025)."
        )
    corte_interno = _corte_por_fecha(
        fechas[fechas < corte_holdout], FRACCION_TRAIN_INTERNO
    )
    return corte_interno, corte_holdout


def cargar_scope(csv_path: str) -> pd.DataFrame:
    """CSV crudo -> filas dentro del alcance, ordenadas, con target y ruta real.

    Se separo de `construir()` para que `ml/walk_forward.py` pueda recortar el
    scope por fecha y volver a particionarlo mes a mes SIN duplicar el filtro
    de alcance. Que el origen rodante y el entrenamiento de produccion
    compartan esta funcion es lo que garantiza que evaluen el mismo universo.
    """
    df = pd.read_csv(csv_path, sep=",", quotechar='"', encoding="utf-8", low_memory=False)
    df["FECHA"] = pd.to_datetime(df["FECHA"], format="%Y%m%d", errors="coerce")
    df["CNAN"] = df["CNAN"].astype(str).str.zfill(10)

    mask = (
        df["CNAN"].str.startswith("4011")
        & (df["VIA_TRANSP"] == 1)
        & (df["ADUA_DESC"].str.upper().str.contains("CALLAO", na=False))
        & (df["CPAIS"].isin(PAISES_ORIGEN))
    )
    scope = df.loc[mask].sort_values("FECHA", kind="mergesort").reset_index(drop=True)
    scope["FLETE_UNIT"] = scope["FLE_DOLAR"] / scope["PESO_NETO"]
    # Valor real por fila. Se conserva con nombre propio: es la fuente del
    # lookup y de los diagnosticos, pero NUNCA es la feature.
    scope["ruta_directa_real"] = (scope["CPAIS_PROC"] == scope["CPAIS"]).astype(int)
    return scope


def construir(csv_path: str, esquema: str = ESQUEMA_DEFAULT) -> Particiones:
    """Carga el CSV crudo y devuelve las particiones + los encoders de train."""
    scope = cargar_scope(csv_path)
    # Cortes provisionales sobre el alcance (se reusan como cortes definitivos:
    # son fechas, no posiciones, asi que el recorte posterior no los desplaza).
    fecha_corte_train, fecha_corte_val = _cortes(scope["FECHA"], esquema)
    return particionar(scope, fecha_corte_train, fecha_corte_val, esquema)


def particionar(
    scope: pd.DataFrame,
    fecha_corte_train: pd.Timestamp,
    fecha_corte_val: pd.Timestamp,
    esquema: str = ESQUEMA_DEFAULT,
) -> Particiones:
    """Recorte, feature engineering, particion y encoders sobre un scope dado.

    Todo lo que se ajusta aqui —lookup de ruta, umbrales de recorte, encoders,
    medianas— usa EXCLUSIVAMENTE las filas anteriores a `fecha_corte_train`.
    `ml/walk_forward.py` llama a esta funcion una vez por mes evaluado, con el
    scope truncado a la historia disponible en ese momento, de modo que el
    origen rodante no puede usar una metodologia distinta de la de produccion:
    es literalmente el mismo codigo.
    """
    scope = scope.sort_values("FECHA", kind="mergesort").reset_index(drop=True)
    # Los umbrales de recorte y el lookup de ruta se ajustan con la CABEZA del
    # bloque de entrenamiento, no con el bloque entero. Asi la cola (val) sigue
    # siendo un conjunto no visto tambien para la calibracion conformal, que es
    # lo que exige la garantia de CQR. En ningun esquema entra aqui una sola
    # fila del holdout.
    es_train = scope["FECHA"] < fecha_corte_train

    # El lookup se ajusta ANTES del recorte porque es la clave de estratificacion.
    ruta_directa_por_puerto, ruta_directa_default = _ajustar_lookup_ruta(scope, es_train)
    scope["ruta_directa"] = (
        scope["PUER_DESC"].map(ruta_directa_por_puerto)
        .fillna(ruta_directa_default).astype(int)
    )
    diag_ruta_default = _diagnostico_ruta_default(
        scope, ruta_directa_por_puerto, ruta_directa_default
    )

    conservar, diag_recorte = _recorte_estratificado(scope, es_train)
    diag_sesgo_recorte = _diagnostico_sesgo_recorte(scope, conservar)

    clean = scope[conservar].drop_duplicates()
    clean = clean[clean["FLE_DOLAR"] > 0].sort_values("FECHA", kind="mergesort").reset_index(drop=True)

    # ── Feature engineering ────────────────────────────────────────────────
    fe = clean
    fe["mes"] = fe["FECHA"].dt.month
    fe["trimestre"] = fe["FECHA"].dt.quarter
    fe["semana_anio"] = fe["FECHA"].dt.isocalendar().week.astype(int)
    fe["mes_sin"] = np.sin(2 * np.pi * fe["mes"] / 12)
    fe["mes_cos"] = np.cos(2 * np.pi * fe["mes"] / 12)

    fe["periodo"] = fe["FECHA"].dt.to_period("M")
    serie_mercado = fe.groupby("periodo")["FLETE_UNIT"].mean().sort_index()
    lags = pd.DataFrame({"mercado_mean": serie_mercado})
    lags["mercado_lag1"] = lags["mercado_mean"].shift(1)
    lags["mercado_lag2"] = lags["mercado_mean"].shift(2)
    lags["mercado_lag3"] = lags["mercado_mean"].shift(3)
    lags["mercado_ma3"] = lags["mercado_mean"].shift(1).rolling(3).mean()
    fe = fe.merge(
        lags[["mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3"]],
        left_on="periodo", right_index=True, how="left",
    )

    fe = fe.dropna(subset=["mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3"])
    fe = fe.sort_values("FECHA", kind="mergesort").reset_index(drop=True)

    # ── Particion por fecha (ningun dia queda repartido) ───────────────────
    train = fe[fe["FECHA"] < fecha_corte_train].copy()
    val = fe[(fe["FECHA"] >= fecha_corte_train) & (fe["FECHA"] < fecha_corte_val)].copy()
    test = fe[fe["FECHA"] >= fecha_corte_val].copy()

    # ── Encoders: ajustados EXCLUSIVAMENTE con train ───────────────────────
    puerto_freq = train["PUER_DESC"].value_counts(normalize=True)
    importador_freq = train["IMPORTADOR"].value_counts(normalize=True)
    puerto_freq_default = float(puerto_freq.median())
    importador_freq_default = float(importador_freq.median())

    densidad_carga_median = float(
        (train["PESO_NETO"] / train["UNID_FIQTY"].replace(0, np.nan)).median()
    )
    ratio_bruto_neto_median = float(
        (train["PESO_BRUTO"] / train["PESO_NETO"].replace(0, np.nan)).median()
    )

    p = Particiones(
        train=train, val=val, test=test,
        fecha_corte_train=fecha_corte_train, fecha_corte_val=fecha_corte_val,
        puerto_freq=puerto_freq, importador_freq=importador_freq,
        puerto_freq_default=puerto_freq_default,
        importador_freq_default=importador_freq_default,
        densidad_carga_median=densidad_carga_median,
        ratio_bruto_neto_median=ratio_bruto_neto_median,
        ruta_directa_por_puerto=ruta_directa_por_puerto,
        ruta_directa_default=ruta_directa_default,
        serie_mercado=serie_mercado,
        diag_recorte=diag_recorte,
        diag_sesgo_recorte=diag_sesgo_recorte,
        diag_ruta_default=diag_ruta_default,
        esquema=esquema,
    )
    p.train = aplicar_encoders(p.train, p)
    p.val = aplicar_encoders(p.val, p)
    p.test = aplicar_encoders(p.test, p)
    return p


def aplicar_encoders(d: pd.DataFrame, p: Particiones) -> pd.DataFrame:
    """Aplica a cualquier particion los encoders ajustados solo con train.

    `ruta_directa` se deriva SIEMPRE del lookup de puerto — el mismo camino que
    usa produccion — y nunca del valor real por fila, para que las metricas se
    midan sobre el vector de features que el sistema desplegado sabe construir.
    """
    d = d.copy()
    d["puerto_freq"] = d["PUER_DESC"].map(p.puerto_freq).fillna(p.puerto_freq_default)
    d["importador_freq"] = (
        d["IMPORTADOR"].map(p.importador_freq).fillna(p.importador_freq_default)
    )
    densidad = d["PESO_NETO"] / d["UNID_FIQTY"].replace(0, np.nan)
    d["densidad_carga"] = densidad.fillna(p.densidad_carga_median)
    ratio = d["PESO_BRUTO"] / d["PESO_NETO"].replace(0, np.nan)
    d["ratio_bruto_neto"] = ratio.fillna(p.ratio_bruto_neto_median)
    d["ruta_directa"] = (
        d["PUER_DESC"].map(p.ruta_directa_por_puerto)
        .fillna(p.ruta_directa_default).astype(int)
    )
    return d


def catalogos_dropdown(p: Particiones) -> tuple[list[str], list[str]]:
    """Catalogos del formulario, curados SOBRE TRAIN.

    Se curan con las mismas filas que ajustaron los encoders para garantizar la
    invariante: toda opcion ofrecida en el formulario tiene una entrada real en
    `puerto_freq` / `importador_freq` / `ruta_directa_por_puerto`. Curarlos
    sobre el historico completo (como se hacia antes) dejaba en el dropdown
    puertos e importadores que el modelo nunca vio y que caian en silencio al
    valor por defecto.
    """
    puertos = p.train["PUER_DESC"].value_counts()
    puertos = sorted(puertos[puertos >= MIN_REGISTROS_CATALOGO].index.astype(str))

    importadores = p.train["IMPORTADOR"].value_counts()
    importadores = importadores[importadores >= MIN_REGISTROS_CATALOGO]
    # Se excluye el bucket anonimizado por la Ley 29733: no es una empresa
    # seleccionable, aunque si permanece en importador_freq por fidelidad.
    importadores = sorted(
        i for i in importadores.index.astype(str) if not i.startswith(PREFIJO_ANONIMO)
    )
    return puertos, importadores


def cobertura_catalogos(p: Particiones, puertos: list[str], importadores: list[str]) -> dict:
    """Que fraccion del volumen historico real cubren los catalogos del formulario.

    La primera auditoria verifico que toda opcion del dropdown tiene encoder.
    Faltaba la pregunta inversa: cuanto del mercado real queda FUERA del
    dropdown. Se mide aqui para que la cifra no haya que estimarla a ojo.
    """
    todo = pd.concat([p.train, p.val, p.test])
    en_pu = todo["PUER_DESC"].isin(puertos)
    en_im = todo["IMPORTADOR"].isin(importadores)
    anonimo = todo["IMPORTADOR"].astype(str).str.startswith(PREFIJO_ANONIMO)
    return {
        "puertos_ofrecidos": len(puertos),
        "puertos_en_historico": int(todo["PUER_DESC"].nunique()),
        "cobertura_filas_puerto_%": round(100 * float(en_pu.mean()), 2),
        "cobertura_peso_puerto_%": round(
            100 * float(todo.loc[en_pu, "PESO_NETO"].sum() / todo["PESO_NETO"].sum()), 2
        ),
        "importadores_ofrecidos": len(importadores),
        "importadores_en_historico": int(todo["IMPORTADOR"].nunique()),
        "cobertura_filas_importador_%": round(100 * float(en_im.mean()), 2),
        "cobertura_filas_importador_sin_anonimo_%": round(
            100 * float(en_im[~anonimo].mean()), 2
        ),
        "pct_bucket_ley_29733": round(100 * float(anonimo.mean()), 2),
    }


def diagnostico_colas_features(p: Particiones) -> dict:
    """Colas de las features de entrada, que el recorte de outliers NUNCA toca.

    El recorte P0.5-P99.5 se aplica solo a `FLETE_UNIT`. Las features entran sin
    filtro, asi que valores imposibles sobreviven al pipeline. Se mide aqui para
    que la cola de las entradas deje de ser una dimension que nadie audita.
    """
    todo = pd.concat([p.train, p.val, p.test])
    out = {}
    for f in ("densidad_carga", "ratio_bruto_neto"):
        s = todo[f]
        out[f] = {
            "p50": round(float(s.median()), 4),
            "p99": round(float(s.quantile(0.99)), 4),
            "max": round(float(s.max()), 4),
            "ratio_max_p99": round(float(s.max() / s.quantile(0.99)), 1),
            "n_sobre_p99": int((s > s.quantile(0.99)).sum()),
        }
    out["nota"] = (
        "El recorte de outliers solo recorta el target. Estas colas entran a "
        "entrenamiento sin filtro. `ratio_bruto_neto` > 2 es fisicamente "
        "imposible para neumaticos (el embalaje no puede pesar mas que la "
        "carga) y aun asi llega a 10.1. Recortarlas se probo: ver "
        "`meta['ablacion_colas_features']`."
    )
    return out


def diagnostico_saturacion(model, p: Particiones, feature: str, valores: list) -> dict:
    """Respuesta del modelo a una feature, con todo lo demas fijo.

    Existe porque `densidad_carga` satura: por encima de ~50 kg/unidad el modelo
    devuelve exactamente el mismo numero para 50, 500 o 50000. Un usuario que se
    equivoque en el campo "unidades" no ve ningun cambio en la estimacion y el
    sistema no se lo advierte. Se mide para poder declararlo.
    """
    base = p.test[FEATURES].iloc[[0]]
    resp = {}
    for v in valores:
        resp[str(v)] = round(float(model.predict(base.assign(**{feature: v}))[0]), 6)
    distintos = sorted(set(resp.values()))
    # Umbral por encima del cual la respuesta ya no cambia.
    ultimo = None
    for v in valores:
        if resp[str(v)] == resp[str(valores[-1])] and ultimo is None:
            ultimo = v
    return {
        "feature": feature,
        "respuesta_por_valor": resp,
        "n_valores_distintos": len(distintos),
        "satura_a_partir_de": ultimo,
        "p99_train": round(float(p.train[feature].quantile(0.99)), 2),
        "max_train": round(float(p.train[feature].max()), 2),
        "nota": (
            f"El modelo devuelve el mismo valor para todo {feature} >= "
            f"{ultimo}, mientras el maximo real de train es "
            f"{float(p.train[feature].max()):.0f}. La feature aporta poco gain "
            "y su cola no discrimina: un error de tipeo en 'unidades' no mueve "
            "la estimacion. Se declara como limitacion en la UI y en §10."
        ),
    }


def diagnostico_recorte(p: Particiones) -> str:
    """Texto auditable del efecto del recorte de outliers por segmento."""
    lineas = [
        "Recorte de outliers P0.5-P99.5 "
        "(percentiles solo-train, estratificados por el MISMO ruta_directa que ve el modelo):"
    ]
    for _, r in p.diag_recorte.iterrows():
        etq = "directo   " if r["ruta_directa"] == 1 else "transbordo"
        lineas.append(
            f"  {etq} | n={int(r['n']):>6,} (train {int(r['n_train']):>6,}) "
            f"| rango [{r['p_low']}, {r['p_high']}] "
            f"| recortadas {int(r['recortadas']):>4} ({r['pct_recortado']:.3f}%)"
        )
    pct = p.diag_recorte["pct_recortado"]
    lineas.append(
        f"  brecha entre regimenes: {pct.max() / max(pct.min(), 1e-9):.2f}x "
        f"(con recorte global no estratificado era 7.0x)"
    )
    s = p.diag_sesgo_recorte
    lineas.append(
        f"  SESGO RESIDUAL por importador: raro {s['pct_importador_raro']:.3f}% vs "
        f"frecuente {s['pct_importador_frecuente']:.3f}% ({s['brecha_importador_x']:.2f}x). "
        "Agrava el sesgo de equidad contra PYME; declararlo en la tesis."
    )
    lineas.append(
        "  ESTABILIDAD DE UMBRALES: el P99.5 del transbordo se estima con ~13 "
        "observaciones por cola; su IC95 bootstrap es [1.9844, 2.3412]. La "
        "diferencia entre regimenes es real (KS p=7.7e-75); el valor del umbral no es preciso."
    )
    d = p.diag_ruta_default
    if d.get("n_filas_puerto_no_visto"):
        lineas.append(
            f"  DEFAULT DE RUTA: se aplica a {d['n_filas_puerto_no_visto']} filas "
            f"({d['pct_del_dataset']:.3f}%) con puerto no visto; acierta el "
            f"{d['acierto_del_default_en_ese_segmento_%']:.1f}% ahi, frente al "
            f"{d['acierto_en_puertos_conocidos_%']:.1f}% en puertos conocidos."
        )
    return "\n".join(lineas)

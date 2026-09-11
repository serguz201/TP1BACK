"""
Predictor de flete marítimo usando el modelo XGBoost entrenado.

Transforma los inputs del formulario en las features del modelo, ejecuta la
predicción, construye el intervalo de confianza 95% vía Conformalized Quantile
Regression y calcula los top-3 SHAP values con etiquetas de negocio.

CORRECCIONES DE AUDITORÍA (2026-09) aplicadas en este archivo:

  · Los rezagos de mercado ya NO son una constante. Antes se tomaban siempre
    los últimos tres meses del artifact, sin importar la fecha pedida: el
    modelo evaluado y el modelo servido no eran el mismo modelo. Ahora, para
    una fecha dentro del histórico se usan los tres meses realmente anteriores
    a esa fecha (igual que en entrenamiento), y solo para fechas futuras se
    recurre a los rezagos vigentes de `market_state` — que un administrador
    puede actualizar en caliente al cerrar cada mes.

  · Toda predicción declara `mercado_vigente_hasta` y `meses_extrapolados`.
    Como las tres variables de mercado concentran ~89% del gain, una cotización
    muy alejada de la última observación real es estructuralmente frágil, y el
    sistema debe decirlo en vez de devolver un número sin contexto.

  · Terminología: lo que el formulario selecciona es el PUERTO DE EMBARQUE, no
    el puerto de origen. El origen es siempre China; Busan, Yokohama, Hong Kong
    o Laem Chabang son puertos de transbordo. Esa distinción es justamente lo
    que codifica `ruta_directa`.

CORRECCIONES DE LA SEGUNDA AUDITORÍA (2026-09) aplicadas en este archivo:

  · TERCERA RAMA para fechas ANTERIORES al inicio de la serie. `_resolver_lags()`
    solo distinguía "dentro del histórico" de "futuro". Una fecha de 2019 caía
    en la rama de futuro, recibía los rezagos vigentes de 2025-12 y — como los
    meses extrapolados se calculaban con `max(0, ...)` — reportaba
    `meses_extrapolados = 0` y ninguna advertencia. El usuario veía una
    cotización de 2019 etiquetada "Mercado observado hasta 2025-12" con
    apariencia de estimación válida. Ahora esa rama existe, usa el primer mes
    real de la serie y declara los meses extrapolados HACIA ATRÁS.

  · Q CONFORMAL SEGÚN EL RÉGIMEN Y EL HORIZONTE. El intervalo servido para una
    cotización a 53 meses vista era un 0.05% más ancho que para una dentro del
    histórico: el sistema advertía por escrito que la estimación era frágil
    mientras su propia medida de incertidumbre no registraba nada. La segunda
    auditoría añadió una segunda constante para el régimen extrapolado; la
    tercera midió su cobertura condicional (54% a seis meses, 18% a nueve) y la
    sustituyó por la tabla `Q(h)` de `_q_conformal()`. Ver
    ml/train_quantile_models.py, sección 3b.

  · MAPE HONESTO POR RÉGIMEN. `mape_modelo` devolvía siempre el MAPE de test
    (~22%), incluso para cotizaciones extrapoladas cuyo error esperado es el del
    escenario congelado (~27.5%). Ahora devuelve el que corresponde al régimen
    en el que se sirvió la predicción, y `mape_regimen` dice cuál es.

  · FECHA INVÁLIDA. Se sigue tolerando una fecha no parseable por compatibilidad
    (el schema ya la valida antes), pero ahora se declara en `advertencia` en vez
    de sustituirla por "hoy" en silencio.
"""

import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

from ml.market_state import get_market_rates

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL ARTIFACT (modelo + metadatos)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "modelo_xgboost_flete.pkl"
QLO_PATH = MODEL_DIR / "modelo_xgboost_flete_q_lo.pkl"
QHI_PATH = MODEL_DIR / "modelo_xgboost_flete_q_hi.pkl"
META_PATH  = MODEL_DIR / "modelo_meta.json"

# Todo lo que se deriva del artifact vive DENTRO de `_derivar_de_meta()`, que
# reasigna los globals de este modulo. Hay una sola copia de esa logica: el
# import inicial y la recarga en caliente (`recargar_artifacts()`, expuesta en
# POST /api/maintenance/model/reload) ejecutan exactamente el mismo codigo, de
# modo que un artifact reentrenado no puede quedar aplicado a medias.
#
# BUG 4 (auditoria 2026-09). Hasta esta version solo `mercado_lag*` era
# actualizable en caliente. `puerto_freq`, `importador_freq`,
# `ruta_directa_por_puerto` y los tres .pkl quedaban congelados desde el
# arranque del proceso: reentrenar exigia reiniciar el servidor. Ahora
# `recargar_artifacts()` los vuelve a leer de disco.

_artifact_lock = threading.RLock()


def _leer_meta() -> dict:
    with open(META_PATH, encoding="utf-8") as f:
        return json.load(f)


def _derivar_de_meta(meta: dict) -> None:
    """Reasigna TODOS los globals derivados del artifact a partir de `meta`.

    Se invoca una vez al importar el modulo y otra vez en cada recarga en
    caliente. No toca los modelos .pkl: de eso se ocupa `recargar_artifacts()`.
    """
    global MODEL_META, MODEL_MAPE, CONFORMAL_Q, CONFORMAL_Q_POR_HORIZONTE
    global CONFORMAL_Q_EXTRAPOLADO, HORIZONTE_IC_CALIBRADO, MAPE_EXTRAPOLADO
    global SERIE_MERCADO, _MESES_SERIE, _PRIMER_MES_SERIE, _ULTIMO_MES_SERIE
    global PORT_FREQ, DEFAULT_PORT_FREQ, _CATALOG_PORTS
    global IMPORTADOR_FREQ, DEFAULT_IMPORTADOR_FREQ, _CATALOG_IMPORTADORES
    global RUTA_DIRECTA_POR_PUERTO, DEFAULT_RUTA_DIRECTA, FEATURE_ORDER

    MODEL_META = meta

    # MAPE del modelo en test (informativo; NO se usa para construir el IC95 —
    # ver ic95_conformal_Q / Conformalized Quantile Regression más abajo).
    MODEL_MAPE = MODEL_META["metricas_test"]["MAPE_%"]

    # Constante de calibración conformal (Romano, Patterson & Candès 2019), ajustada
    # sobre un tramo de validación nunca visto por los modelos de cuantiles ni por el
    # modelo puntual. Ver ml/train_quantile_models.py.
    CONFORMAL_Q = MODEL_META["ic95_conformal_Q"]

    # Tabla de constantes conformales del regimen EXTRAPOLADO, indexada por el
    # horizonte (meses entre el ultimo mercado observado y el que la fecha pedida
    # necesitaria). TERCERA AUDITORIA: antes era UNA sola constante calibrada a ~3
    # meses de horizonte y servida para fechas de hasta 60; su cobertura real caia
    # al 18% a nueve meses mientras la respuesta la etiquetaba "IC 95%". La tabla
    # es Q(h) = max(Q_cruda(1..h)) —monotona por construccion, de modo que mas
    # horizonte nunca da un intervalo mas estrecho— y sostiene >=95% de cobertura
    # condicional en CADA horizonte, no solo en promedio. Ver
    # ml/train_quantile_models.py seccion 3b.
    _Q_HORIZONTE_RAW = MODEL_META.get("ic95_conformal_Q_por_horizonte", {})
    CONFORMAL_Q_POR_HORIZONTE = {
        int(k): float(v) for k, v in _Q_HORIZONTE_RAW.items()
    }

    # Constante de respaldo si el artifact es anterior a la tabla.
    CONFORMAL_Q_EXTRAPOLADO = MODEL_META.get(
        "ic95_conformal_Q_extrapolado", MODEL_META["ic95_conformal_Q"]
    )

    # Ultimo horizonte (en meses) para el que la tabla Q(h) tiene una entrada
    # estimada con datos. Mas alla se sigue aplicando la entrada mas ancha —el
    # intervalo no desaparece— pero YA NO tiene la cobertura que anuncia, y la
    # respuesta lo declara con `ic95_calibrado = False`. Ver
    # ml/train_quantile_models.py, seccion 3b.
    HORIZONTE_IC_CALIBRADO = MODEL_META.get(
        "ic95_horizonte_calibrado_meses",
        max(CONFORMAL_Q_POR_HORIZONTE) if CONFORMAL_Q_POR_HORIZONTE else 6,
    )

    MAPE_EXTRAPOLADO = MODEL_META.get(
        "escenario_produccion_congelado", {}
    ).get("MAPE_%", MODEL_META["metricas_test"]["MAPE_%"])

    # Serie mensual real del mercado observada durante el entrenamiento: 'YYYY-MM'
    # → flete unitario medio (USD/kg). Es la fuente de los rezagos históricos.
    SERIE_MERCADO = MODEL_META["serie_mercado"]
    _MESES_SERIE = sorted(SERIE_MERCADO)
    _PRIMER_MES_SERIE = _MESES_SERIE[0]
    _ULTIMO_MES_SERIE = _MESES_SERIE[-1]

    # Frecuencias de puerto ajustadas SOLO con el tramo de entrenamiento (split
    # temporal), para evitar el look-ahead bias que tenía el pipeline original al
    # calcularlas sobre todo el histórico 2021-2025 antes del split.
    PORT_FREQ = MODEL_META["puerto_freq"]

    # Fallback para puertos no vistos en entrenamiento: mediana real de train
    # (no el mínimo — un puerto nunca visto no es necesariamente "el más raro posible").
    DEFAULT_PORT_FREQ = MODEL_META["puerto_freq_default"]

    # Puertos del dropdown: los de ≥50 registros EN TRAIN. Se curan con las mismas
    # filas que ajustaron los encoders, de modo que toda opción ofrecida tiene una
    # entrada real en PORT_FREQ y en RUTA_DIRECTA_POR_PUERTO. Curarlos sobre el
    # histórico completo (como se hacía antes) dejaba opciones que caían en
    # silencio al valor por defecto.
    _CATALOG_PORTS = MODEL_META["puertos_dropdown"]

    # Frecuencias reales de importador, ajustadas SOLO con train (mismo criterio).
    # Incluye el bucket "No Disponible - Ley 29733" (anonimizado por Ley de
    # Protección de Datos), que no se ofrece en el dropdown pero permanece en el
    # lookup por fidelidad a los datos reales.
    IMPORTADOR_FREQ = MODEL_META["importador_freq"]
    DEFAULT_IMPORTADOR_FREQ = MODEL_META["importador_freq_default"]
    _CATALOG_IMPORTADORES = MODEL_META["importadores_dropdown"]

    # Ruta directa (embarque en un puerto de China) vs. transbordo (embarque vía un
    # tercer país — Corea, Japón, Panamá, Europa, etc.). Es un hecho geográfico
    # determinístico del puerto de embarque (ningún puerto mezcla ambos regímenes:
    # verificado sobre los 57 puertos del tramo de entrenamiento, y el pipeline
    # falla si deja de cumplirse), por lo que se deriva automáticamente del puerto
    # ya seleccionado en el formulario.
    #
    # AVISO DE FIABILIDAD (segunda auditoría): el valor por defecto (1 = directo) es
    # la moda de train y acierta el 95.8% en puertos conocidos, pero solo el 66.2%
    # en el segmento donde realmente se aplica — puertos NO vistos en train, que por
    # construcción son puertos exóticos (Hamburgo, Itajaí, Acajutla) donde el
    # transbordo pesa mucho más. Invertirlo sería peor (33.8%), así que se conserva
    # y el sistema advierte cuando el puerto no figura en el histórico. Ver
    # modelo_meta.json -> diagnostico_ruta_default.
    RUTA_DIRECTA_POR_PUERTO = MODEL_META["ruta_directa_por_puerto"]
    DEFAULT_RUTA_DIRECTA = MODEL_META["ruta_directa_default"]

    # Debe coincidir exactamente (orden y nombres) con ml/data_pipeline.FEATURES.
    FEATURE_ORDER = MODEL_META["features"]

def _q_conformal(meses_extrapolados: int, extrapolado: bool) -> float:
    """Constante conformal que corresponde a esta peticion.

    Dentro del historico, la Q historica. Fuera, la entrada de la tabla para su
    horizonte; mas alla del ultimo horizonte estimable se aplica la ultima
    entrada (la mas ancha) y la respuesta lo declara con ic95_calibrado=False.
    """
    if not extrapolado:
        return CONFORMAL_Q
    if not CONFORMAL_Q_POR_HORIZONTE:
        return CONFORMAL_Q_EXTRAPOLADO
    h = max(1, int(meses_extrapolados))
    return CONFORMAL_Q_POR_HORIZONTE[min(h, max(CONFORMAL_Q_POR_HORIZONTE))]


# Horizonte más allá del cual una cotización se marca como extrapolada. No se
# bloquea la predicción — se devuelve con su advertencia, y quien cotiza decide.
MAX_MESES_EXTRAPOLACION = 12


# Etiquetas de negocio para cada feature del modelo
FEATURE_LABELS: dict[str, str] = {
    "mes": "Estacionalidad del mes",
    "trimestre": "Temporada trimestral",
    "semana_anio": "Semana del año",
    "mes_sin": "Ciclo estacional (seno)",
    "mes_cos": "Ciclo estacional (coseno)",
    "mercado_lag1": "Tendencia reciente del mercado",
    "mercado_lag2": "Tendencia del mes anterior",
    "mercado_lag3": "Tendencia de hace 3 meses",
    "mercado_ma3": "Promedio móvil de mercado (3m)",
    "puerto_freq": "Frecuencia del puerto de embarque",
    "importador_freq": "Perfil histórico del importador",
    "densidad_carga": "Densidad de la carga (kg/unidad)",
    "ratio_bruto_neto": "Ratio bruto/neto de la carga",
    "ruta_directa": "Tipo de ruta (directa vs. transbordo)",
}


_derivar_de_meta(_leer_meta())


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (singleton en memoria)
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_explainer = None
_model_lo = None
_model_hi = None


def modelos_vigentes() -> tuple:
    """Los CUATRO objetos del artifact leidos de una sola vez, bajo el cerrojo.

    H-25. `recargar_artifacts()` tomaba `_artifact_lock`, pero NINGUN lector lo
    tomaba. `predict()` obtenia (model, explainer) por `load_model()` y despues
    `_predict_one` leia `_model_lo` y `_model_hi` directamente del modulo, en
    otro momento: una recarga concurrente podia colar el modelo puntual NUEVO
    con los modelos de cuantiles VIEJOS —exactamente el artifact inconsistente
    que el orquestador de reentrenamiento existe para impedir—. El
    `ThreadPoolExecutor(max_workers=2)` de prediction_service hace que la
    concurrencia sea real, no teorica.

    Devolver los cuatro juntos, bajo el cerrojo, garantiza que una prediccion
    use una unica generacion del artifact de principio a fin.
    """
    global _model, _explainer, _model_lo, _model_hi
    with _artifact_lock:
        if _model is None:
            _model = joblib.load(MODEL_PATH)
            _explainer = shap.TreeExplainer(_model)
            _model_lo = joblib.load(QLO_PATH)
            _model_hi = joblib.load(QHI_PATH)
        return _model, _explainer, _model_lo, _model_hi


def load_model():
    """Compatibilidad: devuelve solo (modelo, explainer). Prefiera `modelos_vigentes()`."""
    m, ex, _lo, _hi = modelos_vigentes()
    return m, ex


def recargar_artifacts() -> dict:
    """Relee de disco los metadatos y los tres modelos, sin reiniciar el proceso.

    BUG 4. `load_model()` es un singleton y `_derivar_de_meta()` corria una sola
    vez al importar, asi que reentrenar con ml/train_model.py +
    ml/train_quantile_models.py no tenia efecto hasta reiniciar el servidor.

    La carga es ATOMICA POR DISENO: primero se leen y deserializan los cuatro
    ficheros en variables locales y solo si los cuatro cargan se publican en los
    globals. Un artifact a medio escribir (o un .pkl corrupto) aborta con
    excepcion dejando en memoria el artifact anterior, que sigue sirviendo
    predicciones validas.

    ORDEN DE REENTRENAMIENTO (no es intercambiable): train_model.py primero
    —genera modelo_meta.json con los encoders nuevos— y train_quantile_models.py
    despues, que lee esos encoders ya guardados. Recargar aqui con un meta nuevo
    y unos .pkl de cuantiles viejos daria intervalos incoherentes con el punto.

    Nota de despliegue: igual que ml/market_state.py, esto vive en la memoria
    del proceso. Con varios workers hay que llamar al endpoint una vez por
    worker o, mejor, reiniciar el servicio.
    """
    global _model, _explainer, _model_lo, _model_hi
    with _artifact_lock:
        meta = _leer_meta()
        m = joblib.load(MODEL_PATH)
        lo = joblib.load(QLO_PATH)
        hi = joblib.load(QHI_PATH)
        ex = shap.TreeExplainer(m)

        _derivar_de_meta(meta)
        _model, _explainer, _model_lo, _model_hi = m, ex, lo, hi

    # H-24: `market_state._state` se inicializaba UNA vez al importar el modulo
    # y no se tocaba al recargar. Tras reentrenar con un corpus mas largo, el
    # ultimo mes de la serie avanza pero los rezagos vigentes se quedan atras:
    # las cotizaciones futuras siguen sirviendose con el mercado viejo y
    # drift._senal_horizonte lo enmascara con su `max(0, ...)`, reportando "0
    # meses, no hay extrapolacion". Se vuelve a derivar de la serie nueva, salvo
    # que un administrador haya puesto un override manual vigente: ese es una
    # decision explicita y no se pisa.
    from ml import market_state
    if market_state.get_market_rates().get("origen") == "artifact":
        market_state.reset_market_rates()
    return info_artifact()


def info_artifact() -> dict:
    """Estado del artifact vigente en ESTE proceso.

    `entrenado_en` es la fecha de modificacion de modelo_meta.json: el artifact
    no guarda su propio timestamp de entrenamiento y el meta es el ultimo
    fichero que escribe train_model.py.
    """
    def _mtime(path: Path) -> Optional[str]:
        try:
            return datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
        except OSError:
            return None

    return {
        "entrenado_en": _mtime(META_PATH),
        "cargado": _model is not None,
        "mape_test": MODEL_MAPE,
        "n_puertos": len(PORT_FREQ),
        "n_importadores": len(IMPORTADOR_FREQ),
        "n_features": len(FEATURE_ORDER),
        "serie_mercado_primer_mes": _PRIMER_MES_SERIE,
        "serie_mercado_ultimo_mes": _ULTIMO_MES_SERIE,
        "archivos": {
            "modelo_meta.json": _mtime(META_PATH),
            "modelo_xgboost_flete.pkl": _mtime(MODEL_PATH),
            "modelo_xgboost_flete_q_lo.pkl": _mtime(QLO_PATH),
            "modelo_xgboost_flete_q_hi.pkl": _mtime(QHI_PATH),
        },
    }


def get_catalog_ports() -> list[dict]:
    """
    Retorna los puertos de embarque del dropdown (≥50 registros en el tramo de
    entrenamiento), ordenados alfabéticamente. Cada entrada tiene:
      - key:  clave MAYÚSCULAS del artifact → se envía al backend
      - name: Title Case                     → label visible en la UI
    Separar key de name garantiza que el predictor reciba exactamente la clave
    de puerto_freq sin normalización adicional.
    """
    return sorted(
        [{"key": p, "name": p.title()} for p in _CATALOG_PORTS],
        key=lambda x: x["name"],
    )


def get_catalog_importadores() -> list[dict]:
    """
    Retorna los importadores del dropdown (≥50 registros en entrenamiento, sin
    el bucket anonimizado por Ley 29733). Mismo contrato key/name.
    """
    return sorted(
        [{"key": i, "name": i.title()} for i in _CATALOG_IMPORTADORES],
        key=lambda x: x["name"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# REZAGOS DE MERCADO
# ──────────────────────────────────────────────────────────────────────────────

def _mes_anterior(anio: int, mes: int, k: int) -> str:
    """Clave 'YYYY-MM' del mes k posiciones antes de (anio, mes)."""
    total = anio * 12 + (mes - 1) - k
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _resolver_lags(ref_date: datetime) -> tuple[float, float, float, str, int, str]:
    """Rezagos de mercado correspondientes a una fecha de embarque.

    Devuelve (lag1, lag2, lag3, vigente_hasta, meses_extrapolados, direccion),
    donde `meses_extrapolados` es SIEMPRE >= 0 —mide distancia, no direccion— y
    `direccion` es 'ninguna' | 'adelante' | 'atras'. La direccion se devuelve
    explicitamente porque la advertencia al usuario depende de ella y antes se
    redactaba suponiendo siempre 'adelante'.

    Tres regimenes:

      1. DENTRO DEL HISTORICO — los tres meses anteriores existen en la serie.
         Se usan tal cual: es exactamente la definicion con la que se entreno.
         Verificado sobre las 8,613 filas de TEST, 0 discrepancias (la unica
         diferencia es el redondeo a 6 decimales con que la serie se guarda en
         el artifact: 5e-7 USD/kg, con impacto exactamente 0 en la prediccion).

      2. POSTERIOR AL HISTORICO — no hay nada que observar hacia adelante. Se
         usan los rezagos vigentes de `market_state`, actualizables en caliente.

      3. ANTERIOR O INCOMPLETO — la fecha es anterior al inicio de la serie, o
         cae en los primeros meses y no tiene los tres rezagos completos.

    CORRECCION DE LA TERCERA AUDITORIA (era un bug servido en produccion). La
    segunda auditoria creo la rama 3 con la condicion `objetivo <
    _PRIMER_MES_SERIE`. Esa condicion deja fuera los meses que SI tienen su
    lag1 en la serie pero no el lag2 o el lag3 — con una serie que empieza en
    2021-01, exactamente 2021-02 y 2021-03. Esas dos fechas caian en la rama 2
    (futuro), recibian los rezagos vigentes de 2025-12 y se presentaban al
    usuario como "N meses MAS ALLA del ultimo mercado observado" cuando estaban
    N meses ANTES. Medido: una cotizacion de 2021-02 se servia a 148 USD cuando
    el mercado real de la epoca implicaba 437 USD (2.95x), y una de 2021-03 a
    148 frente a 613 (4.14x). La condicion correcta no es "anterior al inicio de
    la serie" sino "no posterior al ultimo mes de la serie": todo lo que no es
    futuro y no tiene rezagos completos pertenece a la rama 3.
    """
    claves = [_mes_anterior(ref_date.year, ref_date.month, k) for k in (1, 2, 3)]
    if all(c in SERIE_MERCADO for c in claves):
        return (
            float(SERIE_MERCADO[claves[0]]),
            float(SERIE_MERCADO[claves[1]]),
            float(SERIE_MERCADO[claves[2]]),
            claves[0],
            0,
            "ninguna",
        )

    objetivo = claves[0]
    if objetivo > _ULTIMO_MES_SERIE:
        # Regimen 2: fecha posterior al historico. Rezagos vigentes.
        rates = get_market_rates()
        vigente = rates["vigente_hasta"]
        return (
            rates["lag1"], rates["lag2"], rates["lag3"], vigente,
            _dist_meses(objetivo, vigente), "adelante",
        )

    # Regimen 3: anterior al historico, o dentro de el pero sin los tres rezagos
    # completos. Se ancla en los primeros meses reales y se declara la distancia
    # HACIA ATRAS. La distancia puede ser 0 (caso 2021-02, cuyo lag1 si existe);
    # lo que hace falta advertir ahi no es la distancia sino que los rezagos
    # estan incompletos, y de eso se encarga `rezagos_incompletos`.
    anclas = _MESES_SERIE[:3]
    return (
        float(SERIE_MERCADO[anclas[0]]),
        float(SERIE_MERCADO[anclas[1]]),
        float(SERIE_MERCADO[anclas[2]]),
        _PRIMER_MES_SERIE,
        _dist_meses(objetivo, _PRIMER_MES_SERIE),
        "atras",
    )


def _dist_meses(a: str, b: str) -> int:
    """Distancia ABSOLUTA en meses entre dos claves 'YYYY-MM'.

    El valor absoluto es deliberado. El calculo anterior era
    `max(0, objetivo - vigente)`, que devolvia 0 para todo el pasado y apagaba
    la advertencia justo en el caso que mas la necesitaba. Una fecha 80 meses
    atras es tan poco fiable como una 80 meses adelante.
    """
    try:
        ay, am = int(a[:4]), int(a[5:7])
        by, bm = int(b[:4]), int(b[5:7])
    except (ValueError, IndexError):
        return 0
    return abs((ay * 12 + am) - (by * 12 + bm))


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def build_features(
    puerto_origen: str,
    tipo_contenedor: str,
    peso_kg: float,
    unidades: Optional[int],
    volumen_cbm: Optional[float],
    fecha_embarque: Optional[str],
    importador: Optional[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Transforma los inputs del formulario en el vector de features del modelo.

    Retorna (X, contexto), donde `contexto` reporta la vigencia de los rezagos
    de mercado y si el puerto/importador pedido existe en el histórico.
    """

    # Fecha de referencia
    # Una fecha no parseable ya no se sustituye por "hoy" en silencio. El schema
    # la valida antes (app/schemas/prediction.py usa un `date` real), pero si
    # llegara igual, el usuario tiene que enterarse de que su fecha se descarto.
    fecha_invalida = False
    if fecha_embarque:
        try:
            ref_date = datetime.strptime(fecha_embarque, "%Y-%m-%d")
        except ValueError:
            ref_date = datetime.now(timezone.utc)
            fecha_invalida = True
    else:
        ref_date = datetime.now(timezone.utc)

    mes = ref_date.month
    trimestre = (mes - 1) // 3 + 1
    semana_anio = ref_date.isocalendar()[1]
    mes_sin = math.sin(2 * math.pi * mes / 12)
    mes_cos = math.cos(2 * math.pi * mes / 12)

    lag1, lag2, lag3, vigente_hasta, extrapolados, direccion = _resolver_lags(ref_date)
    mercado_ma3 = (lag1 + lag2 + lag3) / 3

    # Frecuencia de puerto: normalizar a MAYÚSCULAS para coincidir con las claves
    # del artifact (PUER_DESC en SUNAT viene siempre en mayúsculas).
    puerto_key = puerto_origen.upper().strip()
    puerto_conocido = puerto_key in PORT_FREQ
    puerto_freq = PORT_FREQ.get(puerto_key, DEFAULT_PORT_FREQ)

    # ruta_directa: hecho geográfico determinístico del puerto de embarque.
    ruta_directa = RUTA_DIRECTA_POR_PUERTO.get(puerto_key, DEFAULT_RUTA_DIRECTA)

    # densidad_carga = PESO_NETO / UNID_FIQTY (kg/unidad), igual que en entrenamiento
    if unidades and unidades > 0:
        densidad_carga = peso_kg / unidades
    else:
        densidad_carga = MODEL_META["densidad_carga_median"]

    # ratio_bruto_neto: no hay peso bruto en el formulario; se imputa la mediana real
    ratio_bruto_neto = MODEL_META["ratio_bruto_neto_median"]

    # Frecuencia de importador: si no se especifica, o no está en el histórico
    # de entrenamiento, cae a la mediana real de train.
    importador_key = importador.upper().strip() if importador else None
    importador_conocido = bool(importador_key and importador_key in IMPORTADOR_FREQ)
    importador_freq = (
        IMPORTADOR_FREQ.get(importador_key, DEFAULT_IMPORTADOR_FREQ)
        if importador_key else DEFAULT_IMPORTADOR_FREQ
    )

    row = {
        "mes": mes,
        "trimestre": trimestre,
        "semana_anio": semana_anio,
        "mes_sin": mes_sin,
        "mes_cos": mes_cos,
        "mercado_lag1": lag1,
        "mercado_lag2": lag2,
        "mercado_lag3": lag3,
        "mercado_ma3": mercado_ma3,
        "puerto_freq": puerto_freq,
        "importador_freq": importador_freq,
        "densidad_carga": densidad_carga,
        "ratio_bruto_neto": ratio_bruto_neto,
        "ruta_directa": ruta_directa,
    }

    contexto = {
        "mercado_vigente_hasta": vigente_hasta,
        "meses_extrapolados": extrapolados,
        # `direccion` la decide _resolver_lags, que es quien sabe en que regimen
        # cayo la fecha. Derivarla aqui de nuevo fue el origen del bug de
        # 2021-02/2021-03: la condicion duplicada no coincidia con la de la rama.
        "direccion": direccion,
        "extrapola_hacia_atras": direccion == "atras",
        # Rezagos incompletos: la fecha esta dentro del rango de la serie pero
        # no tiene los tres meses previos. Debe advertirse aunque la distancia
        # sea 0 (caso 2021-02).
        "rezagos_incompletos": direccion == "atras",
        "puerto_en_historico": puerto_conocido,
        "importador_en_historico": importador_conocido,
        "fecha_invalida": fecha_invalida,
        "importador_especificado": bool(importador_key),
    }
    return pd.DataFrame([row])[FEATURE_ORDER], contexto


def _construir_advertencia(ctx: dict) -> Optional[str]:
    """Mensaje al usuario cuando la cotización se apoya en datos débiles."""
    avisos = []
    if ctx.get("fecha_invalida"):
        avisos.append(
            "La fecha de embarque enviada no es una fecha valida y se descarto: "
            "la estimacion se calculo para la fecha de hoy."
        )
    if ctx.get("rezagos_incompletos"):
        _d = ctx["meses_extrapolados"]
        _cuando = (
            f"unos {_d} meses antes del inicio de la serie de mercado "
            f"({ctx['mercado_vigente_hasta']})"
            if _d else
            f"dentro de los primeros meses de la serie ({ctx['mercado_vigente_hasta']})"
        )
        avisos.append(
            f"La fecha solicitada cae {_cuando}, donde el modelo no dispone de los "
            "tres meses de mercado previos que necesita. Se anclan los primeros "
            "meses observados: la estimacion NO es representativa de esa fecha y "
            "no debe usarse para cotizar."
        )
    elif ctx["meses_extrapolados"] > MAX_MESES_EXTRAPOLACION:
        avisos.append(
            f"La fecha solicitada está {ctx['meses_extrapolados']} meses más allá "
            f"del último mercado observado ({ctx['mercado_vigente_hasta']}). Las "
            f"variables de mercado explican ~89% del modelo, así que esta "
            f"estimación es indicativa: actualice las tasas de mercado para "
            f"cotizaciones de largo plazo."
        )
    elif ctx["meses_extrapolados"] > 0:
        avisos.append(
            f"Estimación proyectada {ctx['meses_extrapolados']} mes(es) más allá "
            f"del último mercado observado ({ctx['mercado_vigente_hasta']}). El error "
            f"esperado en este régimen es ~{MAPE_EXTRAPOLADO:.1f}%, no el "
            f"{MODEL_MAPE:.1f}% medido dentro del histórico."
        )
    if not ctx.get("ic95_calibrado", True):
        avisos.append(
            f"El intervalo de confianza esta calibrado para extrapolaciones de "
            f"hasta {HORIZONTE_IC_CALIBRADO} meses; esta cotizacion esta a "
            f"{ctx['meses_extrapolados']}. El intervalo se muestra igualmente, "
            "pero su cobertura del 95% NO esta garantizada en este horizonte."
        )
    if not ctx["puerto_en_historico"]:
        avisos.append("El puerto seleccionado no figura en el histórico de entrenamiento.")
    if ctx.get("importador_especificado") and not ctx["importador_en_historico"]:
        avisos.append("El importador no figura en el histórico: se usa el perfil promedio.")
    return " ".join(avisos) if avisos else None


# ──────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def _predict_one(
    model, explainer, X: pd.DataFrame, peso_kg: float,
    extrapolado: bool = False, meses_extrapolados: int = 0,
    model_lo=None, model_hi=None,
) -> tuple[float, float, float, np.ndarray]:
    """Predicción individual: (flete_total_usd, ic_min_usd, ic_max_usd, shap_usd).

    El IC95 se construye con Conformalized Quantile Regression: dos modelos
    XGBoost que predicen los percentiles 2.5% y 97.5% de FLETE_UNIT, corregidos
    por una constante Q calibrada sobre un tramo de validación nunca visto por
    ningún modelo.

    Q SEGÚN EL HORIZONTE (corrección de la tercera auditoría). La segunda
    auditoría introdujo una segunda constante para el régimen extrapolado, lo
    cual era un avance, pero la calibró a ~3 meses de horizonte y la servía para
    fechas de hasta 60: su cobertura real caía al 18% a nueve meses mientras la
    respuesta la etiquetaba "IC 95%". Ahora se usa la tabla `Q(h)`, monotonizada
    por máximo acumulado, que sostiene ≥95% de cobertura condicional en cada
    horizonte estimable. Más allá del último horizonte con datos se aplica la
    entrada más ancha y la respuesta declara `ic95_calibrado = False`.

    El ancho sigue variando poco ENTRE CASOS (≈8% entre segmentos de
    representatividad del importador): el intervalo distingue el régimen temporal,
    pero NO distingue un puerto conocido de uno desconocido. Es una limitación
    declarada, no una adaptación fina que se pueda presentar como precisión.
    """
    q_const = _q_conformal(meses_extrapolados, extrapolado)
    flete_unit = float(model.predict(X)[0])
    flete_unit = max(flete_unit, 0.01)  # evitar negativos
    flete_total = flete_unit * peso_kg

    # H-25: los modelos de cuantiles llegan por parametro, de la MISMA lectura
    # bajo cerrojo que produjo `model`. Leerlos aqui del modulo permitia mezclar
    # generaciones distintas del artifact durante una recarga en caliente.
    _lo = model_lo if model_lo is not None else _model_lo
    _hi = model_hi if model_hi is not None else _model_hi
    q_lo_unit = float(_lo.predict(X)[0]) - q_const
    q_hi_unit = float(_hi.predict(X)[0]) + q_const
    # El recorte en 0 es FISICO, no estadistico: un flete no puede ser negativo.
    # Con las Q de horizonte largo el limite inferior conformal si cae por debajo
    # de cero, y recortarlo estrecha el intervalo respecto del que la calibracion
    # produjo. Es la unica direccion en que se le toca, y se hace porque el
    # espacio de valores posibles esta acotado por abajo, no para que el numero
    # quede mejor. Un limite inferior de 0 debe leerse como "el modelo no puede
    # descartar un flete muy bajo a ese horizonte", que es cierto.
    ic_min = max(q_lo_unit, 0.0) * peso_kg
    ic_max = max(q_hi_unit, 0.0) * peso_kg
    # Salvaguarda: el intervalo siempre debe contener al punto estimado.
    ic_min = min(ic_min, flete_total)
    ic_max = max(ic_max, flete_total)

    shap_values = explainer.shap_values(X)
    shap_row = shap_values[0] if len(shap_values.shape) == 2 else shap_values
    shap_usd = np.array(shap_row) * peso_kg  # escalar a USD total

    return flete_total, ic_min, ic_max, shap_usd


def _build_result(
    flete_total: float, ic_min: float, ic_max: float,
    shap_usd: np.ndarray, t0: float, ctx: dict,
) -> dict:
    """Construye la respuesta final (IC95 vía CQR + top-3 SHAP + contexto)."""
    indices = np.argsort(np.abs(shap_usd))[::-1][:3]
    contribuciones = []
    for idx in indices:
        fname = FEATURE_ORDER[idx]
        val = float(shap_usd[idx])
        contribuciones.append({
            "variable": FEATURE_LABELS.get(fname, fname),
            "aporte": round(val, 2),
            "direction": "positive" if val >= 0 else "negative",
        })

    # MAPE del régimen en el que REALMENTE se sirvió esta predicción. Devolver
    # siempre el MAPE de test (~22%) para una cotización extrapolada, cuyo error
    # esperado es el del escenario congelado (~27.5%), era mostrarle al usuario
    # una precisión que esa cotización no tiene.
    extrapolado = ctx["meses_extrapolados"] > 0 or ctx.get("rezagos_incompletos", False)
    # El IC solo esta calibrado dentro del historico o hasta HORIZONTE_IC_CALIBRADO
    # meses de extrapolacion hacia adelante. Fuera de ahi se sirve igualmente pero
    # se declara: es la diferencia entre un intervalo ancho y un intervalo falso.
    ctx["ic95_calibrado"] = bool(
        not extrapolado
        or (ctx.get("direccion") == "adelante"
            and ctx["meses_extrapolados"] <= HORIZONTE_IC_CALIBRADO)
    )
    return {
        "flete_estimado_usd": round(flete_total, 2),
        "ic95_min": round(ic_min, 2),
        "ic95_max": round(ic_max, 2),
        "ic95_calibrado": ctx["ic95_calibrado"],
        "ic95_horizonte_calibrado_meses": HORIZONTE_IC_CALIBRADO,
        "mape_modelo": MAPE_EXTRAPOLADO if extrapolado else MODEL_MAPE,
        "mape_regimen": "extrapolado" if extrapolado else "historico",
        "tiempo_ms": int((time.monotonic() - t0) * 1000),
        "shap_contribuciones": contribuciones,
        "mercado_vigente_hasta": ctx["mercado_vigente_hasta"],
        "meses_extrapolados": ctx["meses_extrapolados"],
        "advertencia": _construir_advertencia(ctx),
    }


def _resolve_year(fecha_embarque: Optional[str]) -> int:
    """Extrae el año de la fecha o usa el año actual."""
    if fecha_embarque:
        try:
            return datetime.strptime(fecha_embarque, "%Y-%m-%d").year
        except ValueError:
            pass
    return datetime.now(timezone.utc).year


def predict(
    puerto_origen: str,
    tipo_contenedor: str,
    peso_kg: float,
    unidades: Optional[int] = None,
    volumen_cbm: Optional[float] = None,
    fecha_embarque: Optional[str] = None,
    periodo: Optional[str] = None,
    importador: Optional[str] = None,
) -> dict:
    """
    Retorna:
      - flete_estimado_usd: predicción total en USD
      - ic95_min / ic95_max: intervalo de confianza 95% (CQR)
      - mape_modelo: MAPE del modelo en test
      - tiempo_ms: latencia de inferencia
      - shap_contribuciones: top-3 variables con impacto en negocio
      - mercado_vigente_hasta / meses_extrapolados / advertencia: vigencia de
        las variables de mercado, que concentran ~89% del gain del modelo

    Si periodo == "anual", promedia 12 predicciones (una por mes del año
    indicado) para representar el flete de todo el año.

    LIMITACIÓN DEL IC95 ANUAL. Promediar los 12 ic_min/ic_max mensuales es una
    aproximación simple y transparente, consistente con cómo se promedia el
    punto estimado — pero NO es una recalibración conformal conjunta y el
    intervalo resultante no tiene una cobertura garantizada para el agregado
    anual. TERCERA AUDITORÍA: además, un año puede MEZCLAR regímenes — en 2026,
    enero es histórico (sus tres rezagos están observados) y febrero-diciembre
    extrapolan —, así que cada mes aporta un intervalo construido con una `Q`
    distinta de la tabla `Q(h)`. Eso es correcto mes a mes, y es mejor que
    aplicar una constante única, pero el promedio de doce intervalos calibrados
    a horizontes distintos no hereda ninguna de esas garantías. Además, cuando el año pedido cae fuera del histórico, los 12 meses
    comparten los mismos rezagos de mercado, así que las 12 predicciones
    difieren solo por estacionalidad (~6% del gain): el promedio no agrega
    información, solo suaviza. Debe leerse como una referencia orientativa.
    """
    # H-25 (segunda mitad). `_derivar_de_meta()` reasigna ~20 globals uno a uno
    # —FEATURE_ORDER, PORT_FREQ, SERIE_MERCADO, CONFORMAL_Q*...— y los lectores
    # (`build_features`, `_q_conformal`, `_build_result`) no tomaban ningun
    # cerrojo. Una recarga concurrente podia servir una prediccion con encoders
    # nuevos y Q conformal vieja. Sostener el cerrojo durante toda la prediccion
    # es lo unico que garantiza que se use UNA sola generacion del artifact de
    # principio a fin.
    #
    # COSTE ACEPTADO: serializa las predicciones. Con ~70 ms por prediccion y un
    # `ThreadPoolExecutor(max_workers=2)`, el limite practico no cambia; el
    # cerrojo es reentrante, asi que `modelos_vigentes()` anida sin bloquearse.
    # Para un perfil de carga mayor habria que usar un lock de lectura/escritura
    # o publicar el estado derivado como un unico objeto inmutable.
    with _artifact_lock:
        return _predict_bajo_cerrojo(
            puerto_origen, tipo_contenedor, peso_kg, unidades, volumen_cbm,
            fecha_embarque, periodo, importador,
        )


def _predict_bajo_cerrojo(
    puerto_origen: str,
    tipo_contenedor: str,
    peso_kg: float,
    unidades: Optional[int] = None,
    volumen_cbm: Optional[float] = None,
    fecha_embarque: Optional[str] = None,
    periodo: Optional[str] = None,
    importador: Optional[str] = None,
) -> dict:
    """Cuerpo real de `predict()`. Se ejecuta con `_artifact_lock` tomado."""
    model, explainer, model_lo, model_hi = modelos_vigentes()
    t0 = time.monotonic()

    if periodo == "anual":
        year = _resolve_year(fecha_embarque)
        totales: list[float] = []
        ic_mins: list[float] = []
        ic_maxs: list[float] = []
        shap_acc = np.zeros(len(FEATURE_ORDER))
        ctx: dict = {}
        for mes in range(1, 13):
            X, ctx_mes = build_features(
                puerto_origen, tipo_contenedor, peso_kg, unidades,
                volumen_cbm, f"{year}-{mes:02d}-15", importador,
            )
            ft, ic_lo, ic_hi, su = _predict_one(
                model, explainer, X, peso_kg,
                extrapolado=ctx_mes["meses_extrapolados"] > 0 or ctx_mes["rezagos_incompletos"],
                meses_extrapolados=ctx_mes["meses_extrapolados"],
                model_lo=model_lo, model_hi=model_hi,
            )
            totales.append(ft)
            ic_mins.append(ic_lo)
            ic_maxs.append(ic_hi)
            shap_acc += su
            # Se reporta el peor caso de extrapolación del año (diciembre).
            if not ctx or ctx_mes["meses_extrapolados"] >= ctx["meses_extrapolados"]:
                ctx = ctx_mes
        flete_total = sum(totales) / len(totales)
        ic_min = sum(ic_mins) / len(ic_mins)
        ic_max = sum(ic_maxs) / len(ic_maxs)
        shap_usd = shap_acc / 12
    else:
        X, ctx = build_features(
            puerto_origen, tipo_contenedor, peso_kg, unidades,
            volumen_cbm, fecha_embarque, importador,
        )
        flete_total, ic_min, ic_max, shap_usd = _predict_one(
            model, explainer, X, peso_kg,
            extrapolado=ctx["meses_extrapolados"] > 0 or ctx["rezagos_incompletos"],
            meses_extrapolados=ctx["meses_extrapolados"],
            model_lo=model_lo, model_hi=model_hi,
        )

    return _build_result(flete_total, ic_min, ic_max, shap_usd, t0, ctx)

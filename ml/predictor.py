"""
Predictor de flete marítimo usando el modelo XGBoost entrenado.
Transforma los inputs del formulario en las features del modelo,
ejecuta la predicción, genera intervalos de confianza 95% y
calcula los top-3 SHAP values con etiquetas de negocio.
"""

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import warnings

import joblib
import numpy as np
import pandas as pd
import shap

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL ARTIFACT (modelo + metadatos)
# ──────────────────────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "modelo_xgboost_flete.pkl"
META_PATH  = MODEL_DIR / "modelo_meta.json"

with open(META_PATH, encoding="utf-8") as _f:
    MODEL_META: dict = json.load(_f)

# MAPE del modelo en test
MODEL_MAPE: float = MODEL_META["metricas_test"]["MAPE_%"]

# Frecuencias de puerto derivadas del dataset de entrenamiento (claves en MAYÚSCULAS).
# Fuente: fe['PUER_DESC'].value_counts(normalize=True) — resultado_combinado.csv 2021-2025.
# Los 65 puertos reales se usan para el lookup del modelo; el dropdown solo muestra
# los puertos en puertos_dropdown (asiáticos con ≥50 registros).
PORT_FREQ: dict[str, float] = MODEL_META["puerto_freq"]

# Frecuencia mínima observada en entrenamiento → fallback para puertos no vistos.
_PORT_FREQ_MIN: float = min(PORT_FREQ.values())

# Puertos curados para el dropdown: asiáticos con ≥50 registros (≈99.8% del volumen).
_CATALOG_PORTS: list[str] = MODEL_META["puertos_dropdown"]

# Frecuencia media de importador (mediana del entrenamiento, sin dato del formulario)
DEFAULT_IMPORTADOR_FREQ = 0.05


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
    "puerto_freq": "Frecuencia del puerto de origen",
    "importador_freq": "Perfil histórico del importador",
    "densidad_carga": "Densidad de la carga (kg/unidad)",
    "ratio_bruto_neto": "Ratio bruto/neto de la carga",
}

FEATURE_ORDER = [
    "mes", "trimestre", "semana_anio",
    "mes_sin", "mes_cos",
    "mercado_lag1", "mercado_lag2", "mercado_lag3", "mercado_ma3",
    "puerto_freq", "importador_freq",
    "densidad_carga", "ratio_bruto_neto",
]

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DEL MODELO (singleton en memoria)
# ──────────────────────────────────────────────────────────────────────────────

_model = None
_explainer = None


def load_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def get_catalog_ports() -> list[dict]:
    """
    Retorna los puertos del dropdown (asiáticos, ≥50 registros), ordenados alfabéticamente.
    Cada entrada tiene:
      - key:  clave MAYÚSCULAS del JSON  → se envía al backend como value del <option>
      - name: Title Case para mostrar    → label visible en la UI
    Separar key de name garantiza que el predictor reciba exactamente la clave
    del puerto_freq sin normalización adicional (Option A).
    """
    return sorted(
        [{"key": p, "name": p.title()} for p in _CATALOG_PORTS],
        key=lambda x: x["name"],
    )


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
) -> pd.DataFrame:
    """Transforma los inputs del formulario en el vector de features del modelo."""

    # Fecha de referencia
    if fecha_embarque:
        try:
            ref_date = datetime.strptime(fecha_embarque, "%Y-%m-%d")
        except ValueError:
            ref_date = datetime.now(timezone.utc)
    else:
        ref_date = datetime.now(timezone.utc)

    mes = ref_date.month
    trimestre = (mes - 1) // 3 + 1
    semana_anio = ref_date.isocalendar()[1]
    mes_sin = math.sin(2 * math.pi * mes / 12)
    mes_cos = math.cos(2 * math.pi * mes / 12)

    # Lag features de mercado: 3 últimos meses REALES de serie_mercado (no se proyectan)
    _serie = MODEL_META["serie_mercado"]
    _keys = sorted(_serie.keys())  # YYYY-MM → lexicográfico = cronológico
    _n = len(_keys)
    if _n >= 3:
        _recent = _keys[-3:]
    else:
        warnings.warn(
            f"serie_mercado tiene solo {_n} meses; completando con el más antiguo",
            stacklevel=2,
        )
        _recent = [_keys[0]] * (3 - _n) + _keys
    lag1 = _serie[_recent[2]]   # mes más reciente
    lag2 = _serie[_recent[1]]   # penúltimo
    lag3 = _serie[_recent[0]]   # antepenúltimo
    mercado_ma3 = (lag1 + lag2 + lag3) / 3

    # Frecuencia de puerto: normalizar a MAYÚSCULAS para coincidir con las claves del
    # artifact (que provienen de PUER_DESC en SUNAT, siempre en mayúsculas).
    # Fallback = frecuencia mínima observada en entrenamiento.
    puerto_freq = PORT_FREQ.get(
        puerto_origen.upper().strip(),
        _PORT_FREQ_MIN,
    )

    # densidad_carga = PESO_NETO / UNID_FIQTY (kg/unidad), igual que en entrenamiento
    if unidades and unidades > 0:
        densidad_carga = peso_kg / unidades
    else:
        densidad_carga = MODEL_META["densidad_carga_median"]

    # ratio_bruto_neto: no hay peso bruto en el formulario; se imputa con la mediana real
    ratio_bruto_neto = MODEL_META["ratio_bruto_neto_median"]

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
        "importador_freq": DEFAULT_IMPORTADOR_FREQ,
        "densidad_carga": densidad_carga,
        "ratio_bruto_neto": ratio_bruto_neto,
    }

    return pd.DataFrame([row])[FEATURE_ORDER]


# ──────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def predict(
    puerto_origen: str,
    tipo_contenedor: str,
    peso_kg: float,
    unidades: Optional[int] = None,
    volumen_cbm: Optional[float] = None,
    fecha_embarque: Optional[str] = None,
) -> dict:
    """
    Retorna:
      - flete_estimado_usd: predicción total en USD
      - ic95_min / ic95_max: intervalo de confianza 95%
      - mape_modelo: MAPE conocido del modelo
      - tiempo_ms: latencia de inferencia
      - shap_contribuciones: top-3 variables con impacto en negocio
    """
    model, explainer = load_model()

    t0 = time.monotonic()

    X = build_features(puerto_origen, tipo_contenedor, peso_kg, unidades, volumen_cbm, fecha_embarque)

    # Predicción de flete unitario (USD/kg), luego multiplicar por peso
    flete_unit = float(model.predict(X)[0])
    flete_unit = max(flete_unit, 0.01)  # evitar negativos
    flete_total = flete_unit * peso_kg

    # Intervalo de confianza 95% basado en el MAPE del modelo
    margen = 1.96 * (MODEL_MAPE / 100) * flete_total
    ic_min = max(flete_total - margen, 0.0)
    ic_max = flete_total + margen

    # SHAP values
    shap_values = explainer.shap_values(X)
    shap_row = shap_values[0] if len(shap_values.shape) == 2 else shap_values

    # Escalar SHAP a USD total (multiplicar por peso)
    shap_usd = np.array(shap_row) * peso_kg

    # Top 3 por valor absoluto
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

    tiempo_ms = int((time.monotonic() - t0) * 1000)

    return {
        "flete_estimado_usd": round(flete_total, 2),
        "ic95_min": round(ic_min, 2),
        "ic95_max": round(ic_max, 2),
        "mape_modelo": MODEL_MAPE,
        "tiempo_ms": tiempo_ms,
        "shap_contribuciones": contribuciones,
    }

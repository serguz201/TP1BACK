import json
from datetime import date
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────────────────────
# RANGO DE FECHAS ACEPTADO — DERIVADO DEL ARTIFACT, NO HARDCODEADO
#
# TERCERA AUDITORIA. Estas dos fechas estaban fijas en el codigo (2021-01-01 y
# 2030-12-31) mientras la serie real vive en `modelo_meta.json`. Que coincidieran
# era una casualidad del entrenamiento actual: al reentrenar con datos que
# empiecen en otro mes, el schema habria seguido aceptando fechas para las que el
# modelo no tiene mercado observado, alimentando precisamente el bug de rezagos
# incompletos que esta ronda corrigio en `ml/predictor.py`. Ahora se derivan.
#
#   FECHA_MIN = primer mes de la serie + 3 meses. Es el primer mes que tiene sus
#               tres rezagos completos, es decir el primero que el modelo sabe
#               representar. Coincide con el inicio real de TRAIN.
#   FECHA_MAX = ultimo mes de la serie + MESES_HORIZONTE_MAX. Mas alla de ahi la
#               extrapolacion deja de ser informativa.
#
# El intervalo de confianza, ademas, solo esta CALIBRADO hasta
# `ic95_horizonte_calibrado_meses` de extrapolacion. Entre ese horizonte y
# FECHA_MAX la peticion se acepta y se responde, pero con `ic95_calibrado=False`.
# ──────────────────────────────────────────────────────────────────────────────
MESES_HORIZONTE_MAX = 60

_META = json.loads(
    (Path(__file__).resolve().parents[2] / "ml" / "modelo_meta.json").read_text(
        encoding="utf-8"
    )
)


def _sumar_meses(anio: int, mes: int, k: int) -> tuple[int, int]:
    total = anio * 12 + (mes - 1) + k
    return total // 12, total % 12 + 1


def _ultimo_dia(anio: int, mes: int) -> int:
    y, m = _sumar_meses(anio, mes, 1)
    return (date(y, m, 1) - date(anio, mes, 1)).days


_PRIMER = _META["serie_mercado_primer_mes"]
_ULTIMO = _META["serie_mercado_ultimo_mes"]
_y, _m = _sumar_meses(int(_PRIMER[:4]), int(_PRIMER[5:7]), 3)
FECHA_MIN = date(_y, _m, 1)
_y, _m = _sumar_meses(int(_ULTIMO[:4]), int(_ULTIMO[5:7]), MESES_HORIZONTE_MAX)
FECHA_MAX = date(_y, _m, _ultimo_dia(_y, _m))

# Horizonte hasta el que el IC95 esta calibrado (informativo para la UI).
HORIZONTE_IC_CALIBRADO = _META.get("ic95_horizonte_calibrado_meses", 6)


class PredictionRequest(BaseModel):
    # puerto_origen: es el puerto de EMBARQUE (el origen siempre es China; Busan,
    # Yokohama o Hong Kong son puertos de transbordo). El nombre del campo se
    # conserva por compatibilidad de la API ya desplegada.
    puerto_origen: str = Field(..., min_length=1, max_length=100)
    peso_kg: float = Field(..., gt=0)
    unidades: Optional[int] = Field(None, gt=0)
    # SEGUNDA AUDITORÍA: el patrón `^\d{4}-\d{2}-\d{2}$` aceptaba "2025-13-45",
    # que el predictor no podía parsear y sustituía en silencio por la fecha de
    # hoy. El usuario recibía una cotización para otra fecha sin saberlo. Ahora
    # se valida como fecha real y acotada al rango que el modelo puede sostener.
    fecha_embarque: Optional[str] = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        examples=["2026-03-15"],
        description=f"Fecha de embarque YYYY-MM-DD, entre {FECHA_MIN} y {FECHA_MAX}.",
    )
    # periodo: 'semanal' | 'mensual' | 'anual'. En 'anual' se promedian los 12 meses.
    periodo: Optional[str] = Field(None, pattern=r"^(semanal|mensual|anual)$")
    # tipo_contenedor y volumen_cbm no son features del modelo; se aceptan por
    # compatibilidad pero son opcionales y se ignoran en la predicción.
    tipo_contenedor: Optional[str] = Field(None, max_length=50)
    volumen_cbm: Optional[float] = Field(None, gt=0)
    # Importador real de la operación (opcional). Si no se especifica, o no está
    # en el histórico de entrenamiento, el modelo usa la mediana real como fallback.
    importador: Optional[str] = Field(None, max_length=200)

    @field_validator("fecha_embarque")
    @classmethod
    def _fecha_real_y_en_rango(cls, v: Optional[str]) -> Optional[str]:
        """Rechaza fechas imposibles y fuera del horizonte del modelo."""
        if v is None:
            return v
        try:
            f = date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(
                f"'{v}' no es una fecha válida. Use el formato YYYY-MM-DD con un "
                "mes entre 01 y 12 y un día que exista en ese mes."
            ) from exc
        if not FECHA_MIN <= f <= FECHA_MAX:
            raise ValueError(
                f"La fecha {v} está fuera del rango que el modelo puede sostener "
                f"({FECHA_MIN} a {FECHA_MAX}). Antes de {FECHA_MIN} no hay serie de "
                "mercado observada; más allá de "
                f"{FECHA_MAX} la extrapolación deja de ser informativa."
            )
        return v


class SHAPContribution(BaseModel):
    variable: str
    aporte: float
    direction: str


class PredictionResponse(BaseModel):
    flete_estimado_usd: float
    ic95_min: float
    ic95_max: float
    # MAPE del RÉGIMEN en que se sirvió esta predicción, no siempre el de test.
    # Una cotización extrapolada tiene el error del escenario de mercado
    # congelado (~27.5%), no el de test (~22.2%); mostrar siempre el segundo era
    # atribuirle a la cotización una precisión que no tiene.
    mape_modelo: float
    # ¿El IC95 esta calibrado para esta peticion? La constante conformal del
    # regimen extrapolado se calibra agrupando sobre horizontes de 1 a
    # `ic95_horizonte_calibrado_meses`. Mas alla se devuelve el intervalo pero su
    # cobertura del 95% no esta garantizada, y el sistema debe decirlo en vez de
    # presentar un intervalo con una confianza que no tiene.
    ic95_calibrado: bool = True
    ic95_horizonte_calibrado_meses: int = HORIZONTE_IC_CALIBRADO
    mape_regimen: str = Field(
        "historico",
        description="'historico' (fecha dentro de la serie observada) o "
                    "'extrapolado' (mercado congelado; error esperado mayor).",
    )
    tiempo_ms: int
    shap_contribuciones: list[SHAPContribution]
    # Vigencia de las variables de mercado, que concentran ~89% del gain del
    # modelo. Una cotizacion muy alejada del ultimo mes observado es
    # estructuralmente fragil y el sistema debe declararlo.
    mercado_vigente_hasta: str
    meses_extrapolados: int
    advertencia: Optional[str] = None
